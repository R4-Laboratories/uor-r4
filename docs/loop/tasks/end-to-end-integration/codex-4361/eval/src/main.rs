#![allow(dead_code)]
mod calibrated {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../codex-4360/runner/src/bin/device.rs"
    ));

    use std::collections::{BTreeMap, BTreeSet};
    use uor_r4_tokenizer::ByteBpeTokenizer;

    #[derive(Deserialize)]
    struct Article {
        id: u64,
        stratum: String,
        question_sha256: String,
        #[serde(flatten)]
        row: Row,
    }

    fn checked_input(path: &str, pin: &str) -> Result<Vec<u8>> {
        let bytes = fs::read(path)?;
        if hex::encode(Sha256::digest(&bytes)) != pin {
            return Err(format!("input digest mismatch {path}").into());
        }
        Ok(bytes)
    }

    fn load_articles(bytes: &[u8]) -> Result<Vec<Article>> {
        let mut rows = Vec::new();
        for line in bytes.lines() {
            let article: Article = serde_json::from_str(&line?)?;
            let row = &article.row;
            if article.id != (rows.len() + 1) as u64
                || !["near", "far", "intermediate"].contains(&article.stratum.as_str())
                || article.question_sha256.len() != 64
                || !article
                    .question_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit())
                || row.prefix.len() < 2
                || row.prefix.first() != Some(&0)
                || row.prefix[1..].contains(&0)
                || row.prefix.iter().filter(|&&id| id == 1).count() != 2
                || row.expected.len() < 2
                || row.prefix.len() + row.expected.len() > 1088
                || row.expected.last() != Some(&1)
                || row.expected[..row.expected.len() - 1]
                    .iter()
                    .any(|&id| id <= 1)
                || row.prefix.iter().chain(&row.expected).any(|&id| id >= 4096)
            {
                return Err(format!("invalid article {}", article.id).into());
            }
            rows.push(article);
        }
        if rows.len() < 2 {
            return Err("need at least two articles".into());
        }
        Ok(rows)
    }

    // Pair only disjoint members of a literal question bucket for the directional
    // endpoint. For other articles use a deterministic different passage control.
    fn donors(articles: &[Article]) -> (Vec<usize>, BTreeSet<usize>) {
        let mut groups = BTreeMap::<&str, Vec<usize>>::new();
        for (i, article) in articles.iter().enumerate() {
            groups.entry(&article.question_sha256).or_default().push(i);
        }
        let mut donors: Vec<_> = (0..articles.len())
            .map(|i| (i + 1) % articles.len())
            .collect();
        let mut matched = BTreeSet::new();
        for group in groups.values() {
            for &a in group {
                if matched.contains(&a) {
                    continue;
                }
                if let Some(&b) = group.iter().find(|&&b| {
                    b > a
                        && !matched.contains(&b)
                        && articles[a].row.expected[0] != articles[b].row.expected[0]
                }) {
                    donors[a] = b;
                    donors[b] = a;
                    matched.insert(a);
                    matched.insert(b);
                }
            }
        }
        (donors, matched)
    }

    fn evidence(arm: &Arm, reference: &[u16], tok: &ByteBpeTokenizer) -> Value {
        let bytes =
            |ids: &[u16]| tok.decode_bytes(&ids.iter().copied().map(u32::from).collect::<Vec<_>>());
        let content = arm.emitted.strip_suffix(&[1]).unwrap_or(&arm.emitted);
        let actual = bytes(content);
        let gold = bytes(&reference[..reference.len() - 1]);
        json!({"emitted":arm.emitted,"first":arm.first.greedy_token,"cell":arm.first.greedy_coarse,
            "coarse_winner_tied":arm.first.coarse_scores.iter().filter(|&&v|v==arm.first.coarse_scores[arm.first.greedy_coarse]).count()>1,"own":arm.own,"other":arm.other,"stop":arm.stop_reason,"first_wrong":arm.first_error,
            "exact_ids_eos":arm.exact, "exact_bytes_eos":arm.hit_eos && actual.trim_ascii()==gold.trim_ascii(),
            "hidden_sha256":hex::encode(Sha256::digest(arm.first.hidden_q15.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()))})
    }

    pub fn evaluate(args: &[String]) -> Result<()> {
        let [kernel, ksha, model_path, msha, tape, tsha, tokenizer_path, output] = args else {
            return Err("evaluate CUBIN SHA MODEL SHA TAPE SHA TOKENIZER FRESH_JSONL".into());
        };
        checked_input(kernel, ksha)?;
        checked_input(model_path, msha)?;
        let (model, engine, _, _) = load_engine(kernel, model_path)?;
        let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked_input(
            tokenizer_path,
            "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
        )?)
        .ok_or("invalid tokenizer")?;
        let articles = load_articles(&checked_input(tape, tsha)?)?;
        let states: Vec<_> = articles
            .iter()
            .map(|article| prefill(&engine, &model, &article.row))
            .collect::<Result<_>>()?;
        let (donors, matched) = donors(&articles);
        let mut file = BufWriter::new(
            fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(output)?,
        );
        let mut checks = Verification {
            engine: &engine,
            model: &model,
            eos: [1; 2],
            cap: 32,
            digest: Sha256::new(),
            transitions: 0,
        };
        for (i, article) in articles.iter().enumerate() {
            let j = donors[i];
            let mut record = json!({"id":article.id,"stratum":article.stratum,"donor_id":articles[j].id,
                "matched":matched.contains(&i), "reference":article.row.expected, "model_sha256":msha, "tape_sha256":tsha, "cubin_sha256":ksha});
            let cpu_start = if i < 3 {
                Some(rust_prefill(&model, &article.row)?)
            } else {
                None
            };
            let cpu_donor = if i < 3 {
                Some(rust_prefill(&model, &articles[j].row)?)
            } else {
                None
            };
            for name in ["native", "wrong", "zero"] {
                let mut state = states[i].clone();
                let mut cpu = cpu_start.clone();
                if name == "wrong" {
                    state.carry.clone_from(&states[j].carry);
                    if let (Some(cpu), Some(donor)) = (&mut cpu, &cpu_donor) {
                        cpu.carry.clone_from(&donor.carry);
                    }
                } else if name == "zero" {
                    for field in &mut state.carry {
                        field.fill(0);
                    }
                    if let Some(cpu) = &mut cpu {
                        for field in &mut cpu.carry {
                            field.fill(0);
                        }
                    }
                }
                // Establish the actual trajectory without giving the kernel any reference ID.
                let probe = Row {
                    prefix: vec![*article.row.prefix.last().ok_or("no final input")?],
                    expected: vec![1],
                };
                let blinded = arm(&engine, &model, state.clone(), &probe, &probe, [1; 2], 32)?;
                let result = if let Some(s) = cpu {
                    checks.arm(&article.row, &articles[j].row, s, state)?
                } else {
                    arm(
                        &engine,
                        &model,
                        state,
                        &article.row,
                        &articles[j].row,
                        [1; 2],
                        32,
                    )?
                };
                if blinded.first != result.first
                    || blinded.emitted != result.emitted
                    || blinded.stop_reason != result.stop_reason
                {
                    return Err("reference-free generation differs from diagnostic replay".into());
                }
                record[name] = evidence(&result, &article.row.expected, &tok);
            }
            writeln!(file, "{}", record)?;
        }
        file.flush()?;
        println!(
            "{}",
            json!({"gpu":engine.gpu_name,"model":msha,"tape":tsha,"rows":articles.len(),
            "parity_articles":3,"parity_transitions":checks.transitions,
            "parity_sha256":hex::encode(checks.digest.finalize()),"matched_pairs":matched.len()/2})
        );
        Ok(())
    }

    include!("counts.rs");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, params)) if cmd == "evaluate" => calibrated::evaluate(params),
        Some((cmd, params)) if cmd == "counts" => calibrated::counts(params),
        _ => Err("expected evaluate or counts".into()),
    }
}
