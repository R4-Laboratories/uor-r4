use super::*;
use serde_json::{json, Value};

const SALT: &str = "uor-r4-4337-fresh-interval-holdout-a";

const POOLS: &[&str] = &[
    "20261127-codex-whole-turn-census/supply-v1/train-truth.jsonl",
    "20261127-codex-whole-turn-census/supply-v1/reserve-truth.jsonl",
    "20261127-codex-whole-turn-census/supply-v1/dev-truth.jsonl",
    "20261127-codex-whole-turn-census/supply-v1/train-prompts.jsonl",
    "20261127-codex-whole-turn-census/supply-v1/reserve-prompts.jsonl",
    "20261127-codex-whole-turn-census/supply-v1/dev-prompts.jsonl",
    "20261127-codex-whole-turn-census/contrasts-v1/train.jsonl",
    "20261127-codex-whole-turn-census/contrasts-v1/dev.jsonl",
    "20261127-codex-whole-turn-fit/panel-v1/prompts.jsonl",
    "20261127-codex-edge-source/archive-v1/parent-cold/4332/panel/blind/prompts.jsonl",
    "20261127-codex-edge-source/archive-v1/parent-cold/4333/panel/blind/prompts.jsonl",
    "20261127-codex-edge-source/archive-v1/parent-cold/4332/panel/truth/key.jsonl",
    "20261127-codex-edge-source/archive-v1/parent-cold/4333/panel/truth/key.jsonl",
    "20261127-codex-edge-source/archive-v1/parent-cold/4334/transport/reverse/diagnostic-prompts.jsonl",
    "20261127-codex-edge-source/archive-v1/parent-cold/4334/transport/reverse/fresh-prompts.jsonl",
    "20261127-codex-semantic-span/panel-v1/prompts.jsonl",
    "20261127-codex-semantic-span/transfer-v2/inputs.jsonl",
    "20261127-codex-grounded-interval/data/gate-v1/train-input.jsonl",
    "20261127-codex-grounded-interval/data/gate-v1/protected-input.jsonl",
];

fn records(path: &Path) -> Result<Vec<Value>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn identity(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}

fn private_id(source: &str) -> String {
    identity(&format!("{SALT}|{source}"))
}

fn exclude(runs: &Path) -> Result<(Blocklist, Vec<Value>)> {
    let mut blocked = Blocklist::load(runs)?;
    let mut receipts = Vec::new();
    for relative in POOLS {
        let path = runs.join(relative);
        let rows = records(&path)?;
        receipts.push(json!({
            "source":relative,"rows":rows.len(),
            "sha256":hex::encode(Sha256::digest(fs::read(&path)?))
        }));
        for row in rows {
            if let Some(t) = row["article"]
                .as_str()
                .or_else(|| row["source_origin"]["article"].as_str())
            {
                blocked.titles.insert(normalize(t));
            }
            if let Some(q) = row["question"].as_str() {
                blocked.prior_questions.insert(normalize(q));
            }
            if let Some(text) = ["text", "prompt", "user"]
                .into_iter()
                .find_map(|field| row[field].as_str())
            {
                let key = normalize(text);
                blocked.shingles.extend(shingles(&key));
                blocked.prior_prompts.insert(key);
            }
        }
    }
    Ok((blocked, receipts))
}

fn held_candidates(
    dev: &Dataset,
    official_dev: bool,
    render: &Render<'_>,
    blocked: &Blocklist,
) -> (Vec<Value>, Counts) {
    let mut counts = Counts::new();
    let mut selected = Vec::new();
    let seen = HashSet::new();
    for doc in &dev.data {
        if blocked.titles.contains(&normalize(&doc.title)) {
            count(&mut counts, "blocked_titles", 1);
            continue;
        }
        for (index, p) in doc.paragraphs.iter().enumerate() {
            let key = normalize(&p.context);
            if blocked.passages.contains(&key)
                || shingles(&key).any(|s| blocked.shingles.contains(&s))
            {
                count(&mut counts, "blocked_passages", 1);
                continue;
            }
            if !p.context.ends_with(['.', '?', '!']) {
                count(&mut counts, "no_final_passage_delimiter", 1);
                continue;
            }
            for q in p.qas.iter().filter(|q| !q.is_impossible) {
                let c = match render.candidate(p, q, official_dev, blocked, &seen) {
                    Ok(c) => c,
                    Err(why) => {
                        count(&mut counts, why, 1);
                        continue;
                    }
                };
                if !c.label.whole_span_ids || c.label.source_ids.len() > 8 {
                    count(&mut counts, "inexact_or_over_cap", 1);
                    continue;
                }
                count(&mut counts, "mechanically_eligible", 1);
                let case_id = private_id(&q.id);
                selected.push(json!({
                    "case_id":case_id,"source_id":q.id,"text":c.body,"article":doc.title,
                    "paragraph":index,"passage_sha256":identity(&p.context),"question":q.question,
                    "canonical_prefix":c.input,"user_bounds":c.user_bounds,"answer":c.label
                }));
            }
        }
    }
    selected.sort_by(|a, b| a["case_id"].as_str().cmp(&b["case_id"].as_str()));
    (selected, counts)
}

fn exact_span(
    render: &Render<'_>,
    full: &[u32],
    text: &str,
    begin: usize,
    end: usize,
) -> Result<(usize, usize, Vec<u32>, Vec<u32>)> {
    let raw = format!(" {text}");
    let ids = full
        .strip_prefix(render.user_head.as_slice())
        .and_then(|v| v.strip_suffix(&MARKER))
        .ok_or("role identity changed")?;
    if render.tokenizer.decode_bytes(ids) != raw.as_bytes() {
        return Err("user roundtrip".into());
    }
    let mut cursor = 0;
    let mut first = None;
    let mut last = None;
    for (i, &id) in ids.iter().enumerate() {
        let bytes = render.tokenizer.decode_bytes(&[id]);
        let next = cursor + bytes.len();
        let w = bytes.iter().take_while(|b| b.is_ascii_whitespace()).count();
        if cursor + w == begin + 1 && w < bytes.len() {
            first = Some(i);
        }
        if next == end + 1 {
            last = Some(i + 1);
        }
        cursor = next;
    }
    let (first, last) = first.zip(last).ok_or("span not token-aligned")?;
    let source = ids.get(first..last).ok_or("reversed span")?.to_vec();
    if source.is_empty() || source.len() > 8 {
        return Err("outside copy budget".into());
    }
    let answer = text.get(begin..end).ok_or("bad author boundaries")?;
    let canonical = render.tokenizer.encode(&format!(" {answer}"));
    let lexical: Vec<_> = canonical
        .iter()
        .copied()
        .skip_while(|id| *id == 223)
        .collect();
    if lexical != source {
        return Err("source/reply tokenizer mismatch".into());
    }
    Ok((
        render.user_head.len() + first,
        render.user_head.len() + last,
        source,
        canonical,
    ))
}

fn natural(case: &Value, adjudication: &str) -> Value {
    let id = &case["case_id"];
    let answer = &case["answer"];
    let start = answer["first_position"].as_u64().unwrap_or_default();
    let source_ids = answer["source_ids"].as_array().cloned().unwrap_or_default();
    let end = start + source_ids.len() as u64;
    let mut expected = answer["response_ids"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    expected.push(json!(1));
    json!({
        "input":{"case_id":id,"text":case["text"]},
        "gold":{"case_id":id,"start":start,"end":end,"group":case["article"],"swapped_case_id":null},
        "truth":{"case_id":id,"tape":"SQuAD2.0 fresh disjoint positives from official TRAIN","source_id":case["source_id"],
            "article":case["article"],"paragraph":case["paragraph"],"passage_sha256":case["passage_sha256"],
            "question":case["question"],"canonical_prefix":case["canonical_prefix"],"source_ids":source_ids,
            "expected_ids":expected,"response_lexical_offset":answer["response_lexical_offset"],
            "original_bytes":answer["bytes"],"original_char":answer["char_start"],"span_text":answer["text"],
            "adjudication":adjudication}
    })
}

#[derive(Deserialize)]
struct AuthorPair {
    pair: String,
    status: String,
    facts: Vec<AuthorFact>,
    questions: [AuthorQuestion; 2],
}
#[derive(Deserialize)]
struct AuthorFact {
    sentence: String,
    surface: Option<String>,
    asserted: bool,
}
#[derive(Deserialize)]
struct AuthorQuestion {
    question: String,
    asks: usize,
}
fn authored_rows(author: &Path, render: &Render<'_>, blocked: &Blocklist) -> Result<Vec<Value>> {
    let pairs: Vec<AuthorPair> = serde_json::from_slice(&fs::read(author)?)?;
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for p in pairs {
        let mut passage = String::new();
        let mut ranges = Vec::new();
        if p.facts.len() != 2 || p.questions[0].asks == p.questions[1].asks {
            return Err(bad_input("bad matched questions"));
        }
        for fact in &p.facts {
            if !passage.is_empty() {
                passage.push(' ');
            }
            let bounds = if let Some(surface) = &fact.surface {
                let hits = fact
                    .sentence
                    .match_indices(surface)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>();
                let [offset] = hits.as_slice() else {
                    return Err(bad_input("not exactly one value per authored fact"));
                };
                Some((
                    passage.len() + offset,
                    passage.len() + offset + surface.len(),
                ))
            } else {
                None
            };
            if fact.asserted && bounds.is_none() {
                return Err(bad_input("assertion without extent"));
            }
            ranges.push(bounds);
            passage.push_str(&fact.sentence);
        }
        let passage_key = normalize(&passage);
        if blocked.passages.contains(&passage_key)
            || shingles(&passage_key).any(|s| blocked.shingles.contains(&s))
        {
            return Err(bad_input("authored passage overlaps known prior data"));
        }
        for (index, q) in p.questions.iter().enumerate() {
            let text = format!("{passage} {}", q.question);
            let key = normalize(&q.question);
            if blocked.blocked_prompt(&key, &text) || !seen.insert(normalize(&text)) {
                return Err(bad_input(
                    "duplicate or previously exposed authored question",
                ));
            }
            let full = render
                .encoder
                .encode_assistant_prefix(&[Message {
                    role: "user",
                    content: &text,
                }])
                .tokens;
            if full.len() > PREFIX_LIMIT {
                return Err(bad_input("overlong authored question"));
            }
            let sites = ranges
                .iter()
                .map(|r| {
                    r.map(|(s, e)| exact_span(render, &full, &text, s, e))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let fact = p.facts.get(q.asks).ok_or("invalid authored entity")?;
            let gold = if fact.asserted {
                sites[q.asks].as_ref()
            } else {
                None
            };
            let start = gold.map(|s| s.0);
            let end = gold.map(|s| s.1);
            let mut expected = gold.map(|s| s.3.clone());
            if let Some(ref mut ids) = expected {
                ids.push(1);
            }
            let case_id = private_id(&format!("authored:{}:{index}", p.pair));
            let swap = private_id(&format!("authored:{}:{}", p.pair, 1 - index));
            let truth_sites = sites
                .iter()
                .map(|site| {
                    site.as_ref()
                        .map(|s| json!({"start":s.0,"end":s.1,"source_ids":s.2}))
                })
                .collect::<Vec<_>>();
            rows.push(json!({
                "input":{"case_id":case_id,"text":text},
                "gold":{"case_id":case_id,"start":start,"end":end,"group":p.pair,"swapped_case_id":swap},
                "truth":{"case_id":case_id,"tape":"new authored closed-world","status":p.status,"question":q.question,
                    "passage_sha256":identity(&passage),"asserted":fact.asserted,"sites":truth_sites,"canonical_prefix":full,
                    "source_ids":gold.map(|s| &s.2),"expected_ids":expected,
                    "response_lexical_offset":gold.map(|s|s.3.len()-s.2.len()),"span_text":if fact.asserted { fact.surface.as_deref() } else { None }}
            }));
        }
    }
    Ok(rows)
}

pub(super) fn entry() -> Result<()> {
    let args = std::env::args()
        .skip(2)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let [runs, output, options @ ..] = args.as_slice() else {
        return Err(bad_input(
            "held RUNS NEW_OUT [AUTHOR_JSON ADJUDICATION_TSV]",
        ));
    };
    let (blocked, receipts) = exclude(runs)?;
    let dev = runs
        .join("20261127-codex-whole-turn-census/supply-inputs")
        .join(TRAIN);
    let dev: Dataset = serde_json::from_slice(&checked_bytes(&dev, PINS[0].1)?)?;
    if dev.version != "v2.0" {
        return Err(bad_input("TRAIN version drift"));
    }
    let tok = checked_bytes(&runs.join(TOKENIZER), PINS[2].1)?;
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&tok).ok_or("bad tokenizer")?;
    let render = Render::new(&tok)?;
    let (candidates, counts) = held_candidates(&dev, false, &render, &blocked);
    fs::create_dir(output)?;
    fs::write(
        output.join("screen.json"),
        serde_json::to_vec_pretty(&json!({
            "salt":SALT,"known_overlap_sources":receipts,"counts":counts,"raw_sha256":PINS[0].1,"tokenizer_sha256":PINS[2].1
        }))?,
    )?;
    if options.is_empty() {
        let mut w = BufWriter::new(File::create_new(output.join("candidates.jsonl"))?);
        for row in candidates {
            jsonl(&mut w, &row)?;
        }
        w.flush()?;
        return Ok(());
    }
    let [author, adjudication] = options else {
        return Err(bad_input("supply both author and adjudication"));
    };
    let mut adjudicated = HashSet::new();
    let mut articles = HashSet::new();
    let mut rows = authored_rows(author, &render, &blocked)?;
    for row in BufReader::new(File::open(adjudication)?).lines() {
        let row = row?;
        let (id, reason) = row.split_once('\t').ok_or("bad adjudication TSV")?;
        let c = candidates
            .iter()
            .find(|c| c["source_id"] == id)
            .ok_or("adjudication not in eligible pool")?;
        let article = c["article"].as_str().ok_or("missing article")?;
        if reason.is_empty()
            || !adjudicated.insert(id.to_owned())
            || !articles.insert(article.to_owned())
        {
            return Err(bad_input("empty or repeated adjudication or article"));
        }
        rows.push(natural(c, reason));
    }
    rows.sort_by(|a, b| {
        a["input"]["case_id"]
            .as_str()
            .cmp(&b["input"]["case_id"].as_str())
    });
    for part in ["input", "gold", "truth"] {
        let mut w = BufWriter::new(File::create_new(output.join(format!("held-{part}.jsonl")))?);
        for row in &rows {
            jsonl(&mut w, &row[part])?;
        }
        w.flush()?;
    }
    Ok(())
}
