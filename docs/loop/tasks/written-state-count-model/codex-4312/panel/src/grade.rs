use crate::rubric::{score, words};
use crate::*;
use std::collections::{BTreeMap, BTreeSet, HashSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Generation {
    id: String,
    arm: String,
    tokens: Vec<u16>,
}

#[derive(Clone, Serialize)]
struct Outcome {
    id: String,
    arm: String,
    domain: Domain,
    useful_through_eos: Option<bool>,
    content_correct: Option<bool>,
    judgment: &'static str,
    ended_eos: bool,
    reached_cap: bool,
    malformed_special_tokens: usize,
    tokens: Vec<u16>,
    reply: Option<String>,
    exact_reference: Option<bool>,
    first_reference_divergence: Option<usize>,
    novel_prompt_token_emissions: usize,
    distinct_output_words: usize,
    repeated_four_word_windows: usize,
}

fn is_opaque(s: &str) -> bool {
    s.len() == 24
        && s.bytes()
            .all(|ch| ch.is_ascii_digit() || (b'a'..=b'f').contains(&ch))
}

fn first_divergence(left: &[u16], right: &[u16]) -> Option<usize> {
    let common = left.iter().zip(right).take_while(|(a, b)| a == b).count();
    (common != left.len() || common != right.len()).then_some(common + 1)
}

fn outcome(
    generation: Generation,
    prompt: &Prompt,
    gold: Option<&[u16]>,
    judgment: Option<&&Judgment>,
    tokenizer: &ByteBpeTokenizer,
) -> Result<Outcome> {
    let tokens = generation.tokens;
    let ended_eos = tokens.last() == Some(&EOS);
    if tokens.is_empty()
        || tokens.len() > prompt.cap
        || (!ended_eos && tokens.len() != prompt.cap)
        || tokens
            .iter()
            .any(|&id| usize::from(id) >= tokenizer.vocab_size())
        || tokens[..tokens.len() - 1].contains(&EOS)
    {
        return Err(format!("invalid stop, token ID, or EOS for {}", generation.id).into());
    }
    let content = &tokens[..tokens.len() - usize::from(ended_eos)];
    let malformed_special_tokens = content.iter().filter(|&&id| id == BOS || id == UNK).count();
    let text = tokenizer.decode_bytes(&content.iter().map(|&v| u32::from(v)).collect::<Vec<_>>());
    let reply = String::from_utf8(text).ok();
    let content_judgment = match (judgment, reply.as_deref()) {
        _ if malformed_special_tokens != 0 => "malformed_special_tokens",
        (None, _) if prompt.domain == Domain::Open => "human_judgment_pending",
        (None, _) => "prose_diagnostic",
        (Some(_), None) => "invalid_utf8",
        (Some(judge), Some(text)) => score(text, &judge.question, &judge.answer),
    };
    let content_correct = judgment.map(|_| content_judgment == "useful");
    let reason = if malformed_special_tokens == 0 && judgment.is_some() && !ended_eos {
        "cap_without_eos"
    } else {
        content_judgment
    };
    let word_parts = reply.as_deref().map(words).unwrap_or_default();
    let mut word_windows = HashSet::new();
    let repeated_four_word_windows = word_parts
        .windows(4)
        .filter(|w| !word_windows.insert(*w))
        .count();
    let distinct_output_words = word_parts.iter().collect::<HashSet<_>>().len();
    let observed: HashSet<_> = prompt.prefix.iter().copied().collect();
    let novel_prompt_token_emissions = content.iter().filter(|id| !observed.contains(id)).count();
    Ok(Outcome {
        id: generation.id,
        arm: generation.arm,
        domain: prompt.domain,
        useful_through_eos: content_correct.map(|correct| correct && ended_eos),
        content_correct,
        judgment: reason,
        ended_eos,
        reached_cap: !ended_eos,
        malformed_special_tokens,
        reply,
        exact_reference: gold.map(|gold| tokens == gold),
        first_reference_divergence: gold.and_then(|gold| first_divergence(&tokens, gold)),
        tokens,
        novel_prompt_token_emissions,
        distinct_output_words,
        repeated_four_word_windows,
    })
}

fn encoded(tokenizer: &ByteBpeTokenizer, reply: &str) -> Result<Vec<u16>> {
    let mut tokens = ids(&tokenizer.encode(reply))?;
    tokens.push(EOS);
    Ok(tokens)
}

pub fn calibrate(
    tokenizer: &ByteBpeTokenizer,
    rows: &[Teacher],
    judgments: &[Judgment],
) -> Result<Value> {
    let panel: BTreeMap<_, _> = rows.iter().map(|row| (&row.prompt.id, row)).collect();
    let mut checks = BTreeMap::<&str, usize>::new();
    let mut blocked_count = 0;
    for judge in judgments {
        let row = panel
            .get(&judge.id)
            .ok_or("missing calibration panel row")?;
        let generate = |tokens| {
            outcome(
                Generation {
                    id: judge.id.clone(),
                    arm: String::new(),
                    tokens,
                },
                &row.prompt,
                Some(&row.gold),
                Some(&judge),
                tokenizer,
            )
        };
        let original = generate(row.gold.clone())?;
        if original.useful_through_eos != Some(true) {
            return Err(format!("original human or recall anchor failed on {}", judge.id).into());
        }
        *checks.entry("original_good_through_eos").or_default() += 1;
        let canonical = match &judge.answer {
            Expected::Fact(variants) => variants.first().ok_or("empty aliases")?.clone(),
            Expected::Yes => "yes".into(),
            Expected::No => "no".into(),
        };
        let (paraphrase, wrong, negated) = match &judge.answer {
            Expected::Fact(_) => {
                let wrong = judge
                    .custody
                    .get("alternative")
                    .and_then(Value::as_str)
                    .unwrap_or("an unrelated expression");
                (
                    format!("It was {canonical}."),
                    format!("It was {wrong}."),
                    format!("It was not {canonical}."),
                )
            }
            Expected::Yes => (
                "Yes, it did.".into(),
                "No, it did not.".into(),
                "Yes, it did not.".into(),
            ),
            Expected::No => (
                "No, it did not.".into(),
                "Yes, it did.".into(),
                "No, it did.".into(),
            ),
        };
        let accepted = generate(encoded(tokenizer, &paraphrase)?)?;
        if accepted.useful_through_eos != Some(true) {
            return Err(format!("paraphrase detector rejected {}", judge.id).into());
        }
        *checks.entry("slot_resolving_paraphrase").or_default() +=
            usize::from(accepted.exact_reference == Some(false));
        let embedded = format!("We need to choose {canonical} and continue working.");
        let combined = format!("{canonical} or another answer.");
        for (key, text) in [
            ("wrong_slot_or_polarity", wrong),
            ("negated_or_contradictory", negated),
            ("embedded_gold_boilerplate", embedded),
            ("gold_with_alternative", combined),
        ] {
            let observed = generate(encoded(tokenizer, &text)?)?;
            if observed.useful_through_eos != Some(false) {
                return Err(format!("{key} leaked through on {}", judge.id).into());
            }
            *checks.entry(key).or_default() += 1;
        }
        let echo = encoded(tokenizer, &judge.question)?;
        if echo.len() <= row.prompt.cap {
            if generate(echo)?.useful_through_eos != Some(false) {
                return Err(format!("question echo scored useful on {}", judge.id).into());
            }
            *checks.entry("question_echo").or_default() += 1;
        } else {
            blocked_count += 1;
        }
        let mut capped = encoded(tokenizer, &canonical)?;
        capped.pop();
        let fill = ids(&tokenizer.encode(" "))?
            .into_iter()
            .next()
            .ok_or("missing space token")?;
        capped.resize(row.prompt.cap, fill);
        let stop = generate(capped.clone())?;
        if stop.judgment != "cap_without_eos"
            || stop.content_correct != Some(true)
            || stop.useful_through_eos != Some(false)
        {
            return Err("unclosed answer did not fail the EOS gate".into());
        }
        *checks
            .entry("otherwise_useful_at_cap_without_eos")
            .or_default() += 1;
        capped[0] = BOS;
        let mut closed_bos = row.gold.clone();
        closed_bos[0] = BOS;
        for (key, tokens) in [("bos_with_eos", closed_bos), ("bos_at_cap", capped)] {
            let result = generate(tokens)?;
            if result.judgment != "malformed_special_tokens"
                || result.content_correct != Some(false)
                || result.useful_through_eos != Some(false)
                || result.malformed_special_tokens != 1
                || result.tokens.first() != Some(&BOS)
            {
                return Err(format!("BOS was lost or scored as useful on {}", judge.id).into());
            }
            *checks.entry(key).or_default() += 1;
        }
        if generate(vec![EOS])?.judgment != "empty" {
            return Err("empty EOS misgraded".into());
        }
        *checks.entry("empty_eos").or_default() += 1;
    }
    Ok(json!({"controls":checks,"question_echo_longer_than_generation_cap":blocked_count}))
}

fn paired(
    outcomes: &BTreeMap<(String, String), Outcome>,
    arms: &BTreeSet<String>,
    judgments: &BTreeMap<&String, &Judgment>,
) -> Vec<Value> {
    let mut pairs = Vec::new();
    for (i, arm_a) in arms.iter().enumerate() {
        for arm_b in arms.iter().skip(i + 1) {
            for domain in [
                Domain::Conversation,
                Domain::Recall,
                Domain::Prose,
                Domain::Open,
            ] {
                let mut cells = BTreeMap::<&str, usize>::new();
                let mut diverged = BTreeMap::<usize, usize>::new();
                let mut recall_results = BTreeMap::<String, [usize; 2]>::new();
                for ((arm, id), a) in outcomes {
                    if arm != arm_a || a.domain != domain {
                        continue;
                    }
                    let Some(b) = outcomes.get(&(arm_b.clone(), id.clone())) else {
                        continue;
                    };
                    if let Some(index) = first_divergence(&a.tokens, &b.tokens) {
                        *diverged.entry(index).or_default() += 1;
                    } else {
                        *cells.entry("same_tokens").or_default() += 1;
                    }
                    if let (Some(a_pass), Some(b_pass)) =
                        (a.useful_through_eos, b.useful_through_eos)
                    {
                        let cell = match (a_pass, b_pass) {
                            (true, true) => "both_useful",
                            (true, false) => "left_only_useful",
                            (false, true) => "right_only_useful",
                            (false, false) => "neither_useful",
                        };
                        *cells.entry(cell).or_default() += 1;
                        if domain == Domain::Recall {
                            if let Some(pair_id) = judgments
                                .get(id)
                                .and_then(|j| j.custody["pair_id"].as_str())
                            {
                                let pair = recall_results.entry(pair_id.to_owned()).or_default();
                                pair[0] += usize::from(a_pass);
                                pair[1] += usize::from(b_pass);
                            }
                        }
                    }
                    if let (Some(a_pass), Some(b_pass)) = (a.content_correct, b.content_correct) {
                        let cell = match (a_pass, b_pass) {
                            (true, true) => "both_content_correct",
                            (true, false) => "left_only_content_correct",
                            (false, true) => "right_only_content_correct",
                            (false, false) => "neither_content_correct",
                        };
                        *cells.entry(cell).or_default() += 1;
                    }
                }
                if domain == Domain::Recall {
                    for (_, pair) in recall_results {
                        let cell = match pair {
                            [2, 2] => "counterfactual_pairs_both_arms",
                            [2, _] => "counterfactual_pairs_left_only",
                            [_, 2] => "counterfactual_pairs_right_only",
                            _ => "counterfactual_pairs_neither",
                        };
                        *cells.entry(cell).or_default() += 1;
                    }
                }
                pairs.push(json!({"left":arm_a,"right":arm_b,"domain":domain,
                    "counts":cells,"first_token_divergences":diverged}));
            }
        }
    }
    pairs
}

pub fn run(
    tok: &Path,
    panel_dir: &Path,
    generation_file: &Path,
    arm_count: usize,
    out: &Path,
) -> Result<()> {
    if !(2..=8).contains(&arm_count) {
        return Err("declare between two and eight anonymous arms".into());
    }
    let tokenizer = tokenizer(tok)?;
    let receipt: Value = serde_json::from_reader(File::open(panel_dir.join("receipt.json"))?)?;
    if receipt["schema"] != "uor-r4.visible-answer-panel/1" {
        return Err("unrecognized panel receipt".into());
    }
    for name in ["prompts.jsonl", "teacher.jsonl", "judgments.jsonl"] {
        let pin = receipt["outputs"][name]
            .as_str()
            .ok_or("missing panel output pin")?;
        check_pin(&panel_dir.join(name), pin)?;
    }
    let rows: Vec<Prompt> = read_jsonl(&panel_dir.join("prompts.jsonl"))?;
    let teacher_rows: Vec<Teacher> = read_jsonl(&panel_dir.join("teacher.jsonl"))?;
    let records: Vec<Judgment> = read_jsonl(&panel_dir.join("judgments.jsonl"))?;
    let panel: BTreeMap<_, _> = rows.iter().map(|row| (&row.id, row)).collect();
    let teachers: BTreeMap<_, _> = teacher_rows
        .iter()
        .map(|row| (&row.prompt.id, row))
        .collect();
    let judgments: BTreeMap<_, _> = records.iter().map(|record| (&record.id, record)).collect();
    let mut per_domain = BTreeMap::new();
    for row in &rows {
        *per_domain.entry(row.domain).or_insert(0) += 1;
        let has_grade = judgments.contains_key(&row.id);
        let has_teacher = teachers.contains_key(&row.id);
        if !is_opaque(&row.id)
            || has_grade != matches!(row.domain, Domain::Conversation | Domain::Recall)
            || has_teacher != (row.domain != Domain::Open)
        {
            return Err("panel includes invalid row ID or incomplete judgments".into());
        }
    }
    if per_domain
        != BTreeMap::from([
            (Domain::Conversation, CHAT_ROWS),
            (Domain::Recall, RECALL_PAIRS * 2),
            (Domain::Prose, PROSE_ROWS),
            (Domain::Open, OPEN_ROWS),
        ])
        || panel.len() != rows.len()
        || judgments.len() != records.len()
        || judgments.len() != CHAT_ROWS + RECALL_PAIRS * 2
        || teachers.len() != teacher_rows.len()
        || teachers.len() != CHAT_ROWS + RECALL_PAIRS * 2 + PROSE_ROWS
    {
        return Err("panel matrix mismatch".into());
    }
    let mut arms = BTreeSet::new();
    let mut outcomes = BTreeMap::new();
    let mut stats = BTreeMap::<(String, Domain), BTreeMap<&str, usize>>::new();
    for generation in read_jsonl::<Generation>(generation_file)? {
        let id = generation.id.clone();
        let arm = generation.arm.clone();
        if !is_opaque(&arm) {
            return Err("visible or malformed treatment identifier".into());
        }
        arms.insert(arm.clone());
        let row = panel.get(&id).ok_or("unknown generation row")?;
        let gold = teachers.get(&id).map(|row| row.gold.as_slice());
        let result = outcome(generation, row, gold, judgments.get(&id), &tokenizer)?;
        let counts = stats.entry((arm.clone(), result.domain)).or_default();
        for (key, observed) in [
            ("rows", 1),
            (
                "useful_through_eos",
                usize::from(result.useful_through_eos == Some(true)),
            ),
            (
                "content_correct",
                usize::from(result.content_correct == Some(true)),
            ),
            ("ended_eos", usize::from(result.ended_eos)),
            ("reached_cap", usize::from(result.reached_cap)),
            (
                "malformed_special_token_rows",
                usize::from(result.malformed_special_tokens != 0),
            ),
            ("malformed_special_tokens", result.malformed_special_tokens),
            (
                "exact_reference",
                usize::from(result.exact_reference == Some(true)),
            ),
            ("generated_tokens", result.tokens.len()),
            (
                "novel_prompt_token_emissions",
                result.novel_prompt_token_emissions,
            ),
            ("distinct_output_words_sum", result.distinct_output_words),
            (
                "repeated_four_word_windows",
                result.repeated_four_word_windows,
            ),
        ] {
            *counts.entry(key).or_default() += observed;
        }
        if outcomes.insert((arm, id), result).is_some() {
            return Err("duplicate arm/row output".into());
        }
    }
    if arms.len() != arm_count || outcomes.len() != arm_count * rows.len() {
        return Err("anonymous generation matrix incomplete".into());
    }
    let summaries: Vec<_> = stats
        .into_iter()
        .map(|((arm, domain), mut counts)| {
            if matches!(domain, Domain::Prose | Domain::Open) {
                counts.remove("useful_through_eos");
                counts.remove("content_correct");
            }
            if domain == Domain::Open {
                counts.remove("exact_reference");
            }
            json!({"arm":arm,"domain":domain,"counts":counts})
        })
        .collect();
    let paired = paired(&outcomes, &arms, &judgments);
    fs::create_dir(out)?;
    write_jsonl(&out.join("blind-grades.jsonl"), outcomes.values())?;
    write_json(
        &out.join("blind-summary.json"),
        &json!({
            "schema":"uor-r4.visible-answer-blind-grade/1","arms":arms,"arm_domains":summaries,"pairs":paired,
            "input_sha256":bytes_sha(generation_file)?,"panel_receipt_sha256":bytes_sha(&panel_dir.join("receipt.json"))?,
            "grades_sha256":bytes_sha(&out.join("blind-grades.jsonl"))?,
        }),
    )?;
    Ok(())
}
