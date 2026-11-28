use crate::{Result, inputs, pinned, present, require};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

const ROSTERS: [(&str, &str, &str, usize); 10] = [
    (
        "4361-superseded-train",
        "20261128-codex-grounded-reset/cohort/final/train.jsonl",
        "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a",
        256,
    ),
    (
        "4361-superseded-development",
        "20261128-codex-grounded-reset/cohort/final/development.jsonl",
        "43484a97e94253a164367fdd721304ba54cfbdf62ce60da99cdfefc700a25fc9",
        110,
    ),
    (
        "4361-superseded-held",
        "20261128-codex-grounded-reset/cohort/final/held.jsonl",
        "89e57de07d82223d6fc9bc1c57690e8c1d84dddbd51cc319bb56d8f7353a02fe",
        0,
    ),
    (
        "4361-corrected-train",
        "20261128-codex-grounded-reset/cohort/complete-reservation/final/train.jsonl",
        "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a",
        256,
    ),
    (
        "4361-corrected-development",
        "20261128-codex-grounded-reset/cohort/complete-reservation/final/development.jsonl",
        "b7f1c457d5d656a92c742ca55fa9f49435cb6d92fea925f83e000079f44f8598",
        0,
    ),
    (
        "4361-corrected-held",
        "20261128-codex-grounded-reset/cohort/complete-reservation/final/held.jsonl",
        "6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee",
        222,
    ),
    (
        "4360-corrected",
        "20261128-codex-compact-carry-preflight/cohort/v2/final-2048/selected-index.jsonl",
        "d6dd93ab4cbacf4eb6ea6fdaf42e5264dda4a325b78c14f2fcf038c9485adfb7",
        64,
    ),
    (
        "4360-rejected",
        "20261128-codex-compact-carry-preflight/cohort/final-2048/selected-index.jsonl",
        "a335a338dda285c192582256797c6dd2a8afbd633fea90acf38d670314e4b005",
        64,
    ),
    (
        "lexical-supplement-1",
        "20261128-codex-lexical-transfer/supplement-1/truth.jsonl",
        "483bf82f99e7ba454986726abc98b9bdee66b0df230b6a2c089a862dd364feee",
        200,
    ),
    (
        "lexical-supplement-2",
        "20261128-codex-lexical-transfer/supplement-2/truth.jsonl",
        "483bf82f99e7ba454986726abc98b9bdee66b0df230b6a2c089a862dd364feee",
        200,
    ),
];

pub struct Prior {
    pub identity: [HashSet<String>; 3],
    pub question_families: HashSet<String>,
    pub receipts: BTreeMap<&'static str, Value>,
}

impl Prior {
    pub fn load(
        runs: &Path,
        docs: &[inputs::Conversation],
        groups: &[[String; 3]],
    ) -> Result<Self> {
        let by_id: HashMap<_, _> = docs
            .iter()
            .zip(groups)
            .map(|(doc, key)| (doc.id.as_str(), (doc, key)))
            .collect();
        require(
            by_id.len() == docs.len(),
            "CoQA TRAIN conversation IDs are not unique",
        )?;
        let mut prior = Self {
            identity: Default::default(),
            question_families: HashSet::new(),
            receipts: BTreeMap::new(),
        };
        for (name, path, sha, expected) in ROSTERS {
            let bytes = pinned(&runs.join(path), sha)?;
            let mut rows = 0;
            let mut joined_turns = 0;
            for line in std::str::from_utf8(&bytes)?.lines() {
                let row: Value = serde_json::from_str(line)?;
                rows += 1;
                if name.starts_with("4361-") {
                    let doc = add_group(&mut prior, &by_id, parse_group(&row["source_group"])?)?;
                    let turn = present(
                        doc.questions.get(2),
                        "prior #4361 original third turn absent",
                    )?;
                    require(
                        row["question_sha256"] == crate::sha(turn.input_text.as_bytes()),
                        "prior #4361 question hash differs",
                    )?;
                    joined_turns += add_prefix(&mut prior, doc, 3)?;
                } else if name.starts_with("4360-") {
                    for side in ["a", "b"] {
                        let doc = add_group(&mut prior, &by_id, parse_group(&row[side]["group"])?)?;
                        let turn =
                            present(row[side]["target_turn"].as_u64(), "prior #4360 turn absent")?;
                        joined_turns += add_prefix(&mut prior, doc, usize::try_from(turn)?)?;
                    }
                } else {
                    let doc = &row["source"];
                    let file = format!("{}/{}", string(&doc["source"])?, string(&doc["filename"])?);
                    let original = add_group(
                        &mut prior,
                        &by_id,
                        [
                            string(&doc["id"])?,
                            &file,
                            string(&doc["normalized_passage_sha256"])?,
                        ],
                    )?;
                    let questions = present(
                        row["questions"].as_array(),
                        "missing prior lexical questions",
                    )?;
                    require(
                        questions.len() <= original.questions.len(),
                        "prior lexical question count differs",
                    )?;
                    for (prior_q, original_q) in questions.iter().zip(&original.questions) {
                        require(
                            string(prior_q)? == original_q.input_text,
                            "prior lexical question does not rejoin original",
                        )?;
                    }
                    joined_turns += add_prefix(&mut prior, original, questions.len())?;
                }
            }
            require(
                rows > 0 && (expected == 0 || rows == expected),
                format!("unexpected row count in {name}: {rows}"),
            )?;
            prior.receipts.insert(name, serde_json::json!({
                "path": path, "sha256": sha, "rows": rows, "joined_prior_question_occurrences": joined_turns
            }));
        }
        Ok(prior)
    }
}

fn string(value: &Value) -> Result<&str> {
    present(value.as_str(), "missing prior string")
}

fn parse_group(value: &Value) -> Result<[&str; 3]> {
    Ok([string(&value[0])?, string(&value[1])?, string(&value[2])?])
}

fn add_group<'a>(
    prior: &mut Prior,
    by_id: &HashMap<&str, (&'a inputs::Conversation, &[String; 3])>,
    keys: [&str; 3],
) -> Result<&'a inputs::Conversation> {
    let (doc, key) = present(by_id.get(keys[0]), "prior source missing from CoQA TRAIN")?;
    require(
        key.iter().zip(keys).all(|(a, b)| a == b),
        "prior source identity keys differ",
    )?;
    for (set, value) in prior.identity.iter_mut().zip(keys) {
        set.insert(value.to_owned());
    }
    Ok(doc)
}

fn add_prefix(prior: &mut Prior, doc: &inputs::Conversation, turn: usize) -> Result<usize> {
    let questions = present(doc.questions.get(..turn), "prior source turn out of bounds")?;
    require(
        turn > 0
            && questions
                .iter()
                .enumerate()
                .all(|(i, q)| q.turn_id == i + 1),
        "prior question order differs",
    )?;
    prior
        .question_families
        .extend(questions.iter().map(|q| inputs::normalize(&q.input_text)));
    Ok(questions.len())
}
