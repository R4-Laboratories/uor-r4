mod dialogue;
mod inputs;

use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use dialogue::{Render, RoleText};
use inputs::{Conversation, Pool, passage_digest};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CONTEXT: usize = 2048;
const MAX_SELECTED: usize = 200;

fn present<T>(value: Option<T>, message: &str) -> Result<T> {
    value.ok_or_else(|| message.into())
}

fn require(value: bool, message: impl Into<String>) -> Result<()> {
    if value {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn salted(salt: &str, text: &str) -> [u8; 32] {
    let mut sha = Sha256::new();
    sha.update(salt.as_bytes());
    sha.update([0]);
    sha.update(text.as_bytes());
    sha.finalize().into()
}

#[derive(Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Blind {
    id: String,
    messages: Vec<RoleText>,
    prefix: Vec<u32>,
}

#[derive(Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Source {
    dataset: String,
    split: String,
    source: String,
    id: String,
    filename: String,
    passage_sha256: String,
    normalized_passage_sha256: String,
    target_turn: usize,
}

#[derive(Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Truth {
    id: String,
    source: Source,
    story: String,
    questions: Vec<String>,
    answers: Vec<String>,
    gold: Vec<u32>,
}

#[derive(Default, Serialize)]
struct Census {
    source_conversations: usize,
    source_domains: BTreeMap<String, usize>,
    licensed_domains: BTreeMap<String, usize>,
    source_refusals: BTreeMap<&'static str, usize>,
    turn_refusals_before_first_admissible: BTreeMap<&'static str, usize>,
    eligible: usize,
    eligible_domains: BTreeMap<String, usize>,
    eligible_earlier_rounds: BTreeMap<usize, usize>,
    selected: usize,
    selected_domains: BTreeMap<String, usize>,
    selected_earlier_rounds: BTreeMap<usize, usize>,
    prefix_id_range: Option<[usize; 2]>,
    reference_id_range_including_eos: Option<[usize; 2]>,
    exact_standalone_leading_space: usize,
    selected_after_previous_eos: usize,
}

fn increment<K: Ord>(counts: &mut BTreeMap<K, usize>, key: K) {
    *counts.entry(key).or_default() += 1;
}

fn update_range(range: &mut Option<[usize; 2]>, width: usize) {
    *range = Some(range.map_or([width, width], |[min, max]| {
        [min.min(width), max.max(width)]
    }));
}

fn originals(doc: &Conversation, turn: usize) -> (Vec<String>, Vec<String>) {
    let first = |items: &[inputs::Turn]| {
        items[..turn]
            .iter()
            .map(|row| row.input_text.clone())
            .collect()
    };
    (first(&doc.questions), first(&doc.answers))
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let mut file = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?,
    );
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        writeln!(file)?;
    }
    file.flush()?;
    Ok(())
}

fn verify_readback(
    output: &Path,
    selected: usize,
    source: &[Conversation],
    renderer: &Render<'_>,
    exclusions: &inputs::Exclusions,
) -> Result<()> {
    let blind = read_jsonl::<Blind>(&output.join("blind.jsonl"))?;
    let truth = read_jsonl::<Truth>(&output.join("truth.jsonl"))?;
    require(
        blind.len() == selected && truth.len() == selected,
        "written row counts differ",
    )?;
    let sources: BTreeMap<_, _> = source.iter().map(|doc| (doc.id.as_str(), doc)).collect();
    let mut ids = HashSet::new();
    let mut passages = HashSet::new();
    let mut pairs = HashSet::new();
    for (blind, truth) in blind.iter().zip(&truth) {
        let doc = present(
            sources.get(truth.source.id.as_str()),
            "stored source ID absent",
        )?;
        require(
            (2..=5).contains(&truth.source.target_turn)
                && truth.source.target_turn <= doc.questions.len()
                && doc.valid_turns(),
            "stored target turn is outside the original conversation",
        )?;
        let hash = passage_digest(&doc.story);
        let (questions, answers) = originals(doc, truth.source.target_turn);
        require(
            blind.id == truth.id
                && ids.insert(&truth.id)
                && passages.insert(hash)
                && pairs.insert((&blind.prefix, &truth.gold))
                && exclusions.source_hit(doc, hash).is_none()
                && doc.story == truth.story
                && doc.filename == truth.source.filename
                && doc.source == truth.source.source
                && truth.source.passage_sha256 == hex::encode(digest(doc.story.as_bytes()))
                && truth.source.normalized_passage_sha256 == hex::encode(hash)
                && truth.questions == questions
                && truth.answers == answers,
            "written source identity or selected row uniqueness changed",
        )?;
        let history = dialogue::messages(&doc.story, &questions, &answers);
        let actual = renderer
            .encode(
                &history,
                &answers[answers.len() - 1],
                truth.source.target_turn - 1,
            )?
            .map_err(|reason| format!("written row refused at readback: {reason}"))?;
        require(
            blind.messages == history
                && (&blind.prefix, &truth.gold) == (&actual.0, &actual.1)
                && exclusions
                    .prompt_hit(&history[0].content, questions.iter())
                    .is_none()
                && !exclusions.prior_question_hit(questions.iter()),
            "stored blind inputs or original reply fail independent source readback",
        )?;
    }
    Ok(())
}

fn run(runs: &Path, previews: &Path, output: &Path, pool: Pool<'_>) -> Result<()> {
    let mut inputs = inputs::load(runs, previews, &pool)?;
    let renderer = Render::new(&inputs.tokenizer)?;
    inputs::planted_exclusions()?;
    renderer.planted_controls()?;
    let standalone_space = inputs.tokenizer.encode(" ");
    require(
        standalone_space.len() == 1,
        "canonical standalone space is not one ID",
    )?;
    let space = standalone_space[0];
    let mut census = Census::default();
    let mut seen = HashSet::new();
    let mut eligible_passages = HashSet::new();
    let mut blind = Vec::new();
    let mut truth = Vec::new();
    inputs
        .data
        .sort_by_cached_key(|doc| (salted(pool.source_salt(), &doc.id), doc.id.clone()));
    for doc in &inputs.data {
        require(
            !doc.id.is_empty() && seen.insert(&doc.id),
            "empty or repeated original CoQA ID",
        )?;
        census.source_conversations += 1;
        increment(&mut census.source_domains, doc.source.clone());
        if !matches!(doc.source.as_str(), "gutenberg" | "cnn" | "wikipedia") {
            increment(&mut census.source_refusals, "out_of_scope_source_domain");
            continue;
        }
        increment(&mut census.licensed_domains, doc.source.clone());
        let hash = passage_digest(&doc.story);
        if let Some(reason) = inputs.exclusions.source_hit(doc, hash) {
            increment(&mut census.source_refusals, reason);
            continue;
        }
        if eligible_passages.contains(&hash) {
            increment(
                &mut census.source_refusals,
                "duplicate_eligible_complete_passage",
            );
            continue;
        }
        if !doc.valid_turns() || doc.story.trim().is_empty() || doc.filename.is_empty() {
            increment(
                &mut census.source_refusals,
                "malformed_original_conversation",
            );
            continue;
        }
        if doc.questions.len() < 2 {
            increment(&mut census.source_refusals, "no_original_followup");
            continue;
        }

        let mut chosen = None;
        for turn in 2..=doc.questions.len().min(5) {
            let (questions, answers) = originals(doc, turn);
            let history = dialogue::messages(&doc.story, &questions, &answers);
            let refused = inputs
                .exclusions
                .prompt_hit(&history[0].content, questions.iter())
                .or_else(|| {
                    inputs
                        .exclusions
                        .prior_question_hit(questions.iter())
                        .then_some("known_coqa_exact_question")
                });
            if let Some(reason) = refused {
                increment(&mut census.turn_refusals_before_first_admissible, reason);
                continue;
            }
            let (prefix, gold) = match renderer.encode(&history, &answers[turn - 1], turn - 1)? {
                Ok(result) => result,
                Err(reason) => {
                    increment(&mut census.turn_refusals_before_first_admissible, reason);
                    continue;
                }
            };
            chosen = Some((turn, history, questions, answers, prefix, gold));
            break;
        }
        let Some((turn, history, questions, answers, prefix, gold)) = chosen else {
            increment(&mut census.source_refusals, "no_admissible_target_turn");
            continue;
        };
        eligible_passages.insert(hash);
        census.eligible += 1;
        increment(&mut census.eligible_domains, doc.source.clone());
        increment(&mut census.eligible_earlier_rounds, turn - 1);
        if blind.len() == MAX_SELECTED {
            continue;
        }

        let id = hex::encode(&salted(pool.id_salt(), &blind.len().to_string())[..12]);
        census.selected += 1;
        increment(&mut census.selected_domains, doc.source.clone());
        increment(&mut census.selected_earlier_rounds, turn - 1);
        update_range(&mut census.prefix_id_range, prefix.len());
        update_range(&mut census.reference_id_range_including_eos, gold.len());
        census.exact_standalone_leading_space += usize::from(gold.first() == Some(&space));
        census.selected_after_previous_eos += 1;
        blind.push(Blind {
            id: id.clone(),
            messages: history,
            prefix,
        });
        truth.push(Truth {
            id,
            source: Source {
                dataset: "CoQA v1.0".to_owned(),
                split: match pool {
                    Pool::Dev => "dev",
                    Pool::Train(..) => "train",
                }
                .to_owned(),
                source: doc.source.clone(),
                id: doc.id.clone(),
                filename: doc.filename.clone(),
                passage_sha256: hex::encode(digest(doc.story.as_bytes())),
                normalized_passage_sha256: hex::encode(hash),
                target_turn: turn,
            },
            story: doc.story.clone(),
            questions,
            answers,
            gold,
        });
    }
    require(
        census.eligible + census.source_refusals.values().sum::<usize>()
            == census.source_conversations,
        "census lost a source",
    )?;
    fs::DirBuilder::new().mode(0o700).create(output)?;
    write_jsonl(&output.join("blind.jsonl"), &blind)?;
    write_jsonl(&output.join("truth.jsonl"), &truth)?;
    verify_readback(
        output,
        census.selected,
        &inputs.data,
        &renderer,
        &inputs.exclusions,
    )?;
    let output_pins = ["blind.jsonl", "truth.jsonl"]
        .into_iter()
        .map(|name| Ok((name, hex::encode(digest(&fs::read(output.join(name))?)))))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let receipt = serde_json::json!({
        "schema": "uor-r4/held-coqa-source-supply/1",
        "pool": pool.label(),
        "context_ids": CONTEXT,
        "maximum_selected": MAX_SELECTED,
        "dialogue_protocol": renderer.protocol_identity()?,
        "inputs": inputs.receipts,
        "census": census,
        "controls": {
            "planted_prior_id_filename_and_complete_passage": "pass",
            "planted_exact_coqa_and_d87_questions_and_templates": "pass",
            "prior_and_reference_eos_and_embedded_special_id": "pass",
            "original_turn_order_and_context_overflow": "pass",
            "separate_blind_and_truth_roundtrip": "pass"
        },
        "output_sha256": output_pins,
    });
    let mut file = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(output.join("aggregate.json"))?,
    );
    serde_json::to_writer_pretty(&mut file, &receipt)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    let (runs, previews, output, pool) = match args.as_slice() {
        [runs, previews, output] => (*runs, *previews, *output, Pool::Dev),
        [runs, previews, output, "--coqa-train-as-held", source, sha] => (
            *runs,
            *previews,
            *output,
            Pool::Train(Path::new(source), sha),
        ),
        _ => {
            return Err(
                "expected RUNS PREVIEW_DIR NEW_OUTPUT [--coqa-train-as-held FILE SHA256]".into(),
            );
        }
    };
    run(
        Path::new(runs),
        Path::new(previews),
        Path::new(output),
        pool,
    )
}
