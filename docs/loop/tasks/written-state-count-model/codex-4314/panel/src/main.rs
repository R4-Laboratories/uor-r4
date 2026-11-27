#![forbid(unsafe_code)]

mod analyze;
mod controlled;
mod coqa;
mod verify;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SEED: &str = "codex-4314-marker-question-panel-v1";
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const PRIOR_PROMPTS_SHA: &str = "021f4c1fe8e10884ecaa3cb3b67ed290356fb520595cc87a5c5f7b601e70d9f9";
const PRIOR_SWAP_ROWS_SHA: &str =
    "cdd8af93baac3d5cd3a953b0d1d407b96b3f6c2538cae8f23489d331ccf4f571";
const PRIOR_SWAP_KEYS_SHA: &str =
    "866a01878f6c2890a4dbe715a45808df171afe47f7bbc466bf74f6cf578815bd";
const LIMIT: usize = 192;
const SPACE: u32 = 223;
const EOS: u32 = 1;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];

struct Builder<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    assistant_suffix: Vec<u32>,
}

struct Case {
    question: String,
    reply: String,
    user: String,
    ids: Vec<u32>,
    common_len: usize,
    gold: Vec<u32>,
    lexical_offset: usize,
}

#[derive(Deserialize, Serialize)]
struct Row {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    cohort: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    raw_target: u32,
    raw_rival: u32,
    lexical_offset: usize,
    leading_whitespace: Vec<u32>,
    space_token: u32,
    equal_length: bool,
}

#[derive(Deserialize, Serialize)]
struct Key {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    user: String,
    question: String,
    reply: String,
    gold: Vec<u32>,
    source: serde_json::Value,
}

struct Pair {
    group_id: String,
    pair_id: String,
    assignment: usize,
    cohort: &'static str,
    cases: [Case; 2],
    source: serde_json::Value,
}

impl Builder<'_> {
    fn case(&self, context: &str, question: &str, reply: &str) -> Result<Option<Case>> {
        let user = format!("{context}\n\n{question}");
        let message = Message {
            role: "user",
            content: &user,
        };
        let prefix = self.encoder.encode_assistant_prefix(&[message]);
        if prefix.tokens.len() > LIMIT
            || prefix.special_token_occurrences != 0
            || prefix.emitted_turns != 1
        {
            return Ok(None);
        }
        let complete = self.encoder.encode_document(&[
            Message {
                role: "user",
                content: &user,
            },
            Message {
                role: "assistant",
                content: reply,
            },
        ]);
        let raw = complete
            .tokens
            .strip_prefix(prefix.tokens.as_slice())
            .ok_or("protocol document changes the open prompt")?;
        if raw.len() < 2
            || raw.last() != Some(&EOS)
            || complete.special_token_occurrences != 0
            || complete.emitted_turns != 2
            || complete.response_mask[prefix.tokens.len()..]
                .iter()
                .any(|&m| m != 1)
        {
            return Ok(None);
        }
        let gold = raw[..raw.len() - 1].to_vec();
        if self.tokenizer.decode(&gold) != format!(" {}", reply.trim()) {
            return Ok(None);
        }
        let Some(lexical_offset) = gold.iter().position(|&id| {
            !self
                .tokenizer
                .decode(&[id])
                .chars()
                .all(char::is_whitespace)
        }) else {
            return Ok(None);
        };
        if prefix.tokens.len() + lexical_offset > LIMIT {
            return Ok(None);
        }
        let first_bytes = self.tokenizer.decode_bytes(&[gold[lexical_offset]]);
        let Ok(first) = std::str::from_utf8(&first_bytes) else {
            return Ok(None);
        };
        if !first.trim_start().starts_with(char::is_alphanumeric) {
            return Ok(None);
        }
        if !prefix.tokens.ends_with(&self.assistant_suffix) {
            return Err("ordinary assistant suffix changed".into());
        }
        let encoded_common = self.tokenizer.encode(&format!(" {context}"));
        let ordinary_common = [vec![0], self.tokenizer.encode("User:"), encoded_common].concat();
        if !prefix.tokens.starts_with(&ordinary_common) {
            return Ok(None);
        }
        Ok(Some(Case {
            user,
            question: question.to_owned(),
            reply: reply.to_owned(),
            ids: prefix.tokens,
            common_len: ordinary_common.len(),
            gold,
            lexical_offset,
        }))
    }
}

impl Case {
    fn target(&self) -> u32 {
        self.gold[self.lexical_offset]
    }

    fn has_unique_context_target(&self, target: u32) -> bool {
        let mut matches = self.ids.iter().enumerate().filter(|(_, id)| **id == target);
        matches
            .next()
            .is_some_and(|(position, _)| position < self.common_len)
            && matches.next().is_none()
    }
}

fn sha(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut hash)?;
    Ok(hex::encode(hash.finalize()))
}

fn pin(path: &Path, expected: &str) -> Result<()> {
    let actual = sha(path)?;
    if actual != expected {
        return Err(format!(
            "{} SHA-256 is {actual}, expected {expected}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn hash(label: &str, key: &str) -> [u8; 32] {
    Sha256::digest(format!("{SEED}\0{label}\0{key}")).into()
}

fn id(label: &str, key: &str) -> String {
    hex::encode(&hash(label, key)[..12])
}

fn normalized(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn prompt_digest(text: &str) -> String {
    hex::encode(&Sha256::digest(normalized(text))[..8])
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn line(writer: &mut BufWriter<File>, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writeln!(writer)?;
    Ok(())
}

#[derive(Deserialize)]
struct PriorPrompt {
    prefix: Vec<u32>,
}

#[derive(Deserialize)]
struct PriorSwapKey {
    question: String,
    user: String,
    source: serde_json::Value,
}

#[derive(Default, Serialize)]
struct SplitCensus {
    pairs: BTreeMap<&'static str, usize>,
    max_prefix: usize,
    cohorts: BTreeMap<&'static str, CohortCensus>,
}

#[derive(Default, Serialize)]
struct CohortCensus {
    rows: usize,
    raw_space: usize,
    target_seen_in_contrast: usize,
    rival_seen_in_contrast: usize,
    unique_targets: std::collections::BTreeSet<u32>,
}

fn rows(
    pairs: Vec<Pair>,
    output: &Path,
    prior_prefixes: &mut HashSet<Vec<u32>>,
    pair_ids: &mut HashSet<String>,
    bank: &HashSet<String>,
    train_targets: &HashSet<u32>,
    lexical_steps: Option<&Path>,
) -> Result<SplitCensus> {
    fs::create_dir(output)?;
    let mut rows = writer(&output.join("rows.jsonl"))?;
    let mut keys = writer(&output.join("keys.jsonl"))?;
    let mut lexical_steps = lexical_steps.map(writer).transpose()?;
    let mut census = SplitCensus::default();
    for pair in pairs {
        let targets = pair.cases.each_ref().map(Case::target);
        let raw = pair.cases.each_ref().map(|case| case.gold[0]);
        if targets[0] == targets[1] || !pair_ids.insert(pair.pair_id.clone()) {
            return Err("duplicate pair or identical lexical targets".into());
        }
        let equal_length = pair.cases[0].ids.len() == pair.cases[1].ids.len();
        let offsets = pair.cases.each_ref().map(|case| case.lexical_offset);
        *census.pairs.entry(pair.cohort).or_default() += 1;
        let cohort = census.cohorts.entry(pair.cohort).or_default();
        for (side, case) in pair.cases.into_iter().enumerate() {
            census.max_prefix = census.max_prefix.max(case.ids.len());
            if !prior_prefixes.insert(case.ids.clone()) {
                return Err("duplicate prefix across prior, TRAIN or HELD rows".into());
            }
            if [&case.question, &case.user]
                .into_iter()
                .any(|text| bank.contains(&prompt_digest(text)))
            {
                return Err("D87 exact normalized prompt or question matched".into());
            }
            let target = targets[side];
            let rival = targets[1 - side];
            cohort.rows += 1;
            cohort.raw_space += usize::from(raw[side] == SPACE);
            cohort.target_seen_in_contrast += usize::from(train_targets.contains(&target));
            cohort.rival_seen_in_contrast += usize::from(train_targets.contains(&rival));
            cohort.unique_targets.insert(target);
            let mut row = Row {
                pair_id: pair.pair_id.clone(),
                group_id: pair.group_id.clone(),
                assignment: pair.assignment,
                side,
                cohort: pair.cohort.to_owned(),
                prefix: case.ids,
                target,
                rival,
                raw_target: raw[side],
                raw_rival: raw[1 - side],
                lexical_offset: case.lexical_offset,
                leading_whitespace: case.gold[..case.lexical_offset].to_vec(),
                space_token: SPACE,
                equal_length,
            };
            line(&mut rows, &row)?;
            if offsets != [0, 0] {
                let lexical_steps = lexical_steps
                    .as_mut()
                    .ok_or("TRAIN pair has a delayed target")?;
                row.prefix.extend_from_slice(&row.leading_whitespace);
                line(lexical_steps, &row)?;
            }
            line(
                &mut keys,
                &Key {
                    pair_id: pair.pair_id.clone(),
                    group_id: pair.group_id.clone(),
                    assignment: pair.assignment,
                    side,
                    user: case.user,
                    question: case.question,
                    reply: case.reply,
                    gold: case.gold,
                    source: pair.source.clone(),
                },
            )?;
        }
    }
    rows.flush()?;
    keys.flush()?;
    if let Some(output) = lexical_steps.as_mut() {
        output.flush()?;
    }
    Ok(census)
}

fn tokenizer(tok: &Path) -> Result<ByteBpeTokenizer> {
    pin(tok, TOKENIZER_SHA)?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tok)?)
        .ok_or("could not bind the pinned tokenizer")?;
    if tokenizer.vocab_size() != 4096 || tokenizer.encode(" ") != vec![SPACE] {
        return Err("pinned tokenizer vocabulary or space identity changed".into());
    }
    Ok(tokenizer)
}

fn run() -> Result<()> {
    let raw_args: Vec<_> = std::env::args().skip(1).collect();
    let args: Vec<_> = raw_args.iter().map(String::as_str).collect();
    if let ["--verify", rest @ ..] = args.as_slice() {
        return verify::run(rest);
    }
    if let ["--exposure", rest @ ..] = args.as_slice() {
        return verify::exposure(rest);
    }
    if let ["--summarize", rest @ ..] = args.as_slice() {
        return analyze::run(rest);
    }
    let [tok, coqa, prior, bank, prior_prompts, prior_swap_rows, prior_swap_keys, review, output] =
        args.as_slice()
    else {
        return Err("usage: marker-question-panel TOKENIZER COQA 4312_JUDGMENTS D87_DIGESTS 4312_PROMPTS 4313_ROWS 4313_KEYS REVIEW NEW_OUTPUT".into());
    };
    let tokenizer = tokenizer(Path::new(tok))?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let builder = Builder {
        tokenizer: &tokenizer,
        encoder: protocol.bind(&tokenizer)?,
        assistant_suffix: [
            tokenizer.encode("\n"),
            tokenizer.encode(protocol.assistant_marker()),
        ]
        .concat(),
    };
    if builder.assistant_suffix != MARKER {
        return Err("pinned assistant marker IDs changed".into());
    }
    let prior_prompts = Path::new(prior_prompts);
    let prior_swap_rows = Path::new(prior_swap_rows);
    let prior_swap_keys = Path::new(prior_swap_keys);
    pin(prior_prompts, PRIOR_PROMPTS_SHA)?;
    pin(prior_swap_rows, PRIOR_SWAP_ROWS_SHA)?;
    pin(prior_swap_keys, PRIOR_SWAP_KEYS_SHA)?;
    let old_keys: Vec<PriorSwapKey> = read_jsonl(prior_swap_keys)?;
    let (natural, coqa) = coqa::pairs(
        &builder,
        Path::new(coqa),
        Path::new(prior),
        Path::new(bank),
        &old_keys,
        review,
    )?;
    if *review == "--preview" {
        return Ok(());
    }
    let bank = coqa::load_bank(Path::new(bank))?;
    let mut prior_prefixes: HashSet<_> = [prior_prompts, prior_swap_rows]
        .into_iter()
        .map(read_jsonl::<PriorPrompt>)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .map(|prompt| prompt.prefix)
        .collect();
    let prior_distinct_prefixes = prior_prefixes.len();
    let mut used_subjects = HashSet::new();
    for key in old_keys {
        if let Some(subjects) = key.source.get("subjects") {
            let [left, right]: [String; 2] = serde_json::from_value(subjects.clone())?;
            used_subjects.insert(controlled::subject_pair(&left, &right));
        }
    }
    let old_subject_pairs = used_subjects.len();
    let (train, train_controlled) = controlled::pairs(&builder, &mut used_subjects, "train", None)?;
    let train_targets = train
        .iter()
        .flat_map(|pair| pair.cases.iter().map(Case::target))
        .collect();
    let (mut held, held_controlled) =
        controlled::pairs(&builder, &mut used_subjects, "held", Some(&train_targets))?;
    held.extend(natural);
    let output = Path::new(output);
    fs::create_dir(output)?;
    let mut pair_ids = HashSet::new();
    let train = rows(
        train,
        &output.join("train"),
        &mut prior_prefixes,
        &mut pair_ids,
        &bank,
        &train_targets,
        None,
    )?;
    let held = rows(
        held,
        &output.join("held"),
        &mut prior_prefixes,
        &mut pair_ids,
        &bank,
        &train_targets,
        Some(&output.join("held/lexical-steps.jsonl")),
    )?;
    let files = [
        "train/keys.jsonl",
        "train/rows.jsonl",
        "held/keys.jsonl",
        "held/rows.jsonl",
        "held/lexical-steps.jsonl",
    ];
    let mut output_hashes = BTreeMap::new();
    for name in files {
        output_hashes.insert(name, sha(&output.join(name))?);
    }
    let mut receipt = writer(&output.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema":"uor-r4.marker-question-panel/1",
            "seed":SEED,
            "tokenizer_sha256":TOKENIZER_SHA,
            "protocol":protocol.identity()?,
            "assistant_suffix":builder.assistant_suffix,
            "space_token":SPACE,
            "context_limit":LIMIT,
            "prior":{
                "4312_prompts_sha256":PRIOR_PROMPTS_SHA,
                "4313_rows_sha256":PRIOR_SWAP_ROWS_SHA,
                "4313_keys_sha256":PRIOR_SWAP_KEYS_SHA,
                "distinct_encoded_prefixes":prior_distinct_prefixes,
                "4313_subject_pairs":old_subject_pairs,
            },
            "train":train,
            "held":held,
            "controlled":{"train":train_controlled,"held":held_controlled},
            "coqa":coqa,
            "outputs":output_hashes,
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    let mut seals = writer(&output.join("sha256sums.txt"))?;
    for name in files.into_iter().chain(["receipt.json"]) {
        writeln!(seals, "{}  {name}", sha(&output.join(name))?)?;
    }
    seals.flush()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("marker-question-panel: {error}");
        std::process::exit(1);
    }
}
