#![forbid(unsafe_code)]

mod controlled;
mod coqa;

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
const SEED: &str = "codex-4313-question-swap-v1";
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const LIMIT: usize = 192;
const SPACE: u32 = 223;
const EOS: u32 = 1;

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

#[derive(Serialize)]
struct Row {
    pair_id: String,
    group_id: String,
    side: usize,
    cohort: &'static str,
    prefix: Vec<u32>,
    target: u32,
    raw_target: u32,
    lexical_offset: usize,
    leading_whitespace: Vec<u32>,
    space_token: u32,
    equal_length: bool,
}

#[derive(Serialize)]
struct Key {
    pair_id: String,
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

fn rows(pairs: Vec<Pair>, output: &Path) -> Result<(BTreeMap<&'static str, usize>, usize)> {
    let mut rows = writer(&output.join("rows.jsonl"))?;
    let mut keys = writer(&output.join("keys.jsonl"))?;
    let mut counts = BTreeMap::new();
    let mut ids = HashSet::new();
    let mut prefixes = HashSet::new();
    let mut maximum = 0;
    for pair in pairs {
        if pair.cases[0].target() == pair.cases[1].target() || !ids.insert(pair.pair_id.clone()) {
            return Err("duplicate pair or identical lexical targets".into());
        }
        let equal_length = pair.cases[0].ids.len() == pair.cases[1].ids.len();
        *counts.entry(pair.cohort).or_default() += 1;
        for (side, case) in pair.cases.into_iter().enumerate() {
            maximum = maximum.max(case.ids.len());
            if !prefixes.insert(case.ids.clone()) {
                return Err("duplicate prefix".into());
            }
            let target = case.target();
            line(
                &mut rows,
                &Row {
                    pair_id: pair.pair_id.clone(),
                    group_id: pair.group_id.clone(),
                    side,
                    cohort: pair.cohort,
                    prefix: case.ids,
                    target,
                    raw_target: case.gold[0],
                    lexical_offset: case.lexical_offset,
                    leading_whitespace: case.gold[..case.lexical_offset].to_vec(),
                    space_token: SPACE,
                    equal_length,
                },
            )?;
            line(
                &mut keys,
                &Key {
                    pair_id: pair.pair_id.clone(),
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
    Ok((counts, maximum))
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (tok, coqa_inputs, output) = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [tok, output] => (*tok, None, *output),
        [tok, coqa, prior, bank, review, output] => (*tok, Some((*coqa, *prior, *bank, *review)), *output),
        _ => return Err("usage: question-swap-panel TOKENIZER [COQA PRIOR_JUDGMENTS D87_DIGESTS REVIEW] NEW_OUTPUT".into()),
    };
    let tok = Path::new(tok);
    pin(tok, TOKENIZER_SHA)?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tok)?)
        .ok_or("could not bind the pinned tokenizer")?;
    if tokenizer.vocab_size() != 4096 || tokenizer.encode(" ") != vec![SPACE] {
        return Err("pinned tokenizer vocabulary or space identity changed".into());
    }
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
    let (mut pairs, controlled) = controlled::pairs(&builder)?;
    let coqa = if let Some((dataset, prior, bank, review)) = coqa_inputs {
        let (transfer, census) = coqa::pairs(
            &builder,
            Path::new(dataset),
            Path::new(prior),
            Path::new(bank),
            review,
        )?;
        pairs.extend(transfer);
        Some(census)
    } else {
        None
    };
    let output = Path::new(output);
    fs::create_dir(output)?;
    let (pairs, max_prefix) = rows(pairs, output)?;
    let mut output_hashes = BTreeMap::new();
    for name in ["keys.jsonl", "rows.jsonl"] {
        output_hashes.insert(name, sha(&output.join(name))?);
    }
    let mut receipt = writer(&output.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema":"uor-r4.question-swap-panel/1",
            "seed":SEED,
            "tokenizer_sha256":TOKENIZER_SHA,
            "protocol":protocol.identity()?,
            "assistant_suffix":builder.assistant_suffix,
            "space_token":SPACE,
            "context_limit":LIMIT,
            "max_prefix":max_prefix,
            "pairs":pairs,
            "controlled":controlled,
            "coqa":coqa,
            "outputs":output_hashes,
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    let mut seals = writer(&output.join("sha256sums.txt"))?;
    for name in ["keys.jsonl", "receipt.json", "rows.jsonl"] {
        writeln!(seals, "{}  {name}", sha(&output.join(name))?)?;
    }
    seals.flush()?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("question-swap-panel: {error}");
        std::process::exit(1);
    }
}
