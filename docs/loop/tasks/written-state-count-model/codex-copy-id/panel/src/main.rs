#![forbid(unsafe_code)]

mod history;
mod panel;
mod report;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LIMIT: usize = 192;
const EOS: u32 = 1;
const SPACE: u32 = 223;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const SEED: &str = "codex-authenticated-copy-opening-panel-v1";
const COLLISION: [u32; 2] = [1808, 3984];

#[derive(Debug, Serialize)]
struct Row {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    cohort: String,
    stratum: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    raw_target: u32,
    raw_rival: u32,
    lexical_offset: usize,
    leading_whitespace: Vec<u32>,
    space_token: u32,
    equal_length: bool,
    context_start: usize,
    context_end: usize,
    gold_pos: Option<usize>,
    rival_pos: Option<usize>,
    wrong_same_id_pos: Option<usize>,
    source_status: &'static str,
    expected_second: Option<u32>,
    expected_terminal_after_first: Option<bool>,
    expected_terminal_after_second: Option<bool>,
    variant: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct Key {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    matched_group: Option<String>,
    context: String,
    user: String,
    question: String,
    reply: Option<String>,
    gold: Vec<u32>,
    pieces: Vec<String>,
    topic: &'static str,
    subjects: [String; 2],
    values_by_subject: [String; 2],
    requested_subject: String,
    first_fact_side: usize,
    value_before_subject: bool,
    punctuation: bool,
    occurrence_witness: Option<Witness>,
}

#[derive(Debug, Serialize)]
struct Witness {
    authorizing_fact_bytes: [usize; 2],
    duplicate_note_bytes: [usize; 2],
    fact_text: String,
    note_text: String,
    fact_token_pos: usize,
    note_token_pos: usize,
    original_id: u32,
}

struct Case {
    user: String,
    prefix: Vec<u32>,
    context_start: usize,
    context_end: usize,
    gold: Vec<u32>,
}

impl Case {
    fn positions(&self, id: u32) -> Vec<usize> {
        self.prefix
            .iter()
            .enumerate()
            .filter_map(|(pos, &value)| (value == id).then_some(pos))
            .collect()
    }

    fn unique_source(&self, id: u32) -> Option<usize> {
        match self.positions(id).as_slice() {
            [pos] if (self.context_start..self.context_end).contains(pos) => Some(*pos),
            _ => None,
        }
    }

    fn source_in(
        &self,
        tokenizer: &ByteBpeTokenizer,
        id: u32,
        bounds: [usize; 2],
    ) -> Option<usize> {
        // The context span starts with the wire-format joining space.
        let [start, end] = bounds.map(|bound| bound + 1);
        let mut cursor = 0;
        let mut found = None;
        for pos in self.context_start..self.context_end {
            let bytes = tokenizer.decode_bytes(&[self.prefix[pos]]);
            if self.prefix[pos] == id
                && cursor + 1 >= start
                && cursor + bytes.len() <= end
                && found.replace(pos).is_some()
            {
                return None;
            }
            cursor += bytes.len();
        }
        found
    }
}

struct Builder<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    user_head: Vec<u32>,
}

impl<'a> Builder<'a> {
    fn new(tokenizer: &'a ByteBpeTokenizer, protocol: &DialogueProtocol) -> Result<Self> {
        require(
            tokenizer.vocab_size() == 4096 && tokenizer.encode(" ") == [SPACE],
            "tokenizer identity drift",
        )?;
        let user_head = [vec![0], tokenizer.encode("User:")].concat();
        let marker = [
            tokenizer.encode("\n"),
            tokenizer.encode(protocol.assistant_marker()),
        ]
        .concat();
        require(marker == MARKER, "canonical marker drift")?;
        Ok(Self {
            tokenizer,
            encoder: protocol.bind(tokenizer)?,
            user_head,
        })
    }

    fn case(
        &self,
        context: &str,
        question: &str,
        reply: Option<&str>,
        separator: &str,
    ) -> Result<Option<Case>> {
        let user = format!("{context}{separator}{question}");
        let open = self.encoder.encode_assistant_prefix(&[Message {
            role: "user",
            content: &user,
        }]);
        let context_ids = self.tokenizer.encode(&format!(" {context}"));
        let context_start = self.user_head.len();
        let context_end = context_start + context_ids.len();
        if open.special_token_occurrences != 0
            || open.emitted_turns != 1
            || open.tokens.len() > LIMIT
            || !open.tokens.ends_with(&MARKER)
            || !open.tokens.starts_with(&self.user_head)
            || open.tokens.get(context_start..context_end) != Some(&context_ids)
        {
            return Ok(None);
        }
        if separator == "\n\n" {
            let end = open.tokens.len() - MARKER.len();
            let delimiters: Vec<_> = open.tokens[context_start..end]
                .windows(2)
                .enumerate()
                .filter_map(|(pos, ids)| (ids == [201, 201]).then_some(context_start + pos))
                .collect();
            if delimiters != [context_end] {
                return Ok(None);
            }
        }
        let mut gold = Vec::new();
        if let Some(reply) = reply {
            let closed = self.encoder.encode_document(&[
                Message {
                    role: "user",
                    content: &user,
                },
                Message {
                    role: "assistant",
                    content: reply,
                },
            ]);
            let tail = closed
                .tokens
                .strip_prefix(open.tokens.as_slice())
                .ok_or("reply changed the open prefix")?;
            if closed.emitted_turns != 2
                || closed.special_token_occurrences != 0
                || tail.last() != Some(&EOS)
                || closed.response_mask[open.tokens.len()..]
                    .iter()
                    .any(|&mask| mask != 1)
                || self.tokenizer.decode_bytes(&tail[..tail.len() - 1])
                    != format!(" {reply}").as_bytes()
            {
                return Ok(None);
            }
            gold.extend_from_slice(tail);
        }
        Ok(Some(Case {
            user,
            prefix: open.tokens,
            context_start,
            context_end,
            gold,
        }))
    }

    fn piece(&self, id: u32) -> String {
        if id == EOS {
            return "<EOS>".into();
        }
        self.tokenizer.decode(&[id])
    }
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

fn hash(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    io::copy(&mut File::open(path)?, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn prefix_hash(tokens: &[u32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for &token in tokens {
        hash.update((token as u16).to_le_bytes());
    }
    hash.finalize().into()
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn fresh_writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let [operation, tokenizer, panel, observation, output] = args.as_slice() {
        require(operation == "report", "invalid operation")?;
        return report::run(
            Path::new(tokenizer),
            Path::new(panel),
            Path::new(observation),
            Path::new(output),
        );
    }
    let [operation, root, output] = args.as_slice() else {
        return Err("usage: copy-opening-panel <build|verify> RUNS_ROOT OUTPUT_DIRECTORY | report TOKENIZER PANEL OBSERVATIONS NEW_OUTPUT".into());
    };
    let root = Path::new(root);
    let output = Path::new(output);
    require(
        matches!(operation.as_str(), "build" | "verify"),
        "invalid operation",
    )?;
    let history = history::History::load(root)?;
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(root.join(history::TOKENIZER))?)
            .ok_or("invalid pinned tokenizer")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let builder = Builder::new(&tokenizer, &protocol)?;
    let prior = history.screens(root, &tokenizer)?;
    let tapes = panel::build(&builder, &prior)?;
    let mut file_hashes = BTreeMap::new();
    if operation == "build" {
        fs::create_dir(output)?;
    }
    for (name, tape) in [
        ("held", &tapes.held),
        ("absent", &tapes.absent),
        ("finite", &tapes.finite),
    ] {
        if operation == "build" {
            fs::create_dir(output.join(name))?;
        }
        for (kind, expected) in [
            ("rows", panel::serialized(tape.iter().map(|(row, _)| row))?),
            ("keys", panel::serialized(tape.iter().map(|(_, key)| key))?),
        ] {
            let path = format!("{name}/{kind}.jsonl");
            if operation == "build" {
                let mut writer = fresh_writer(&output.join(&path))?;
                writer.write_all(&expected)?;
                writer.flush()?;
            } else {
                require(
                    expected == fs::read(output.join(&path))?,
                    format!("rebuild mismatch: {path}"),
                )?;
            }
            file_hashes.insert(path, format!("{:x}", Sha256::digest(expected)));
        }
    }
    let receipt = serde_json::json!({
        "schema": "uor-r4.authenticated-copy-opening-panel/1",
        "seed": SEED, "limit": LIMIT, "space_token": SPACE, "eos": EOS, "assistant_suffix": MARKER,
        "protocol": protocol.identity()?, "inputs": history.pins, "audit": tapes.audit, "outputs": file_hashes,
    });
    let mut receipt_bytes = serde_json::to_vec_pretty(&receipt)?;
    receipt_bytes.push(b'\n');
    let receipt_path = output.join("receipt.json");
    if operation == "build" {
        let mut writer = fresh_writer(&receipt_path)?;
        writer.write_all(&receipt_bytes)?;
        writer.flush()?;
    } else {
        require(
            receipt_bytes == fs::read(&receipt_path)?,
            "receipt rebuild mismatch",
        )?;
    }
    file_hashes.insert(
        "receipt.json".into(),
        format!("{:x}", Sha256::digest(receipt_bytes)),
    );
    let manifest = file_hashes
        .iter()
        .map(|(path, hash)| format!("{hash}  {path}\n"))
        .collect::<String>();
    let manifest_path = output.join("SHA256SUMS");
    if operation == "build" {
        let mut writer = fresh_writer(&manifest_path)?;
        writer.write_all(manifest.as_bytes())?;
        writer.flush()?;
    } else {
        require(
            manifest == fs::read_to_string(&manifest_path)?,
            "manifest rebuild mismatch",
        )?;
    }
    println!("{}  SHA256SUMS", hash(&manifest_path)?);
    Ok(())
}

fn main() -> Result<()> {
    run()
}
