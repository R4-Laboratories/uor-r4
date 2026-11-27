#![forbid(unsafe_code)]

mod build;
#[path = "../../../codex-4312/donor/src/source.rs"]
mod source;
mod verify;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use source::{sha, Inputs};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::{dialogue::Message, ByteBpeTokenizer};

type CorpusResult<T> = Result<T, Box<dyn Error>>;
const GROUPS: usize = 4060;
const TURNS: usize = 4345;
const BOS: u32 = 0;
const EOS: u32 = 1;
const VOCAB: u32 = 4096;
const COLON: u32 = 28;
const SPACE: u32 = 223;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, COLON];
const WINDOW: usize = 192;
const HEADER: usize = 64;
const TAPE_NAME: &str = "answer-first.tokens.u16";
const MASK_NAME: &str = "answer-first.mask.u8";
const GROUPS_NAME: &str = "groups.jsonl";
const OPENINGS_NAME: &str = "openings.jsonl";
const COLON_NAME: &str = "colon-targets.jsonl";
const DATA_FILES: [&str; 5] = [TAPE_NAME, MASK_NAME, GROUPS_NAME, OPENINGS_NAME, COLON_NAME];
const OLD_RECEIPT: &str = "4ff029e7962b6c58c88014f1baf6adeb4bd5fd1e3d0464cdbbf66d929d10bbd8";
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

fn require(ok: bool, why: impl Into<String>) -> CorpusResult<()> {
    if !ok {
        return Err(why.into().into());
    }
    Ok(())
}

fn create(root: &Path, name: &str) -> CorpusResult<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(name))?,
    ))
}

fn write_row(writer: &mut BufWriter<File>, row: &impl Serialize) -> CorpusResult<()> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn rows<T: DeserializeOwned>(path: &Path) -> CorpusResult<impl Iterator<Item = CorpusResult<T>>> {
    let lines = BufReader::new(File::open(path)?).lines();
    Ok(lines.map(|line| Ok(serde_json::from_str(&line?)?)))
}

fn read_tape(bytes: &[u8]) -> CorpusResult<Vec<u32>> {
    require(
        bytes.len() >= HEADER
            && &bytes[..8] == b"UORT\x01\0\0\0"
            && bytes[16..20] == VOCAB.to_le_bytes()
            && bytes[20..HEADER].iter().all(|&b| b == 0)
            && (bytes.len() - HEADER) % 2 == 0
            && u64::from_le_bytes(bytes[8..16].try_into()?) == ((bytes.len() - HEADER) / 2) as u64,
        "invalid UORT v1 header",
    )?;
    let ids: Vec<_> = bytes[HEADER..]
        .chunks_exact(2)
        .map(|pair| u32::from(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();
    require(ids.iter().all(|&id| id < VOCAB), "out-of-range original ID")?;
    Ok(ids)
}

fn load_tokenizer(path: &Path) -> CorpusResult<ByteBpeTokenizer> {
    let bytes = fs::read(path)?;
    require(sha(&bytes) == TOKENIZER, "P4 tokenizer hash mismatch")?;
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes).ok_or("invalid P4 tokenizer")?;
    require(
        tok.vocab_size() == VOCAB as usize
            && tok.encode(":") == [COLON]
            && tok.encode(" ") == [SPACE]
            && [tok.encode("\n"), tok.encode("Assistant:")].concat() == MARKER,
        "P4 colon, space, marker or vocabulary differs",
    )?;
    Ok(tok)
}

#[derive(Deserialize)]
struct Record {
    id: String,
    messages: Vec<User>,
    turns: Vec<Turn>,
}

#[derive(Deserialize)]
struct User {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Turn {
    answer: String,
}

impl Record {
    fn messages(&self) -> Vec<Message<'_>> {
        self.messages
            .iter()
            .zip(&self.turns)
            .flat_map(|(user, turn)| {
                [
                    Message {
                        role: &user.role,
                        content: &user.content,
                    },
                    Message {
                        role: "assistant",
                        content: &turn.answer,
                    },
                ]
            })
            .collect()
    }
}

#[derive(Deserialize)]
struct PriorGroup {
    id: String,
    raw_record_sha256: String,
    group: usize,
    canonical_train_group: usize,
    payload_start: usize,
    payload_tokens_both_arms: usize,
    visible_tokens_answer_first: usize,
    turns: usize,
    reasoning_turns: usize,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct Group {
    group: usize,
    canonical_train_group: usize,
    payload_start: usize,
    visible_end: usize,
    payload_end: usize,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct Opening {
    group: usize,
    turn: usize,
    prefix_start: usize,
    answer_start: usize,
    answer_end: usize,
    immediate_id: u32,
    lexical_id: u32,
    lexical_offset: usize,
    lexical_answer_offset: usize,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ColonRow {
    prefix: Vec<u32>,
    target: u32,
    offset: usize,
    genuine_assistant: bool,
}

fn prior_receipt(root: &Path) -> CorpusResult<Value> {
    let bytes = fs::read(root.join("receipt.json"))?;
    require(
        sha(&bytes) == OLD_RECEIPT,
        "historical donor receipt differs",
    )?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn old_file(root: &Path, receipt: &Value, name: &str) -> CorpusResult<Vec<u8>> {
    let bytes = fs::read(root.join(name))?;
    let pin = &receipt["files"][name];
    require(
        pin["bytes"].as_u64() == Some(bytes.len() as u64)
            && pin["sha256"].as_str() == Some(&sha(&bytes)),
        format!("historical donor file pin differs for {name}"),
    )?;
    Ok(bytes)
}

fn main() -> CorpusResult<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [command, raw, split, d87_bpe, p4_bpe, train, r, prior, out, threads] if command == "build" => {
            let inputs = Inputs {
                raw: raw.into(),
                split: split.into(),
                d87_bpe: d87_bpe.into(),
                p4_bpe: p4_bpe.into(),
                train: train.into(),
                r: r.into(),
            };
            let threads = threads.to_str().ok_or("invalid thread argument")?.parse()?;
            require((1..=16).contains(&threads), "threads must be 1..=16")?;
            build::run(&inputs, Path::new(prior), Path::new(out), threads)
        }
        [command, tokenizer, prior, out] if command == "verify" => {
            verify::run(Path::new(tokenizer), Path::new(prior), Path::new(out))
        }
        _ => Err("usage: uor-canonical-donor build RAW SPLIT D87_TOKENIZER P4_TOKENIZER FULL_TRAIN R PRIOR OUT THREADS | verify P4_TOKENIZER PRIOR OUT".into()),
    }
}
