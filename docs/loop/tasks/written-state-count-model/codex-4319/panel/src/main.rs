#![forbid(unsafe_code)]

mod design;
#[allow(dead_code)]
#[path = "../../../codex-copy-id/panel/src/history.rs"]
mod history;
mod prepare;
mod score;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LIMIT: usize = 192;
const EOS: u32 = 1;
const PERIOD: u32 = 16;
const QUESTION: u32 = 33;
const NEWLINE: u32 = 201;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const COLLISION: [u32; 2] = [1808, 3984];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Row {
    pair_id: String,
    side: u8,
    prefix: Vec<u32>,
    gold_pos: u32,
    target: u32,
    expected: Vec<u32>,
    family: String,
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
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(&line?)
                .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1).into())
        })
        .collect()
}

fn fresh_writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn tokenizer(path: &Path) -> Result<ByteBpeTokenizer> {
    require(
        hash(path)? == history::TOKENIZER_SHA256,
        "tokenizer digest changed",
    )?;
    let result = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(path)?)
        .ok_or("invalid canonical tokenizer")?;
    require(
        result.vocab_size() == 4096
            && result.encode("?") == [QUESTION]
            && result.encode(".") == [PERIOD]
            && result.encode("\n\n") == [NEWLINE, NEWLINE],
        "canonical token identity changed",
    )?;
    Ok(result)
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command, runs, out] if command == "prepare" => {
            prepare::run(Path::new(runs), Path::new(out))
        }
        [command, vocab, panel, split, observations, out] if command == "score" => {
            score::run(
                &tokenizer(Path::new(vocab))?,
                Path::new(panel),
                split,
                Path::new(observations),
                Path::new(out),
            )
        }
        _ => Err("usage: word-ending-panel prepare RUNS NEW_PANEL | score TOKENIZER PANEL SPLIT OBSERVATIONS NEW_ANALYSIS".into()),
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
