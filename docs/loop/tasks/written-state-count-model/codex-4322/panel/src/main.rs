#![forbid(unsafe_code)]

mod design;
mod prepare;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Row {
    pair_id: String,
    side: u8,
    contrast: u8,
    positions: [u32; 4],
    ids: [u32; 4],
    prefix: Vec<u32>,
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
        hash(path)? == prepare::TOKENIZER_SHA256,
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
        _ => Err("usage: period-margin-panel prepare RUNS NEW_PANEL".into()),
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

fn verify_manifest(dir: &Path, manifest: &Path) -> Result<()> {
    for line in fs::read_to_string(manifest)?.lines() {
        let (expected, relative) = line.split_once("  ").ok_or("malformed digest line")?;
        require(
            hash(&dir.join(relative))? == expected,
            format!("digest changed: {relative}"),
        )?;
    }
    Ok(())
}

fn write_manifest(directory: &Path, relatives: &[&str]) -> Result<()> {
    let mut file = fresh_writer(&directory.join("SHA256SUMS"))?;
    for file_name in relatives {
        writeln!(file, "{}  {file_name}", hash(&directory.join(file_name))?)?;
    }
    file.flush()?;
    Ok(())
}
