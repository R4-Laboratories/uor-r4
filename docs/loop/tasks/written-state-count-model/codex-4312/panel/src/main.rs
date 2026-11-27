#![forbid(unsafe_code)]

mod build;
mod grade;
mod open;
mod rubric;
mod stats;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const COQA_SHA: &str = "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a";
const WIKI_SHA: &str = "e9f695672d916793439d8f48893ecde94c61b0eb038a0034c4e1f1f194205979";
const BANK_SHA: &str = "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541";
const SEED: &str = "codex-4312-visible-answer-panel-v1";
const BOS: u16 = 0;
const EOS: u16 = 1;
const UNK: u16 = 2;
const CHAT_ROWS: usize = 200;
const OPEN_ROWS: usize = 32;
const OPEN_CAP: usize = 128;
const RECALL_PAIRS: usize = 32;
const PROSE_ROWS: usize = 128;
const CHAT_CAP: usize = 32;
const PROSE_CAP: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Domain {
    Conversation,
    Recall,
    Prose,
    Open,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Prompt {
    id: String,
    domain: Domain,
    prefix: Vec<u16>,
    cap: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Teacher {
    #[serde(flatten)]
    prompt: Prompt,
    gold: Vec<u16>,
    new_teacher_tokens: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Expected {
    Fact(Vec<String>),
    Yes,
    No,
}

#[derive(Debug, Serialize, Deserialize)]
struct Judgment {
    id: String,
    question: String,
    answer: Expected,
    references: Vec<String>,
    custody: Value,
}

fn bytes_sha(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut hash)?;
    Ok(hex::encode(hash.finalize()))
}

fn check_pin(path: &Path, pin: &str) -> Result<()> {
    let hash = bytes_sha(path)?;
    if hash != pin {
        return Err(format!("{} SHA-256 is {hash}, expected {pin}", path.display()).into());
    }
    Ok(())
}

fn hash(label: &str, key: &str) -> [u8; 32] {
    Sha256::digest(format!("{SEED}\0{label}\0{key}")).into()
}

fn opaque_id(label: &str, key: &str) -> String {
    hex::encode(&hash(label, key)[..12])
}

fn tokenizer(path: &Path) -> Result<ByteBpeTokenizer> {
    check_pin(path, TOKENIZER_SHA)?;
    ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(path)?)
        .ok_or_else(|| "could not bind the pinned tokenizer".into())
}

fn ids(tokens: &[u32]) -> Result<Vec<u16>> {
    Ok(tokens
        .iter()
        .copied()
        .map(u16::try_from)
        .collect::<std::result::Result<_, _>>()?)
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line?;
            serde_json::from_str(&line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1).into())
        })
        .collect()
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = create(path)?;
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn write_jsonl<'a, T: Serialize + 'a>(
    path: &Path,
    entries: impl IntoIterator<Item = &'a T>,
) -> Result<()> {
    let mut writer = create(path)?;
    for entry in entries {
        serde_json::to_writer(&mut writer, entry)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["build", tok, coqa, wiki, oasst, bank, prior_fit, prior_held, limit, out] => build::run(
            Path::new(tok), Path::new(coqa), Path::new(wiki), Path::new(oasst), Path::new(bank),
            Path::new(prior_fit), Path::new(prior_held), limit.parse()?, Path::new(out),
        ),
        ["grade", tok, panel, generations, arms, out] => {
            grade::run(Path::new(tok), Path::new(panel), Path::new(generations), arms.parse()?, Path::new(out))
        }
        ["stats", panel, grades, gpu_teacher, out] => stats::run(Path::new(panel), Path::new(grades), Path::new(gpu_teacher), Path::new(out)),
        _ => Err("usage: visible-answer-panel build TOKENIZER COQA WIKI OASST2 BANK_DIGESTS PRIOR_FIT PRIOR_HELD CONTEXT OUT\n       visible-answer-panel grade TOKENIZER PANEL BLIND_GENERATIONS ARM_COUNT NEW_OUTPUT\n       visible-answer-panel stats SEALED_PANEL BLIND_GRADE_DIR BLIND_GPU_TEACHER_JSONL NEW_STATS_DIR".into()),
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("visible-answer-panel: {err}");
        std::process::exit(1);
    }
}
