use super::{require, Audit, VOCAB};
use serde::{de::DeserializeOwned, Deserialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Deserialize)]
pub struct Record {
    pub id: String,
    pub kind: String,
    pub source: String,
    pub messages: Vec<User>,
    pub turns: Vec<Turn>,
}

#[derive(Deserialize)]
pub struct User {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct Turn {
    pub answer: String,
}

#[derive(Deserialize)]
pub struct SourceGroup {
    pub id: String,
    pub raw_record_sha256: String,
    pub group: usize,
    pub canonical_train_group: usize,
    pub turns: usize,
}

#[derive(Deserialize)]
pub struct Group {
    pub group: usize,
    pub canonical_train_group: usize,
    pub payload_start: usize,
    pub visible_end: usize,
    pub payload_end: usize,
}

#[derive(Deserialize)]
pub struct Opening {
    pub group: usize,
    pub turn: usize,
    pub prefix_start: usize,
    pub answer_start: usize,
    pub answer_end: usize,
    pub immediate_id: u32,
    pub lexical_id: u32,
    pub lexical_offset: usize,
    pub lexical_answer_offset: usize,
}

#[derive(Deserialize)]
pub struct Colon {
    pub prefix: Vec<u32>,
    pub target: u32,
    pub offset: usize,
    pub genuine_assistant: bool,
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn check(path: &Path, expected: &str) -> Audit<()> {
    let actual = digest(&fs::read(path)?);
    require(
        actual == expected,
        &format!(
            "{} SHA-256 {actual} differs from {expected}",
            path.display()
        ),
    )
}

pub fn checked_data(path: &Path, expected: &str) -> Audit<Vec<u8>> {
    let data = fs::read(path)?;
    require(
        digest(&data) == expected,
        &format!("{} SHA-256 differs from {expected}", path.display()),
    )?;
    Ok(data)
}

pub fn lines(path: &Path) -> Audit<impl Iterator<Item = std::io::Result<String>>> {
    Ok(BufReader::new(File::open(path)?).lines())
}

pub fn rows<T: DeserializeOwned>(path: &Path) -> Audit<impl Iterator<Item = Audit<T>>> {
    Ok(lines(path)?.map(|line| Ok(serde_json::from_str(&line?)?)))
}

pub fn next<T>(rows: &mut impl Iterator<Item = Audit<T>>) -> Audit<T> {
    rows.next().ok_or("truncated frozen row file")?
}

pub fn tape(bytes: &[u8]) -> Audit<Vec<u32>> {
    const HEADER: usize = 64;
    require(
        bytes.len() >= HEADER
            && &bytes[..8] == b"UORT\x01\0\0\0"
            && bytes[16..20] == (VOCAB as u32).to_le_bytes()
            && bytes[20..HEADER].iter().all(|&byte| byte == 0)
            && (bytes.len() - HEADER).is_multiple_of(2)
            && u64::from_le_bytes(bytes[8..16].try_into()?) == ((bytes.len() - HEADER) / 2) as u64,
        "invalid UORT v1 header",
    )?;
    let tokens: Vec<_> = bytes[HEADER..]
        .chunks_exact(2)
        .map(|pair| u32::from(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();
    require(
        tokens.iter().all(|&id| (id as usize) < VOCAB),
        "out-of-range original ID",
    )?;
    Ok(tokens)
}
