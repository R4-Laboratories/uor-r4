#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, env, error::Error, fs, io, path::Path};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const KEY_SHA: &str = "866a01878f6c2890a4dbe715a45808df171afe47f7bbc466bf74f6cf578815bd";
const ROW_SHA: &str = "cdd8af93baac3d5cd3a953b0d1d407b96b3f6c2538cae8f23489d331ccf4f571";
const TOK_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

#[derive(Deserialize)]
struct Key<'a> {
    pair_id: &'a str,
    side: u8,
    reply: String,
    gold: Vec<u32>,
}

#[derive(Deserialize)]
struct Row<'a> {
    pair_id: &'a str,
    side: u8,
    cohort: &'a str,
    target: u32,
    raw_target: u32,
    lexical_offset: usize,
}

fn read_pinned(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != sha {
        return Err(format!("frozen pin differs for {}", path.display()).into());
    }
    Ok(bytes)
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let [panel, tok] = args.as_slice() else {
        return Err("usage: held-label-boundary FROZEN_PANEL_DIR TOKENIZER".into());
    };
    let panel = Path::new(panel);
    let tok = ByteBpeTokenizer::from_tokenizer_json_bytes(&read_pinned(Path::new(tok), TOK_SHA)?)
        .ok_or("invalid tokenizer")?;
    let keys = read_pinned(&panel.join("keys.jsonl"), KEY_SHA)?;
    let rows = read_pinned(&panel.join("rows.jsonl"), ROW_SHA)?;
    let keys: Vec<_> = std::str::from_utf8(&keys)?.lines().collect();
    let rows: Vec<_> = std::str::from_utf8(&rows)?.lines().collect();
    if keys.len() != 810 || keys.len() != rows.len() {
        return Err("sealed key and row inventories differ".into());
    }
    let mut output = Vec::with_capacity(keys.len());
    let mut seen = BTreeSet::new();
    for (key, row) in keys.into_iter().zip(rows) {
        let key: Key<'_> = serde_json::from_str(key)?;
        let row: Row<'_> = serde_json::from_str(row)?;
        let reply = key.reply.trim();
        if (key.pair_id, key.side) != (row.pair_id, row.side)
            || row.side > 1
            || !seen.insert((row.pair_id, row.side))
            || key.gold != tok.encode(&format!(" {reply}"))
            || key.gold.first() != Some(&row.raw_target)
            || key.gold.get(row.lexical_offset) != Some(&row.target)
        {
            return Err("sealed gold, canonical target or row identity differs".into());
        }
        let first = tok.encode(reply).first().copied();
        let surface = first.map(|id| tok.decode_bytes(&[id]));
        let status = match surface.as_deref().map(std::str::from_utf8) {
            None => "no_token",
            Some(Err(_)) => "invalid_utf8_first_token",
            Some(Ok(first)) if first.chars().all(char::is_whitespace) => "whitespace_first_token",
            Some(Ok(first)) if first.starts_with(char::is_alphanumeric) => {
                "alphanumeric_first_token"
            }
            Some(Ok(_)) => "other_non_alphanumeric_first_token",
        };
        output.push(
            json!({"pair_id":row.pair_id, "side":row.side, "cohort":row.cohort,
            "canonical_id":row.target, "unprefixed_first_id":first,
            "canonical_id_identical":first == Some(row.target), "unprefixed_status":status}),
        );
    }
    write_jsonl(&output)
}

fn write_jsonl(rows: &[Value]) -> Result<()> {
    use io::Write;
    let mut out = io::BufWriter::new(io::stdout().lock());
    for row in rows {
        serde_json::to_writer(&mut out, row)?;
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}
