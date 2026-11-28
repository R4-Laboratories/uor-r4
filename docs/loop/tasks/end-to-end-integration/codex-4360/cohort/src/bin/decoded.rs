#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use uor_r4_tokenizer::ByteBpeTokenizer;

#[derive(Deserialize)]
struct Pair {
    pair_id: u64,
    a: Side,
    b: Side,
}

#[derive(Deserialize)]
struct Side {
    row: Gold,
    native: Arm,
    swap: Arm,
    zero: Arm,
}

#[derive(Deserialize)]
struct Gold {
    expected: Vec<u32>,
}

#[derive(Deserialize)]
struct Arm {
    emitted: Vec<u32>,
    stop_reason: String,
}

#[derive(Default, Serialize)]
struct Counts {
    rows: usize,
    cap: usize,
    token_first_match: usize,
    token_first_mismatch: usize,
    first_word_equal: usize,
    first_word_equal_with_token_first_mismatch: usize,
    begins_reference_word_bytes_with_token_first_mismatch: usize,
    begins_reference_word_bytes_only_with_token_first_mismatch: usize,
    trimmed_full_equal: usize,
    trimmed_full_equal_at_cap: usize,
    trimmed_full_equal_with_token_first_mismatch: usize,
    nonempty_output_prefix_of_reference_with_token_first_mismatch: usize,
    output_begins_full_reference_with_token_first_mismatch: usize,
    empty_output: usize,
    invalid_utf8_output: usize,
}

fn strip_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|b| !b.is_ascii_whitespace());
    let end = bytes.iter().rposition(|b| !b.is_ascii_whitespace());
    match (start, end) {
        (Some(start), Some(end)) => &bytes[start..=end],
        _ => &[],
    }
}

fn first_word(bytes: &[u8]) -> Option<&[u8]> {
    bytes
        .split(|b| b.is_ascii_whitespace())
        .find(|word| !word.is_empty())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [tokenizer_path, rows_path] = args.as_slice() else {
        return Err("usage: decoded-audit TOKENIZER_JSON ROWS_JSONL".into());
    };
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer_path)?)
        .ok_or("invalid tokenizer")?;
    if tokenizer.token_id("<|eos|>") != Some(1) || tokenizer.token_id("<|bos|>") != Some(0) {
        return Err("unexpected tokenizer special IDs".into());
    }
    let mut counts = BTreeMap::<&str, Counts>::new();
    let mut pair_ids = BTreeSet::new();
    for line in BufReader::new(File::open(rows_path)?).lines() {
        let pair: Pair = serde_json::from_str(&line?)?;
        if !pair_ids.insert(pair.pair_id) {
            return Err("duplicate pair ID".into());
        }
        for side in [pair.a, pair.b] {
            let expected = side
                .row
                .expected
                .strip_suffix(&[1])
                .ok_or("gold EOS missing")?;
            if expected.is_empty()
                || expected
                    .iter()
                    .any(|&id| id < 2 || id as usize >= tokenizer.vocab_size())
            {
                return Err("unexpected gold token IDs".into());
            }
            let expected_bytes = tokenizer.decode_bytes(expected);
            let reference = strip_ascii(&expected_bytes);
            let reference_word = first_word(reference).ok_or("empty reference text")?;
            for (name, arm) in [
                ("native", side.native),
                ("swap", side.swap),
                ("zero", side.zero),
            ] {
                let cap = arm.stop_reason == "cap";
                let content = match arm.stop_reason.as_str() {
                    "eos" => arm
                        .emitted
                        .strip_suffix(&[1])
                        .ok_or("emitted EOS missing")?,
                    "invalid_bos" => arm
                        .emitted
                        .strip_suffix(&[0])
                        .ok_or("emitted BOS missing")?,
                    "cap" => arm.emitted.as_slice(),
                    _ => return Err("unexpected stop reason".into()),
                };
                if content
                    .iter()
                    .any(|&id| id < 2 || id as usize >= tokenizer.vocab_size())
                {
                    return Err("unexpected emitted token IDs".into());
                }
                let emitted_bytes = tokenizer.decode_bytes(content);
                let emitted = strip_ascii(&emitted_bytes);
                let first_token_match = arm.emitted.first() == expected.first();
                let word_equal = first_word(emitted) == Some(reference_word);
                let word_prefix = emitted.starts_with(reference_word);
                let full_equal = emitted == reference;
                let stats = counts.entry(name).or_default();
                stats.rows += 1;
                stats.cap += usize::from(cap);
                stats.token_first_match += usize::from(first_token_match);
                stats.token_first_mismatch += usize::from(!first_token_match);
                stats.first_word_equal += usize::from(word_equal);
                stats.first_word_equal_with_token_first_mismatch +=
                    usize::from(word_equal && !first_token_match);
                stats.begins_reference_word_bytes_with_token_first_mismatch +=
                    usize::from(word_prefix && !first_token_match);
                stats.begins_reference_word_bytes_only_with_token_first_mismatch +=
                    usize::from(word_prefix && !word_equal && !first_token_match);
                stats.trimmed_full_equal += usize::from(full_equal);
                stats.trimmed_full_equal_at_cap += usize::from(full_equal && cap);
                stats.trimmed_full_equal_with_token_first_mismatch +=
                    usize::from(full_equal && !first_token_match);
                stats.nonempty_output_prefix_of_reference_with_token_first_mismatch += usize::from(
                    !emitted.is_empty() && reference.starts_with(emitted) && !first_token_match,
                );
                stats.output_begins_full_reference_with_token_first_mismatch +=
                    usize::from(emitted.starts_with(reference) && !first_token_match);
                stats.empty_output += usize::from(emitted.is_empty());
                stats.invalid_utf8_output +=
                    usize::from(std::str::from_utf8(&emitted_bytes).is_err());
            }
        }
    }
    if pair_ids.len() != 64 || counts.values().any(|arm| arm.rows != 128) {
        return Err("unexpected saved cohort size".into());
    }
    println!("{}", serde_json::to_string_pretty(&counts)?);
    Ok(())
}
