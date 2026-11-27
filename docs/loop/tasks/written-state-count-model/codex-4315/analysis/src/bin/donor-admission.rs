//! Count agreement with the proposed source supervision directly in the corrected donor tape.
#![forbid(unsafe_code)]
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};

#[derive(Deserialize)]
struct Donor {
    prefix: Vec<u32>,
    target: u32,
    genuine_assistant: bool,
}

fn facts(ids: &[u32]) -> Option<&[u32]> {
    const ASSISTANT: [u32; 6] = [201, 35, 560, 652, 714, 28];
    let body = ids
        .strip_prefix(&[0, 55, 2728, 28])?
        .strip_suffix(&ASSISTANT)?;
    if body.iter().any(|&v| v < 2)
        || body.windows(6).any(|w| w == ASSISTANT)
        || body.windows(4).any(|w| w == [201, 55, 2728, 28])
    {
        return None;
    }
    let mut breaks = body
        .windows(2)
        .enumerate()
        .filter(|(_, w)| *w == [201, 201])
        .map(|(i, _)| i);
    let split = breaks.next()?;
    if split == 0 || split + 2 >= body.len() || breaks.next().is_some() {
        return None;
    }
    Some(&body[..split])
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [tape, classes, out] = args.as_slice() else {
        return Err("usage: donor-admission DONOR.jsonl CLASSES.u8 NEW_RECEIPT.json".into());
    };
    let classes = fs::read(classes)?;
    if classes.len() != 4096
        || format!("{:x}", Sha256::digest(&classes))
            != "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23"
    {
        return Err("invalid canonical classes".into());
    }
    let mut rows = 0;
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut learned = BTreeSet::new();
    let mut openings = 0;
    let mut parsed = 0;
    let mut positive_target_ids = BTreeSet::new();
    for line in BufReader::new(File::open(tape)?).lines() {
        let row: Donor = serde_json::from_str(&line?)?;
        if row.prefix.last() != Some(&28) || row.target >= 4096 {
            return Err("bad donor row".into());
        }
        rows += 1;
        let role = row.prefix.ends_with(&[201, 35, 560, 652, 714, 28]);
        if role != row.genuine_assistant {
            return Err("genuine marker versus provenance disagreement".into());
        }
        let class = classes[row.target as usize];
        let kind = if !role {
            "other_colon"
        } else {
            openings += 1;
            positive_target_ids.insert(row.target);
            match facts(&row.prefix) {
                None => "unparsed_genuine",
                Some(span) => {
                    parsed += 1;
                    if class == 0 || class == 9 {
                        "nonlexical_genuine"
                    } else {
                        match span.iter().filter(|&&v| v == row.target).count() {
                            0 => "source_absent",
                            1 => {
                                learned.insert(row.target);
                                "source_unique"
                            }
                            _ => "source_ambiguous",
                        }
                    }
                }
            }
        };
        *counts.entry(kind).or_default() += 1;
    }
    if rows != 118915 || openings != 4345 {
        return Err("incomplete registered donor".into());
    }
    let result = json!({"schema":"uor-r4.canonical-donor-source-admission/1","rows":rows,"genuine":openings,"parsed_genuine":parsed,"partition":counts,"supported_source_target_ids":learned,"distinct_genuine_target_ids":positive_target_ids.len()});
    let mut out = File::create_new(out)?;
    serde_json::to_writer_pretty(&mut out, &result)?;
    writeln!(out)?;
    println!("{result}");
    Ok(())
}
