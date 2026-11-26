#![forbid(unsafe_code)]

mod packet;
mod replay;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use replay::{Arm, Coverage};

const SAMPLE_LIMIT: usize = 24;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn sha256(contents: &[u8]) -> String {
    format!("{:x}", Sha256::digest(contents))
}

fn read(path: &Path) -> Result<Vec<u8>> {
    Ok(fs::read(path).map_err(|err| io::Error::other(format!("{}: {err}", path.display())))?)
}

#[derive(Default)]
struct Comparison {
    checks: u64,
    errors: u64,
    wrong_rows: u64,
    cases: Vec<Value>,
    fields: BTreeMap<String, u64>,
}

impl Comparison {
    fn compare(&mut self, field: &str, row: usize, expected: &Value, observed: &Value) {
        match (expected, observed) {
            (Value::Object(expected), Value::Object(observed))
                if expected.keys().eq(observed.keys()) =>
            {
                for (key, sub) in expected {
                    self.compare(&format!("{field}.{key}"), row, sub, &observed[key]);
                }
            }
            (Value::Array(expected), Value::Array(observed))
                if expected.len() == observed.len() =>
            {
                for (i, (a, b)) in expected.iter().zip(observed).enumerate() {
                    self.compare(&format!("{field}[{i}]"), row, a, b);
                }
            }
            _ => {
                self.checks += 1;
                if expected != observed {
                    self.errors += 1;
                    *self.fields.entry(field.to_owned()).or_default() += 1;
                    if self.cases.len() < SAMPLE_LIMIT {
                        self.cases.push(json!({
                            "row": row, "field": field, "expected": expected, "observed": observed
                        }));
                    }
                }
            }
        }
    }
}

fn audit(
    rows: &[packet::Row],
    signs: &[packet::Sign; 2],
    banks: &[packet::Bank; 3],
    arm: Arm,
    trace: &[u8],
) -> Result<(u64, Value)> {
    let mut comparison = Comparison::default();
    let mut coverage = Coverage::default();
    let mut observed_lines = std::str::from_utf8(trace)?.lines();
    let mut selected_address_words = 0u64;
    for (ordinal, row) in rows.iter().enumerate() {
        let line = observed_lines.next().ok_or_else(|| {
            io::Error::other(format!(
                "{} row {ordinal} missing from GPU trace",
                arm.label()
            ))
        })?;
        let observed: Value = serde_json::from_str(line).map_err(|err| {
            io::Error::other(format!(
                "{} row {ordinal} malformed GPU JSON: {err}",
                arm.label()
            ))
        })?;
        let expected = replay::predict(ordinal, row, signs, banks, arm, &mut coverage);
        let before = comparison.errors;
        comparison.compare("$", ordinal, &expected, &observed);
        comparison.wrong_rows += u64::from(before != comparison.errors);
        if expected["tuple"]["ordinal"].is_number() {
            // Both signed governors and their noun endpoints plus the heading.
            // Each distinct occurrence contains four raw u32 span bounds.
            selected_address_words += 5 * 4;
        }
    }
    if observed_lines.next().is_some() {
        return Err(io::Error::other(format!("{} has extra GPU rows", arm.label())).into());
    }
    Ok((
        comparison.errors,
        json!({
            "trace_sha256": sha256(trace),
            "rows": rows.len(),
            "raw_field_comparisons": comparison.checks,
            "disagreements": comparison.errors,
            "rows_with_disagreement": comparison.wrong_rows,
            "fields_with_disagreement": comparison.fields,
            "first_disagreements": comparison.cases,
            "unique_winners": coverage.unique_winners,
            "selected_unique_address_words_replayed": selected_address_words,
            "tied_rows": coverage.tied_rows,
            "no_witness_rows": coverage.no_witness_rows,
            "minimal_joint_witnesses": coverage.minimal_witnesses,
            "candidate_triples": coverage.candidate_triples,
            "eligible_triples": coverage.eligible_triples,
            "eligible_word_site_sign_links": coverage.eligible_links,
            "signed_lexical_costs_recomputed": coverage.lexical_costs,
            "transport_costs_recomputed": coverage.transport_costs,
            "gap_costs_recomputed": coverage.gap_costs,
            "frozen_scope_costs_recomputed": coverage.frozen_costs,
            "i16_coordinate_differences_recomputed": coverage.coordinate_differences
        }),
    ))
}

fn run() -> Result<ExitCode> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [raw, pooled, state, learned, governor_gap] = args.as_slice() else {
        return Err(io::Error::other(
            "usage: textlinks-trace-audit TEXT.bin POOLED.words LEARNED.state LEARNED.jsonl GOVERNOR_GAP.jsonl",
        ).into());
    };
    let raw = read(raw)?;
    let pooled = read(pooled)?;
    let state = read(state)?;
    let learned = read(learned)?;
    let governor_gap = read(governor_gap)?;
    let rows = packet::text(&raw)?;
    let banks = packet::pooled(&pooled)?;
    let signs = packet::state(&state)?;
    let (learned_bad, learned_receipt) = audit(&rows, &signs, &banks, Arm::Learned, &learned)?;
    let (gap_bad, gap_receipt) = audit(&rows, &signs, &banks, Arm::GovernorGap, &governor_gap)?;
    let pass = learned_bad == 0 && gap_bad == 0;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": 1,
            "scope": "blind offline audit of recorded physical integer transitions",
            "pass": pass,
            "text_sha256": sha256(&raw),
            "pooled_words_sha256": sha256(&pooled),
            "trained_state_sha256": sha256(&state),
            "learned": learned_receipt,
            "governor_gap": gap_receipt
        }))?
    );
    Ok(ExitCode::from(if pass { 0 } else { 1 }))
}

fn main() -> ExitCode {
    match run() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("trace audit invalid: {error}");
            ExitCode::from(2)
        }
    }
}
