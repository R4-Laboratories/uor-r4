use crate::{
    Panel,
    wire::{
        Arm, BANKS, BOUND, Keys, Result, Span, WORDS, Words, gold_index, invalid, read_jsonl,
        score, winner,
    },
    write_new,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

const MAGIC: &[u8; 8] = b"UINTFIT1";
const EPOCHS: u32 = 32;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LockedPeriod {
    case_id: String,
    span: Span,
    original_ids: Vec<u32>,
}

pub fn eligible_train(panel: Panel, exclusion_path: &Path) -> Result<Panel> {
    let mut exclusions = BTreeMap::new();
    for row in read_jsonl::<LockedPeriod>(exclusion_path)? {
        if !row.original_ids.contains(&16)
            || row.span.end.checked_sub(row.span.start).map(usize::from)
                != Some(row.original_ids.len())
            || exclusions.insert(row.case_id.clone(), row).is_some()
        {
            return Err(invalid("invalid or duplicate locked-period exclusion"));
        }
    }
    let denominator = panel.rows.len();
    let mut rows = Vec::new();
    let mut excluded = Vec::new();
    for (row, mate) in panel.rows.into_iter().zip(panel.mates) {
        match (gold_index(&row), exclusions.remove(&row.case_id)) {
            (Some(_), None) => rows.push(row),
            (None, Some(record)) if row.gold == Some(record.span) && mate.is_none() => {
                excluded.push(record);
            }
            _ => {
                return Err(invalid(format!(
                    "{}: training coverage differs from exclusion manifest",
                    row.case_id
                )));
            }
        }
    }
    if !exclusions.is_empty() || rows.is_empty() {
        return Err(invalid("unused exclusions or no remaining training rows"));
    }
    let paired = rows
        .iter()
        .filter(|row| row.swapped_case_id.is_some())
        .count();
    println!(
        "{}",
        serde_json::json!({
            "full_denominator": denominator,
            "trained_rows": rows.len(),
            "permuted_rows": paired,
            "epochs": EPOCHS,
            "exclusion": "original_interval_contains_locked_period_id_16",
            "excluded": excluded,
        })
    );
    Panel::from_rows(rows)
}

pub fn arm_name(path: &Path) -> Result<Arm> {
    match path.to_str() {
        Some("geometry") => Ok(Arm::Geometry),
        Some("count") => Ok(Arm::Count),
        _ => Err(invalid("expected geometry or count arm")),
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: usize) -> Result<()> {
    bytes.extend_from_slice(&u32::try_from(value)?.to_le_bytes());
    Ok(())
}

pub fn prepare(panel: &Panel, arm: Arm, output: &Path, training: bool) -> Result<()> {
    let total_choices = panel.rows.iter().map(|row| row.choices.len()).sum();
    let mut bytes = MAGIC.to_vec();
    push_u32(&mut bytes, panel.rows.len())?;
    push_u32(&mut bytes, total_choices)?;
    bytes.extend_from_slice(&(if training { EPOCHS } else { 0 }).to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());

    let mut first = 0;
    for (row, mate) in panel.rows.iter().zip(&panel.mates) {
        push_u32(&mut bytes, first)?;
        push_u32(&mut bytes, row.choices.len())?;
        if training {
            let gold =
                gold_index(row).ok_or_else(|| invalid(format!("{}: gold absent", row.case_id)))?;
            let swapped = mate
                .map(|i| {
                    row.choices
                        .binary_search(&panel.rows[i].gold)
                        .map_err(|_| invalid(format!("{}: swapped label absent", row.case_id)))
                })
                .transpose()?
                .unwrap_or(gold);
            if row.choices.len() < 2 {
                return Err(invalid(format!(
                    "{}: training row has no rivals",
                    row.case_id
                )));
            }
            push_u32(&mut bytes, gold)?;
            push_u32(&mut bytes, swapped)?;
        } else {
            // Evaluation is gold blind, even when the interval coverage fails.
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&0_u32.to_le_bytes());
        }
        first += row.choices.len();
    }
    for row in &panel.rows {
        for keys in &row.keys[&arm] {
            for key in keys {
                bytes.extend_from_slice(&key.to_le_bytes());
            }
        }
    }
    write_new(output, &bytes)
}

pub fn read_words(path: &Path) -> Result<Words> {
    let bytes = fs::read(path)?;
    if bytes.len() != 2 * WORDS {
        return Err(invalid("expected exact 8192-byte weight payload"));
    }
    let mut words = [0; WORDS];
    for (word, raw) in words.iter_mut().zip(bytes.chunks_exact(2)) {
        *word = i16::from_le_bytes([raw[0], raw[1]]);
    }
    if words.iter().any(|word| !(-BOUND..=BOUND).contains(word)) {
        return Err(invalid("persisted weight outside training bounds"));
    }
    Ok(words)
}

pub fn zero_payload(output: &Path) -> Result<()> {
    write_new(output, &[0; 2 * WORDS])
}

fn parse_scores(raw: &[u8]) -> Result<Vec<i32>> {
    if !raw.len().is_multiple_of(4) {
        return Err(invalid("device result length is not a multiple of i32"));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|v| i32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .collect())
}

pub fn verify(panel: &Panel, cold: &Panel, arm: Arm, payload: &Path, output: &Path) -> Result<()> {
    if panel.rows.len() != cold.rows.len() {
        return Err(invalid("cold runtime case count changed"));
    }
    let words = read_words(payload)?;
    let raw = parse_scores(&fs::read(output)?)?;
    let mut results = raw.as_slice();
    let mut choices = 0;
    for (before, after) in panel.rows.iter().zip(&cold.rows) {
        if before.case_id != after.case_id
            || before.bounds != after.bounds
            || before.choices != after.choices
            || before.keys != after.keys
        {
            return Err(invalid(format!(
                "{}: candidate identity or keys drifted after export",
                before.case_id
            )));
        }
        let inputs = &before.keys[&arm];
        let Some((row, rest)) = results.split_at_checked(1 + inputs.len()) else {
            return Err(invalid("short device score stream"));
        };
        results = rest;
        let (device_selection, device_scores) = (&row[0], &row[1..]);
        let cold_score = after.cold.get(&arm).ok_or_else(|| {
            invalid(format!(
                "{}: cold client did not score selected arm",
                before.case_id
            ))
        })?;
        let (picked, _) = winner(&words, inputs);
        let best_candidate = device_scores[1..].iter().copied().max();
        if usize::try_from(*device_selection)? != picked
            || cold_score.selection.span != before.choices[picked]
            || device_scores[0] != cold_score.selection.no_read_score
            || device_scores[1..] != cold_score.candidates
            || best_candidate != cold_score.selection.best_candidate_score
            || inputs
                .iter()
                .zip(device_scores)
                .any(|(keys, &value): (&Keys, &i32)| score(&words, keys) != value)
        {
            return Err(invalid(format!(
                "{}: CUDA/independent scorer/cold client divergence",
                before.case_id
            )));
        }
        choices += inputs.len();
    }
    if !results.is_empty() {
        return Err(invalid("trailing data in device score stream"));
    }
    println!(
        "{{\"cases\":{},\"exact_scores\":{choices},\"banks\":{BANKS},\"payload_bytes\":{}}}",
        panel.rows.len(),
        2 * WORDS
    );
    Ok(())
}
