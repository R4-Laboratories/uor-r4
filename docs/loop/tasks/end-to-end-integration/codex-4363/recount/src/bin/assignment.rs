//! Descriptive grading of captured CUDA words. No model or learned correction.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const TRAIN: &str = "f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a";

#[derive(Clone)]
struct Row {
    id: u64,
    cell: usize,
    fine: usize,
    admitted: bool,
    coarse: [i32; 64],
    reference: [i32; 64],
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

fn read_row(v: &Value) -> Result<Row> {
    let id = |field: &str| v[field].as_u64().ok_or_else(|| invalid(field));
    let slot = usize::try_from(id("target_slot")?)?;
    let cell = usize::try_from(id("target_cell")?)?;
    if slot / 64 != cell || cell >= 64 || v["train_sha256"] != TRAIN {
        return Err(invalid("record and TRAIN binding").into());
    }
    let scores = |field: &str| -> Result<[i32; 64]> {
        let words: Vec<i32> = serde_json::from_value(v[field].clone())?;
        Ok(words.try_into().map_err(|_| invalid(field))?)
    };
    Ok(Row {
        id: id("id")?,
        cell,
        fine: slot % 64,
        admitted: v["target_is_admitted"]
            .as_bool()
            .ok_or_else(|| invalid("admission"))?,
        coarse: scores("cuda_raw_coarse_q15")?,
        reference: scores("cuda_raw_target_fine_q15")?,
    })
}

struct Site<'a> {
    label: usize,
    scores: &'a [i32; 64],
}

#[derive(Default)]
struct Tally {
    positive: u64,
    tied: u64,
    negative: u64,
    both_strict: u64,
    total_delta: i64,
}

fn count(sites: &[Site<'_>], shift: usize, tally: &mut Tally) {
    for (i, left) in sites.iter().enumerate() {
        for (j, right) in sites.iter().enumerate().skip(i + 1) {
            let a = sites[(i + shift) % sites.len()].label;
            let b = sites[(j + shift) % sites.len()].label;
            if a == b {
                continue;
            }
            let d1 = i64::from(left.scores[a]) - i64::from(left.scores[b]);
            let d2 = i64::from(right.scores[b]) - i64::from(right.scores[a]);
            let delta = d1 + d2;
            tally.total_delta += delta;
            tally.positive += u64::from(delta > 0);
            tally.tied += u64::from(delta == 0);
            tally.negative += u64::from(delta < 0);
            tally.both_strict += u64::from(d1 > 0 && d2 > 0);
        }
    }
}

fn bounds(mut values: Vec<i64>) -> Value {
    values.sort_unstable();
    let n = values.len();
    if n == 0 {
        return Value::Null;
    }
    json!({"min":values[0],"median_lower":values[(n-1)/2],"median_upper":values[n/2],"max":values[n-1]})
}

fn describe(groups: BTreeMap<usize, Vec<Site<'_>>>) -> Result<Value> {
    let max_size = groups.values().map(Vec::len).max().unwrap_or(0);
    let mut registered = Tally::default();
    for group in groups.values() {
        count(group, 0, &mut registered);
    }
    let pairs = registered.positive + registered.tied + registered.negative;
    let mut offsets = Vec::new();
    for k in 1..max_size {
        let mut rotated = Tally::default();
        for group in groups.values().filter(|g| g.len() > 1) {
            // Every nonsingleton group gets a nonidentity rotation, including
            // when multiple shorter groups are pooled with the largest group.
            count(group, 1 + (k - 1) % (group.len() - 1), &mut rotated);
        }
        if rotated.positive + rotated.tied + rotated.negative != pairs {
            return Err(invalid("rotation changed pair denominator").into());
        }
        offsets.push(rotated);
    }
    let range = |field: fn(&Tally) -> i64| bounds(offsets.iter().map(field).collect());
    Ok(json!({
        "articles":groups.values().map(Vec::len).sum::<usize>(), "distinct_label_pairs":pairs,
        "correct_assignment":{"positive":registered.positive,"tied":registered.tied,"negative":registered.negative,
            "both_strict":registered.both_strict,"sum_q15":registered.total_delta},
        "rotations":{"count":offsets.len(),"positive":range(|x|x.positive as i64),
            "both_strict":range(|x|x.both_strict as i64),"sum_q15":range(|x|x.total_delta),
            "at_or_above_observed_positive":offsets.iter().filter(|x|x.positive >= registered.positive).count()},
    }))
}

fn analyze(rows: &[Row]) -> Result<Value> {
    let mut coarse = BTreeMap::new();
    let mut fine = BTreeMap::new();
    let mut admitted = BTreeMap::new();
    for r in rows {
        coarse.entry(0).or_insert_with(Vec::new).push(Site {
            label: r.cell,
            scores: &r.coarse,
        });
        fine.entry(r.cell).or_insert_with(Vec::new).push(Site {
            label: r.fine,
            scores: &r.reference,
        });
        if r.admitted {
            admitted.entry(r.cell).or_insert_with(Vec::new).push(Site {
                label: r.fine,
                scores: &r.reference,
            });
        }
    }
    Ok(
        json!({"coarse":describe(coarse)?,"conditional_fine":describe(fine)?,"admitted_fine":describe(admitted)?}),
    )
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).ok_or("assignment ROWS.jsonl")?;
    let mut arms: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let v: Value = serde_json::from_str(&line?)?;
        let arm = v["checkpoint"].as_str().ok_or("checkpoint")?;
        arms.entry(arm.to_owned()).or_default().push(read_row(&v)?);
    }
    if arms.keys().map(String::as_str).collect::<Vec<_>>() != ["frozen", "preserved", "reset"] {
        return Err(invalid("checkpoint roster").into());
    }
    let mut report = BTreeMap::new();
    let mut expected = None;
    for (arm, mut rows) in arms {
        rows.sort_by_cached_key(|r| (Sha256::digest(r.id.to_le_bytes()), r.id));
        let ids: BTreeSet<_> = rows.iter().map(|r| r.id).collect();
        let roster: Vec<_> = rows.iter().map(|r| (r.id, r.cell, r.fine)).collect();
        if ids.len() != 256 || rows.len() != 256 || expected.as_ref().is_some_and(|r| r != &roster)
        {
            return Err(invalid("unmatched article and gold slot roster").into());
        }
        expected = Some(roster);
        report.insert(arm, analyze(&rows)?);
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn common_bias_cancels_and_rotation_reverses_assignment() {
        let mut a = [0; 64];
        let mut b = [0; 64];
        a[..2].copy_from_slice(&[103, 1]);
        b[..2].copy_from_slice(&[100, 4]);
        let sites = [
            Site {
                label: 0,
                scores: &a,
            },
            Site {
                label: 1,
                scores: &b,
            },
        ];
        let mut tally = Tally::default();
        count(&sites, 0, &mut tally);
        assert_eq!(
            (tally.positive, tally.both_strict, tally.total_delta),
            (1, 0, 6)
        );
        let mut swapped = Tally::default();
        count(&sites, 1, &mut swapped);
        assert_eq!((swapped.negative, swapped.total_delta), (1, -6));
        let same = [
            Site {
                label: 0,
                scores: &a,
            },
            Site {
                label: 1,
                scores: &a,
            },
        ];
        let mut tied = Tally::default();
        count(&same, 0, &mut tied);
        assert_eq!((tied.tied, tied.total_delta), (1, 0));
    }
}
