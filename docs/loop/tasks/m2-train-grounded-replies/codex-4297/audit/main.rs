#![forbid(unsafe_code)]

mod controls;
mod gate;
mod held;
mod identity;
mod packet;
mod panel;
#[allow(dead_code)]
mod replay;
mod signed;
#[cfg(test)]
mod tests;
mod train;

use std::{
    env, fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use identity::{address, Ids, Pairs, Reference};
use packet::{Bank, Row, Sign};
use replay::{Arm, Coverage};
use signed::{Edge, Totals};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn check(ok: bool, detail: impl std::fmt::Display) -> Result<()> {
    if !ok {
        return Err(detail.to_string().into());
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|err| format!("{}: {err}", path.display()).into())
}

fn read_lines(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .enumerate()
        .map(|(row, line)| {
            serde_json::from_str(line)
                .map_err(|err| format!("{} line {}: {err}", path.display(), row + 1).into())
        })
        .collect()
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let output = fs::File::create_new(path)?;
    let mut output = BufWriter::new(output);
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    Ok(())
}

fn same(expected: &Value, observed: &Value, location: &str) -> Result<u64> {
    match (expected, observed) {
        (Value::Object(a), Value::Object(b)) if a.keys().eq(b.keys()) => {
            a.iter().try_fold(0, |n, (key, value)| {
                Ok(n + same(value, &b[key], &format!("{location}.{key}"))?)
            })
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            a.iter().zip(b).enumerate().try_fold(0, |n, (i, (a, b))| {
                Ok(n + same(a, b, &format!("{location}[{i}]"))?)
            })
        }
        _ => {
            check(
                expected == observed,
                format_args!("{location}: expected {expected}, observed {observed}"),
            )?;
            Ok(1)
        }
    }
}

fn equal_rows(wanted: &[Value], got: &[Value], location: &str) -> Result<u64> {
    check(
        wanted.len() == got.len(),
        format!("{location}: physical and reference row count differs"),
    )?;
    wanted
        .iter()
        .zip(got)
        .enumerate()
        .try_fold(0, |n, (i, (wanted, got))| {
            Ok(n + same(wanted, got, &format!("{location}[{i}]"))?)
        })
}

fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn replay(rows: &[Row], state: &[Sign; 2], banks: &[Bank; 3], arm: Arm) -> Vec<Value> {
    let mut coverage = Coverage::default();
    rows.iter()
        .enumerate()
        .map(|(i, row)| replay::predict(i, row, state, banks, arm, &mut coverage))
        .collect()
}

fn edge_list<'a>(
    rows: &'a [Row],
    labels: &[[[usize; 2]; 2]],
    state: &[Sign; 2],
) -> Result<Vec<Edge<'a>>> {
    rows.iter()
        .zip(labels)
        .flat_map(|(row, pair)| {
            pair.iter()
                .copied()
                .enumerate()
                .map(move |(sign, label)| Edge::new(row, sign, label, state))
        })
        .collect()
}

fn main() -> ExitCode {
    match run(env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>()
        .as_slice())
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[PathBuf]) -> Result<()> {
    match args {
        [mode, banks, parent, far_only, study_panel, training, physical, graded, out] if mode == Path::new("held") => {
            held::audit(banks,parent,far_only,study_panel,training,physical,graded,out)
        }
        [mode, core, out] if mode == Path::new("gate-fixture") => gate::fixture(core, out),
        [mode, study_panel, original, out, previous @ ..] if mode == Path::new("panel") => {
            panel::audit(study_panel, original, out, previous)
        }
        [mode, mixed, protected, omission, pooled, out] if mode == Path::new("antecedent") => {
            antecedent(mixed, protected, omission, pooled, out)
        }
        [mode, method, pool, geometry, labels, base, far_rows, physical, out] if mode == Path::new("round") => {
            let arm = method.to_str().ok_or("invalid objective")?;
            let rows = far_rows.to_str().ok_or("invalid far rows")?.parse()?;
            let report = train::round(arm, pool, geometry, labels, base, rows, physical)?;
            write_json(out, &report.value)
        }
        [mode, pooled, physical, out] if mode == Path::new("fixture") => train::fixture(pooled, physical, out),
        [mode, frozen, pooled, panel, physical, out] if mode == Path::new("train") => {
            train::study(frozen, pooled, panel, physical, out)
        }
        _ => Err("usage: bridge-audit antecedent MIXED OLD_PROTECTED OLD_OMISSION POOLED OUTPUT | round OBJECTIVE POOLED GEOMETRY LABELS BASE FAR_ROWS PHYSICAL OUTPUT | train FROZEN POOLED PANEL PHYSICAL OUTPUT".into()),
    }
}

fn antecedent(
    mixed: &Path,
    protected: &Path,
    omission: &Path,
    pooled: &Path,
    out: &Path,
) -> Result<()> {
    let physical = protected.join("measurement-v1");
    let historical = physical.join("train/protected/device/240");
    let shape = packet::text(&fs::read(mixed.join("blind/text.bin"))?)?;
    let labels = signed::labels(&fs::read(mixed.join("links.bin"))?, shape.len())?;
    let refs = Reference::read(
        mixed,
        &mixed.join("blind/coordinates.jsonl"),
        &shape,
        &labels,
    )?;
    check(
        refs.len() == 256
            && refs
                .iter()
                .enumerate()
                .all(|(i, r)| (i < 128) == (r.distance == "far")),
        "historical near/far split changed",
    )?;
    let frozen = fs::read(physical.join("states/protected.state"))?;
    check(
        frozen == fs::read(historical.with_extension("state"))?,
        "protected 240 state bytes differ",
    )?;
    let state = packet::state(&frozen)?;
    let banks = packet::pooled(&fs::read(pooled)?)?;
    let captured = read_lines(&historical.with_extension("reader.jsonl"))?;
    let neutral = omission.join("measurement-v1/raw/neutral.jsonl");
    check(
        fs::read(&neutral)? == fs::read(historical.with_extension("reader.jsonl"))?,
        "prior unmasked physical capture differs byte-for-byte",
    )?;
    let replayed = replay(&shape, &state, &banks, Arm::Learned);
    let joint_fields = equal_rows(&replayed, &captured, "predecessor joint capture")?;
    let premise = read_json(&physical.join("premise.json"))?;
    let protected_ids: Ids = serde_json::from_value(premise["protected"].clone())?;
    let graded = Pairs::read(&refs, &replayed, &protected_ids)?;
    let election = read_json(&physical.join("selection.json"))?;
    let expected_far: Ids = serde_json::from_value(election["protected"]["far_full"].clone())?;
    check(
        protected_ids.len() == 24
            && graded.full_near == protected_ids
            && expected_far.len() == 40
            && graded.full_far == expected_far,
        "historical protected-pair identity set changed",
    )?;
    let edges = edge_list(&shape, &labels, &state)?;
    let historic_edges: Vec<_> = edges
        .iter()
        .enumerate()
        .map(|(i, e)| Ok(e.record(i / 2, &e.read(&state)?, true)))
        .collect::<Result<_>>()?;
    let signed_fields = equal_rows(
        &historic_edges,
        &read_lines(&historical.with_extension("edges.jsonl"))?,
        "predecessor signed capture",
    )?;
    let ranks = read_lines(&omission.join("measurement-v1/raw/neutral.ranks.jsonl"))?;
    let mut totals = Totals::default();
    let mut seen = Ids::new();
    let num = |value: &Value| -> Result<usize> {
        Ok(usize::try_from(
            value.as_u64().ok_or("invalid predecessor ordinal")?,
        )?)
    };
    let mut ranked_fields = 0;
    for raw in &ranks {
        let row = num(&raw["row"])?;
        let sign = num(&raw["sign"])?;
        check(row < 128 && sign < 2, "wrong predecessor row or sign")?;
        let edge = edges
            .get(2 * row + sign)
            .ok_or("wrong predecessor row or sign")?;
        check(
            seen.insert(refs[row].id.clone()),
            "duplicate enrolled predecessor prompt",
        )?;
        let evaluation = edge.read(&state)?;
        totals.accumulate(evaluation.totals());
        let mut candidates = evaluation.wrong;
        candidates.push(evaluation.gold);
        candidates.sort_by_key(|candidate| candidate.site);
        // All ranks at one fixed original governor have the same lexical cost.
        // Compute it from the frozen lexical bank before comparing the full rank objects.
        let lexical = state[sign]
            .lexicon
            .iter()
            .map(|c| {
                c.iter()
                    .zip(&shape[row].words[edge.word].point)
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum::<u64>()
            })
            .min()
            .ok_or("empty antecedent lexical bank")?;
        let wanted = candidates.iter().map(|candidate| {
            json!({"site":candidate.site,"noun":address(shape[row].sites[candidate.site].address),
                "cost":[lexical, candidate.cost],"transport_center":candidate.center})
        }).collect::<Vec<_>>();
        ranked_fields += same(
            &json!({"row":row,"sign":sign,"word":edge.word,"governor":address(shape[row].words[edge.word].address),
            "omitted_site":null,"requested_site":null,"candidates":wanted}),
            raw,
            &format!("predecessor original ranks at {row}/{sign}"),
        )?;
    }
    check(
        ranks.len() == 48 && totals.cheaper == 176 && totals.losses == 48,
        "antecedent all-rival premise changed",
    )?;
    let exact_identities = refs.iter().enumerate().filter(|(_, row)| protected_ids.contains(&row.pair)).map(|(i,r)| {
        json!({"row":i,"pair":r.pair,"prompt_id":r.id,"admission":r.admission,"prompt_sha256":r.digest,"governors":r.governors,"nouns":r.gold})
    }).collect::<Vec<_>>();
    check(
        exact_identities.len() == 48,
        "protected prompt identity set differs",
    )?;
    write_json(
        out,
        &json!({"pass":true,"rows":shape.len(),"protected":protected_ids,"original_near_points":exact_identities,
        "full_far":expected_far,"signed_fields":signed_fields,"joint_fields":joint_fields,"ranked_fields":ranked_fields,
        "all_outrankers":totals.cheaper,"residual_rows":ranks.len(),"state_sha256":sha(&frozen)}),
    )
}
