// Standalone replay of finite certificate records. Rebuild the comparisons from
// original target addresses independently of the solver's coefficient emitter.
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Axis = [i64; 32];
type Edge = (String, Option<usize>, Option<usize>);
fn need(ok: bool, detail: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(detail.into())
    }
}
fn lines<T: serde::de::DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|x| Ok(serde_json::from_str(&x?)?))
        .collect()
}
#[derive(Deserialize)]
struct Point {
    pos: usize,
    id: u32,
}
#[derive(Deserialize)]
struct Key {
    case_id: String,
    pair_id: String,
    target: Option<Point>,
}
#[derive(Deserialize)]
struct Candidate {
    pos: usize,
    id: u32,
    score: i64,
    real_signed: Axis,
    fixed_signed: Axis,
    swapped_signed: Axis,
    no_question_signed: Axis,
}
#[derive(Deserialize)]
struct Threshold {
    no_read_score: i64,
}
#[derive(Deserialize)]
struct Capture {
    case_id: String,
    candidates: Vec<Candidate>,
    expanded_frozen: Threshold,
}
#[derive(Deserialize, PartialEq, Debug)]
struct Expr {
    coefficients: Axis,
    minimum: i64,
}
#[derive(Deserialize)]
struct Inequality {
    case_id: String,
    winner: Option<usize>,
    loser: Option<usize>,
    #[serde(flatten)]
    expr: Expr,
}
#[derive(Deserialize)]
struct Term {
    multiplier: String,
    inequality: Inequality,
}
#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Verdict {
    Feasible {
        words: Axis,
    },
    Impossible {
        terms: Vec<Term>,
        weighted_minimum: String,
        box_maximum: String,
    },
    Unknown {
        reason: String,
    },
}
#[derive(Deserialize)]
struct Record {
    arm: String,
    offsets: String,
    scope: String,
    margin: i64,
    level: String,
    id: String,
    requirement_count: usize,
    actual_witness_pass: Option<bool>,
    verdict: Verdict,
}

fn value(row: &Capture, origin: Option<usize>, r: &Record) -> Result<(i64, Axis)> {
    let (bias, axes) = match origin {
        None => (row.expanded_frozen.no_read_score, [0; 32]),
        Some(pos) => {
            let c = row
                .candidates
                .iter()
                .find(|c| c.pos == pos)
                .ok_or("absent original site")?;
            let x = match r.arm.as_str() {
                "actual" => c.real_signed,
                "fixed" => c.fixed_signed,
                "swapped" => c.swapped_signed,
                "removed" => c.no_question_signed,
                "zero_weight" => [0; 32],
                _ => return Err("unknown arm".into()),
            };
            (c.score, x)
        }
    };
    match r.offsets.as_str() {
        "frozen" => Ok((bias, axes)),
        "zero" => Ok((0, axes)),
        _ => Err("unknown offsets".into()),
    }
}

fn check_record(
    r: &Record,
    keys: &[Key],
    captures: &BTreeMap<String, Capture>,
    actual: Option<&Axis>,
    replay_actual: bool,
) -> Result<()> {
    let mut edges = BTreeMap::new();
    for k in keys {
        let take = match r.level.as_str() {
            "row" => k.case_id == r.id,
            "pair" => k.pair_id == r.id,
            "panel" => true,
            "panel_positive_only" => k.target.is_some(),
            "panel_omit_row" => k.case_id != r.id,
            _ => return Err("bad group level".into()),
        };
        if !take {
            continue;
        }
        let row = &captures[&k.case_id];
        let winner = k.target.as_ref().map(|t| t.pos);
        if let Some(t) = &k.target {
            need(
                row.candidates
                    .iter()
                    .any(|c| c.pos == t.pos && c.id == t.id),
                "gold ID/address mismatch",
            )?;
        }
        let losers = match r.scope.as_str() {
            "designated" => {
                let other = keys
                    .iter()
                    .find(|o| o.pair_id == k.pair_id && o.case_id != k.case_id)
                    .ok_or("unpaired row")?;
                vec![other.target.as_ref().map(|t| t.pos)]
            }
            "supported_gold_vs_noread" => vec![None],
            "all" | "supported_gold_vs_sites" => row
                .candidates
                .iter()
                .map(|c| Some(c.pos))
                .chain((r.scope == "all").then_some(None))
                .filter(|p| *p != winner)
                .collect(),
            _ => return Err("unknown scope".into()),
        };
        let (b, x) = value(row, winner, r)?;
        for loser in losers {
            need(loser != winner, "self-comparison")?;
            let (a, y) = value(row, loser, r)?;
            edges.insert(
                (k.case_id.clone(), winner, loser),
                Expr {
                    coefficients: std::array::from_fn(|i| x[i] - y[i]),
                    minimum: r.margin + a - b,
                },
            );
        }
    }
    need(
        !edges.is_empty() && edges.len() == r.requirement_count,
        "incomplete comparison coverage",
    )?;
    let passes = |words: &Axis| {
        words.iter().all(|w| (-4096..=4096).contains(w))
            && edges.values().all(|e| {
                words
                    .iter()
                    .zip(e.coefficients)
                    .map(|(w, x)| w * x)
                    .sum::<i64>()
                    >= e.minimum
            })
    };
    if replay_actual {
        need(
            actual.map(passes) == r.actual_witness_pass,
            "incorrect actual-witness replay",
        )?;
    }
    match &r.verdict {
        Verdict::Feasible { words } => need(passes(words), "invalid bounded integer witness")?,
        Verdict::Impossible {
            terms,
            weighted_minimum,
            box_maximum,
        } => {
            let mut combined = [0_i128; 32];
            let mut lower = 0i128;
            for t in terms {
                let a = &t.inequality;
                let key: Edge = (a.case_id.clone(), a.winner, a.loser);
                let e = edges
                    .get(&key)
                    .ok_or("certificate addressed an unrelated inequality")?;
                need(
                    e == &a.expr,
                    "certificate differs from independent coefficients or offsets",
                )?;
                let m: i128 = t.multiplier.parse()?;
                need(m > 0, "nonpositive multiplier")?;
                lower = lower
                    .checked_add(
                        m.checked_mul(i128::from(e.minimum))
                            .ok_or("integer overflow")?,
                    )
                    .ok_or("integer overflow")?;
                for (v, x) in combined.iter_mut().zip(e.coefficients) {
                    *v = v
                        .checked_add(m.checked_mul(i128::from(x)).ok_or("integer overflow")?)
                        .ok_or("integer overflow")?;
                }
            }
            let upper = combined
                .iter()
                .try_fold(0_i128, |sum, x| sum.checked_add(x.checked_abs()?))
                .and_then(|l1| l1.checked_mul(4096))
                .ok_or("integer overflow")?;
            need(
                lower.to_string() == *weighted_minimum
                    && upper.to_string() == *box_maximum
                    && lower > upper,
                "certificate fails box upper bound",
            )?;
        }
        Verdict::Unknown { reason } => need(!reason.is_empty(), "unknown lacks diagnostic")?,
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    need(
        args.len() >= 3,
        "usage: verify-signed-verdicts FEATURES KEYS RESULTS [--positive|--omit CASE_ID|--split]",
    )?;
    let rows: Vec<Capture> = lines(&args[0])?;
    let n = rows.len();
    let captures: BTreeMap<_, _> = rows.into_iter().map(|r| (r.case_id.clone(), r)).collect();
    let keys: Vec<Key> = lines(&args[1])?;
    need(
        captures.len() == n
            && keys.len() == n
            && keys
                .iter()
                .map(|k| &k.case_id)
                .collect::<BTreeSet<_>>()
                .len()
                == n,
        "non-unique identity",
    )?;
    let mode: Vec<_> = args[3..].iter().map(String::as_str).collect();
    need(
        matches!(
            mode.as_slice(),
            [] | ["--positive"] | ["--split"] | ["--omit", _]
        ),
        "unknown replay mode",
    )?;
    if let ["--omit", id] = mode.as_slice() {
        need(
            keys.iter().any(|k| k.case_id == *id),
            "omitted row does not exist",
        )?;
    }
    let split = mode == ["--split"];
    let records: Vec<Record> = if split {
        let mut split_records = Vec::new();
        let positive_ids: BTreeSet<_> = keys
            .iter()
            .filter(|k| k.target.is_some())
            .map(|k| &k.case_id)
            .collect();
        for mut row in lines::<Value>(&args[2])? {
            if row["scope"] == "original_singletons_all_rivals" {
                continue; // Replayed directly from the full verdict file.
            }
            let case_ids: Vec<String> = serde_json::from_value(row["case_ids"].take())?;
            need(
                case_ids.len() == positive_ids.len()
                    && case_ids.iter().collect::<BTreeSet<_>>() == positive_ids,
                "wrong supported row set",
            )?;
            row["level"] = json!("panel_positive_only");
            row["id"] = json!("supported");
            split_records.push(serde_json::from_value(row)?);
        }
        split_records
    } else {
        lines(&args[2])?
    };
    let witness_key = |r: &Record| {
        (
            r.offsets.clone(),
            r.scope.clone(),
            r.margin,
            r.level.clone(),
            r.id.clone(),
        )
    };
    let actual_tables: BTreeMap<_, _> = records
        .iter()
        .filter_map(|r| match (&r.verdict, r.arm.as_str()) {
            (Verdict::Feasible { words }, "actual") => Some((witness_key(r), words)),
            _ => None,
        })
        .collect();
    let mut seen = BTreeSet::new();
    let mut stats = BTreeMap::new();
    let mut counts = BTreeMap::new();
    let mut singles = BTreeMap::new();
    for r in &records {
        let registered = match mode.as_slice() {
            [] => {
                ["row", "pair", "panel"].contains(&r.level.as_str())
                    && (r.level != "panel" || r.id == "shared")
                    && ["designated", "all"].contains(&r.scope.as_str())
            }
            ["--split"] => {
                ["supported_gold_vs_sites", "supported_gold_vs_noread"].contains(&r.scope.as_str())
                    && ["fixed", "actual"].contains(&r.arm.as_str())
            }
            ["--positive"] => {
                r.level == "panel_positive_only"
                    && r.id == "supported"
                    && r.scope == "all"
                    && r.margin == 1
            }
            ["--omit", id] => {
                r.level == "panel_omit_row" && r.id == *id && r.scope == "all" && r.margin == 1
            }
            _ => false,
        };
        need(
            registered
                && ["actual", "fixed", "swapped", "removed", "zero_weight"]
                    .contains(&r.arm.as_str())
                && ["frozen", "zero"].contains(&r.offsets.as_str()),
            "unregistered comparison",
        )?;
        need(
            seen.insert((&r.offsets, &r.scope, &r.arm, r.margin, &r.level, &r.id)),
            "repeated group",
        )?;
        need([1, 8].contains(&r.margin), "unregistered margin")?;
        check_record(
            r,
            &keys,
            &captures,
            actual_tables.get(&witness_key(r)).copied(),
            !split,
        )
        .map_err(|e| {
            format!(
                "{} {} {} {} {}: {e}",
                r.arm, r.offsets, r.scope, r.level, r.id
            )
        })?;
        let status = match r.verdict {
            Verdict::Feasible { .. } => "feasible",
            Verdict::Impossible { .. } => "impossible",
            Verdict::Unknown { .. } => "unknown",
        };
        *stats.entry(status).or_insert(0usize) += 1;
        let group = (&r.offsets, &r.scope, r.margin, &r.arm, &r.level);
        let entry = counts.entry(group).or_insert([0usize; 4]);
        entry[0] += 1;
        entry[1] += usize::from(status == "feasible");
        entry[2] += usize::from(status == "unknown");
        entry[3] += usize::from(r.actual_witness_pass == Some(true));
        if r.level == "row" {
            singles.insert(
                (&r.id, &r.offsets, &r.scope, r.margin, r.arm.as_str()),
                status,
            );
        }
    }
    let pairs = keys
        .iter()
        .map(|k| &k.pair_id)
        .collect::<BTreeSet<_>>()
        .len();
    need(
        records.len()
            == if mode.is_empty() {
                40 * (n + pairs + 1)
            } else if split {
                16
            } else {
                10
            },
        "missing registered arm or group",
    )?;
    for ((id, offset, scope, m, arm), status) in &singles {
        if *arm == "actual" {
            let fixed = singles
                .get(&(id, offset, scope, *m, "fixed"))
                .ok_or("missing fixed control")?;
            need(
                *fixed == "unknown" || *status == "unknown" || status == fixed,
                "signed gauge single-row invariant failed",
            )?;
        }
    }
    let summaries: Vec<_> = counts.into_iter().map(|((offsets, scope, margin, arm, level), [total, feasible, unknown, same_table_pass])| {
        json!({ "offsets": offsets, "scope": scope, "margin":margin, "arm":arm, "level":level,
            "total":total, "feasible":feasible, "unknown":unknown, "actual_table_still_passes":same_table_pass })
    }).collect();
    let out: Value = json!({"cases":n,"pairs":pairs,"records":records.len(),"validated":stats,"groups":summaries});
    println!("{out}");
    Ok(())
}
