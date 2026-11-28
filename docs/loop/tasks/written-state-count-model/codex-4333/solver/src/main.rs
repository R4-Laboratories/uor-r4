#![forbid(unsafe_code)]

mod solver;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;
use solver::{CAPACITY, Inequality, Verdict, meets, solve};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Write},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Features = [i64; CAPACITY];

fn check(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.into());
    }
    Ok(())
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
struct Site {
    pos: usize,
    id: u32,
}

#[derive(Deserialize)]
struct Key {
    case_id: String,
    pair_id: String,
    kind: String,
    target: Option<Site>,
    sites: Vec<Site>,
}

#[derive(Deserialize)]
struct Candidate {
    pos: usize,
    id: u32,
    score: i64,
    real_signed: Features,
    fixed_signed: Features,
    swapped_signed: Features,
    no_question_signed: Features,
}

#[derive(Deserialize)]
struct Frozen {
    no_read_score: i64,
}

#[derive(Deserialize)]
struct Row {
    case_id: String,
    prefix: Vec<u32>,
    candidates: Vec<Candidate>,
    expanded_frozen: Frozen,
}

#[derive(Clone, Copy)]
enum Arm {
    Actual,
    Fixed,
    Swapped,
    Removed,
    ZeroWeight,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Actual => "actual",
            Self::Fixed => "fixed",
            Self::Swapped => "swapped",
            Self::Removed => "removed",
            Self::ZeroWeight => "zero_weight",
        }
    }
}

struct Member<'a> {
    key: &'a Key,
    row: &'a Row,
}

impl Member<'_> {
    fn find(&self, site: Site) -> Result<&Candidate> {
        self.row
            .candidates
            .iter()
            .find(|c| (c.pos, c.id) == (site.pos, site.id))
            .ok_or_else(|| {
                format!(
                    "missing original site ({}, {}) in {}",
                    site.pos, site.id, self.key.case_id
                )
                .into()
            })
    }
}

fn read<T: DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn validate_features(features: &Features) -> bool {
    features.chunks_exact(8).all(|block| {
        block
            .iter()
            .all(|&coordinate| (-1..=1).contains(&coordinate))
            && block.iter().filter(|&&coordinate| coordinate != 0).count() == 2
    })
}

fn validate(member: &Member<'_>) -> Result<()> {
    let mut seen_positions = BTreeSet::new();
    check(!member.row.candidates.is_empty(), "empty candidate list")?;
    for c in &member.row.candidates {
        check(
            seen_positions.insert(c.pos) && member.row.prefix.get(c.pos) == Some(&c.id),
            "candidate address or ID disagrees with captured prefix",
        )?;
        check(i32::try_from(c.score).is_ok(), "invalid frozen site score")?;
        check(
            [
                &c.real_signed,
                &c.fixed_signed,
                &c.swapped_signed,
                &c.no_question_signed,
            ]
            .into_iter()
            .all(validate_features),
            "signed features must have two unit coordinates per block",
        )?;
    }
    check(
        i32::try_from(member.row.expanded_frozen.no_read_score).is_ok(),
        "invalid frozen NoRead score",
    )?;
    check(
        !member.key.sites.is_empty()
            && member
                .key
                .target
                .is_none_or(|s| member.key.sites.contains(&s)),
        "target must be among annotated sites",
    )?;
    for &s in &member.key.sites {
        member.find(s)?;
    }
    Ok(())
}

fn inequalities(
    member: &Member<'_>,
    counterpart: &Member<'_>,
    arm: Arm,
    frozen: bool,
    whole: bool,
    margin: i64,
) -> Result<Vec<Inequality>> {
    let nr = if frozen {
        member.row.expanded_frozen.no_read_score
    } else {
        0
    };
    let val = |c: Option<&Candidate>| match c {
        None => (None, nr, [0; CAPACITY]),
        Some(c) => (
            Some(c.pos),
            if frozen { c.score } else { 0 },
            match arm {
                Arm::Actual => c.real_signed,
                Arm::Fixed => c.fixed_signed,
                Arm::Swapped => c.swapped_signed,
                Arm::Removed => c.no_question_signed,
                Arm::ZeroWeight => [0; CAPACITY],
            },
        ),
    };
    let mut rows = Vec::new();
    let mut beats = |winner: Option<&Candidate>, loser: Option<&Candidate>| {
        let (winner, winner_offset, winner_features) = val(winner);
        let (loser, loser_offset, loser_features) = val(loser);
        rows.push(Inequality {
            case_id: member.key.case_id.clone(),
            winner,
            loser,
            coefficients: std::array::from_fn(|i| winner_features[i] - loser_features[i]),
            minimum: margin + loser_offset - winner_offset,
        });
    };
    match member.key.target {
        Some(site) => {
            let gold = Some(member.find(site)?);
            if whole {
                beats(gold, None);
                for rival in &member.row.candidates {
                    if rival.pos != site.pos {
                        beats(gold, Some(rival));
                    }
                }
            } else if let Some(other) = counterpart.key.target {
                check(
                    other.pos != site.pos,
                    "positive contrast needs two distinct occurrences",
                )?;
                beats(gold, Some(member.find(other)?));
            } else {
                beats(gold, None);
            }
        }
        None => {
            if whole {
                for rival in &member.row.candidates {
                    beats(None, Some(rival));
                }
            } else {
                let other = counterpart
                    .key
                    .target
                    .ok_or("support contrast lacks a positive member")?;
                beats(None, Some(member.find(other)?));
            }
        }
    }
    Ok(rows)
}

struct Conditions {
    frozen: bool,
    whole: bool,
    margin: i64,
}

fn emit(
    output: &mut impl Write,
    z3: &str,
    members: &[(&Member<'_>, &Member<'_>)],
    condition: &Conditions,
    level: &str,
    id: &str,
) -> Result<()> {
    let mut actual_words = None;
    let mut actual_status = None;
    for arm in [
        Arm::Actual,
        Arm::Fixed,
        Arm::Swapped,
        Arm::Removed,
        Arm::ZeroWeight,
    ] {
        let rows: Vec<_> = members
            .iter()
            .map(|(member, other)| {
                inequalities(
                    member,
                    other,
                    arm,
                    condition.frozen,
                    condition.whole,
                    condition.margin,
                )
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        let verdict = solve(z3, &rows, matches!(arm, Arm::ZeroWeight))?;
        let status = match &verdict {
            Verdict::Feasible { .. } => Some(true),
            Verdict::Impossible { .. } => Some(false),
            Verdict::Unknown { .. } => None,
        };
        if matches!(arm, Arm::Actual) {
            if let Verdict::Feasible { words } = &verdict {
                actual_words = Some(*words);
            }
            actual_status = status;
        }
        if level == "row"
            && matches!(arm, Arm::Fixed)
            && let (Some(actual), Some(fixed)) = (actual_status, status)
        {
            check(actual == fixed, "actual/fixed one-row feasibility mismatch")?;
        }
        serde_json::to_writer(
            &mut *output,
            &json!({
                "offsets": if condition.frozen { "frozen" } else { "zero" },
                "scope": if condition.whole { "all" } else { "designated" },
                "margin": condition.margin,
                "arm": arm.name(),
                "level": level,
                "id": id,
                "requirement_count": rows.len(),
                "actual_witness_pass": actual_words.map(|w| meets(&w, &rows)),
                "verdict": verdict,
            }),
        )?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let (filter_level, omitted_id) = match args.first().map(String::as_str) {
        Some("--panel-positive-only") => {
            args.remove(0);
            (Some("panel_positive_only"), None)
        }
        Some("--panel-omit-row") => {
            check(args.len() >= 2, "--panel-omit-row requires a case ID")?;
            args.remove(0);
            (Some("panel_omit_row"), Some(args.remove(0)))
        }
        _ => (None, None),
    };
    check(
        args.len() >= 3 && args.len() % 2 == 1,
        "usage: signed-site-gate [--panel-positive-only | --panel-omit-row CASE_ID] Z3 CAPTURE.jsonl KEY.jsonl [CAPTURE.jsonl KEY.jsonl ...]",
    )?;
    let mut captured = BTreeMap::new();
    let mut keys = Vec::new();
    for pair in args[1..].chunks_exact(2) {
        for row in read::<Row>(&pair[0])? {
            check(
                captured.insert(row.case_id.clone(), row).is_none(),
                "duplicate capture ID",
            )?;
        }
        keys.extend(read::<Key>(&pair[1])?);
    }
    check(
        !keys.is_empty() && keys.len() == captured.len(),
        "key/capture cardinality mismatch",
    )?;
    let mut pairs = BTreeMap::<&str, Vec<Member<'_>>>::new();
    let mut seen_ids = BTreeSet::new();
    for key in &keys {
        check(seen_ids.insert(&key.case_id), "duplicate key ID")?;
        let row = captured.get(&key.case_id).ok_or("missing capture")?;
        let member = Member { key, row };
        validate(&member)?;
        pairs.entry(&key.pair_id).or_default().push(member);
    }
    let mut panel = Vec::new();
    for pair in pairs.values() {
        check(
            pair.len() == 2 && pair[0].key.kind == pair[1].key.kind,
            "each pair must have two members of the same kind",
        )?;
        panel.extend([(&pair[0], &pair[1]), (&pair[1], &pair[0])]);
    }
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    if let Some(level) = filter_level {
        let filtered: Vec<_> = panel
            .iter()
            .copied()
            .filter(|(member, _)| match omitted_id.as_deref() {
                Some(id) => member.key.case_id != id,
                None => member.key.target.is_some(),
            })
            .collect();
        check(
            !filtered.is_empty() && filtered.len() < panel.len(),
            "empty or unchanged filtered panel",
        )?;
        check(
            omitted_id.is_none() || filtered.len() + 1 == panel.len(),
            "expected exactly one omitted row",
        )?;
        for frozen in [true, false] {
            let condition = Conditions {
                frozen,
                whole: true,
                margin: 1,
            };
            emit(
                &mut output,
                &args[0],
                &filtered,
                &condition,
                level,
                omitted_id.as_deref().unwrap_or("supported"),
            )?;
        }
        return Ok(());
    }
    for whole in [true, false] {
        for frozen in [true, false] {
            for margin in [1, 8] {
                let condition = Conditions {
                    frozen,
                    whole,
                    margin,
                };
                emit(&mut output, &args[0], &panel, &condition, "panel", "shared")?;
            }
        }
    }
    for whole in [true, false] {
        for frozen in [true, false] {
            for margin in [1, 8] {
                let condition = Conditions {
                    frozen,
                    whole,
                    margin,
                };
                for (id, pair) in &pairs {
                    let members = [(&pair[0], &pair[1]), (&pair[1], &pair[0])];
                    emit(&mut output, &args[0], &members, &condition, "pair", id)?;
                    for &(member, counterpart) in &members {
                        emit(
                            &mut output,
                            &args[0],
                            &[(member, counterpart)],
                            &condition,
                            "row",
                            &member.key.case_id,
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}
