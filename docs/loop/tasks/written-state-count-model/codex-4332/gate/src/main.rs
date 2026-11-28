mod solver;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use solver::{solve, Constraint, Verdict, REFERENCE};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn check(ok: bool, msg: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(msg.into())
    }
}
#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
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
    score: i32,
    recurrence_code: u16,
    real_mask: usize,
    fixed_mask: usize,
}
#[derive(Deserialize)]
struct Base {
    no_read_score: i32,
}
#[derive(Deserialize)]
struct Row {
    case_id: String,
    query_code: u16,
    fixed_query_code: u16,
    no_question_code: u16,
    prefix: Vec<u32>,
    candidates: Vec<Candidate>,
    expanded_frozen: Base,
}
fn read<T: DeserializeOwned>(p: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(p)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}
fn mask(a: u16, b: u16) -> usize {
    (0..4)
        .filter(|s| ((a ^ b) >> (4 * s)) & 15 == 0)
        .map(|s| 1 << s)
        .sum()
}
fn edges(
    key: &Key,
    row: &Row,
    opponent: &Key,
    q: u16,
    margin: i32,
    whole: bool,
) -> Result<Vec<Constraint>> {
    let mut choices = vec![];
    let find = |site: Site| {
        row.candidates
            .iter()
            .find(|c| (c.pos, c.id) == (site.pos, site.id))
            .ok_or("annotated site absent from capture")
    };
    for &s in &key.sites {
        find(s)?;
    }
    let nr = (
        REFERENCE,
        row.expanded_frozen.no_read_score,
        String::from("NoRead"),
    );
    let val = |c: &Candidate| (mask(c.recurrence_code, q), c.score, c.pos.to_string());
    let mut beats = |winner: &(usize, i32, String), loser: &(usize, i32, String)| {
        choices.push(Constraint {
            from: loser.0,
            to: winner.0,
            minimum: margin + loser.1 - winner.1,
            why: format!("{} {} > {}", key.case_id, winner.2, loser.2),
        });
    };
    match key.target {
        Some(s) => {
            let gold = val(find(s)?);
            if whole || opponent.target.is_none() {
                beats(&gold, &nr);
            }
            if whole {
                for c in &row.candidates {
                    if c.pos != s.pos {
                        beats(&gold, &val(c));
                    }
                }
            } else if let Some(other) = opponent.target {
                check(
                    other.pos != s.pos,
                    "requested reversal has duplicate target positions",
                )?;
                beats(&gold, &val(find(other)?));
            }
        }
        None => {
            if whole {
                for c in &row.candidates {
                    beats(&nr, &val(c));
                }
            } else {
                let other = opponent.target.ok_or("pair lacks a positive")?;
                beats(&nr, &val(find(other)?));
            }
        }
    }
    Ok(choices)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    check(
        args.len() == 3,
        "usage: request-site-gate CAPTURE.jsonl KEY.jsonl",
    )?;
    let captured: Vec<Row> = read(&args[1])?;
    let total = captured.len();
    let rows: BTreeMap<_, _> = captured
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let keys: Vec<Key> = read(&args[2])?;
    check(
        rows.len() == total && total == keys.len() && total > 0,
        "size mismatch",
    )?;
    let mut groups: BTreeMap<String, Vec<Key>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for k in keys {
        check(seen.insert(k.case_id.clone()), "duplicate key")?;
        let r = rows.get(&k.case_id).ok_or("missing capture")?;
        check(
            r.fixed_query_code == 0
                && r.candidates.iter().all(|c| {
                    r.prefix.get(c.pos) == Some(&c.id)
                        && c.real_mask == mask(c.recurrence_code, r.query_code)
                        && c.fixed_mask == mask(c.recurrence_code, 0)
                }),
            "capture mask or ID mismatch",
        )?;
        check(
            !k.sites.is_empty() && k.target.is_none_or(|s| k.sites.contains(&s)),
            "target not among annotated sites",
        )?;
        groups.entry(k.pair_id.clone()).or_default().push(k);
    }
    let mut results = Vec::<Value>::new();
    for scope in ["annotated", "whole"] {
        for margin in [1, 8] {
            let mut actual_witnesses = BTreeMap::new();
            for arm in ["actual", "fixed", "swapped", "no_question"] {
                let mut pair_results = vec![];
                let mut combined = vec![];
                let mut successes = 0;
                for (id, pair) in &groups {
                    check(pair.len() == 2, "expected two texts per question pair")?;
                    let mut required = vec![];
                    for i in 0..2 {
                        let k = &pair[i];
                        let other = &pair[1 - i];
                        let r = &rows[&k.case_id];
                        let q = match arm {
                            "actual" => r.query_code,
                            "fixed" => r.fixed_query_code,
                            "swapped" => rows[&other.case_id].query_code,
                            _ => r.no_question_code,
                        };
                        required.extend(edges(k, r, other, q, margin, scope == "whole")?);
                    }
                    let verdict = solve(&required);
                    successes += usize::from(matches!(verdict, Verdict::Feasible { .. }));
                    if let ("actual", Verdict::Feasible { active_words }) = (arm, &verdict) {
                        let mut words = active_words.clone();
                        words.push(0); // fixed NoRead reference
                        actual_witnesses.insert(id, words);
                    }
                    let actual_witness_pass = actual_witnesses
                        .get(id)
                        .map(|w| required.iter().all(|e| w[e.to] - w[e.from] >= e.minimum));
                    let contrast = if scope == "annotated" {
                        check(required.len() == 2, "expected one contrast per member")?;
                        let mut difference = [0; REFERENCE + 1];
                        // Each side is oriented in favor of its own target; these
                        // opposed inequalities cancel when aligned features coincide.
                        for edge in &required {
                            difference[edge.to] += 1;
                            difference[edge.from] -= 1;
                        }
                        Some(json!({"constraints":required, "aligned_difference":difference}))
                    } else {
                        None
                    };
                    pair_results.push(json!({"pair":id,"kind":pair[0].kind,"verdict":verdict,
                        "frozen_base_pass":required.iter().all(|e|e.minimum<=0),
                        "actual_witness_pass":actual_witness_pass,"contrast":contrast}));
                    combined.extend(required);
                }
                results.push(json!({"scope":scope,"margin":margin,"arm":arm,"feasible_pairs":successes,"pairs":pair_results,"joint":solve(&combined)}));
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"cases":total,"pairs":groups.len(),"outcomes":results})
        )?
    );
    Ok(())
}
