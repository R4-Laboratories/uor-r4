use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

type R<T> = Result<T, Box<dyn Error>>;
const ARMS: [&str; 5] = ["learned", "correct", "swapped", "absent", "marker"];
const SPACE: u32 = 223;

fn require(ok: bool, message: &str) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn json_rows<T: DeserializeOwned>(path: &Path) -> R<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let line = line?;
            Ok(serde_json::from_str(&line)
                .map_err(|error| format!("{} row {}: {error}", path.display(), index + 1))?)
        })
        .collect()
}

#[derive(Deserialize, PartialEq, Eq, Debug)]
struct Row {
    group_id: String,
    pair_id: String,
    assignment: usize,
    side: usize,
    cohort: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    raw_target: u32,
    raw_rival: u32,
    gold_pos: Option<usize>,
    rival_pos: Option<usize>,
    context_start: usize,
    context_end: usize,
    source_status: String,
}

#[derive(Deserialize, Debug)]
struct Candidate {
    pos: usize,
    id: u32,
    eligible: bool,
    score: Option<i32>,
    rank: Option<usize>,
    #[serde(default)]
    features: Option<Vec<usize>>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    source_pos: Option<usize>,
    gate_id: u32,
    emitted_id: u32,
    #[serde(default)]
    lexical: Option<Value>,
}

#[derive(Deserialize)]
struct Observation {
    #[serde(flatten)]
    row: Row,
    no_read_score: i32,
    winner_pos: Option<usize>,
    best_score: Option<i32>,
    best_ties: usize,
    candidates: Vec<Candidate>,
    arms: BTreeMap<String, Choice>,
}

#[derive(Default, Clone, Copy, Serialize)]
struct ArmMetrics {
    rows: usize,
    reads: usize,
    correct_source: usize,
    gold_selected_id: usize,
    correct_first_id: usize,
    comparable_first_ids: usize,
    no_answer_emitted_counterfactual_id: usize,
    writer_changed_gate: usize,
    source_and_emission: usize,
    emitted_space: usize,
    emitted_eos: usize,
    continue_to_lexical: usize,
    subsequent_lexical_correct: usize,
}

#[derive(Default, Serialize)]
struct Stratum {
    rows: usize,
    unique_gold: usize,
    absent_gold: usize,
    gold_eligible: usize,
    two_candidates_eligible: usize,
    gold_gt_rival: usize,
    gold_eq_rival: usize,
    gold_lt_rival: usize,
    gold_above_no_read: usize,
    higher_other_than_gold: usize,
    learned_source_alias: usize,
    learned_no_read: usize,
    correct_source_given_oracle_emits: usize,
    oracle_first_correct_on_unique: usize,
    learned_first_correct_given_oracle_emits: usize,
    factual_positions_scored: usize,
    arms: BTreeMap<String, ArmMetrics>,
}

#[derive(Clone, Copy)]
struct Flags {
    pref: bool,
    source: bool,
    first: bool,
    joint: bool,
    oracle: bool,
    marker: bool,
}

fn candidate<'a>(rows: &'a [Candidate], at: Option<usize>) -> Option<&'a Candidate> {
    at.and_then(|pos| rows.iter().find(|c| c.pos == pos && c.eligible))
}

fn audit(obs: &Observation, weights: &[i16]) -> R<()> {
    let row = &obs.row;
    require(
        row.prefix.len() <= 192
            && row.context_start <= row.context_end
            && row.context_end <= row.prefix.len(),
        "bad fact bounds",
    )?;
    let mut positions = BTreeSet::new();
    let eligible: Vec<_> = obs.candidates.iter().filter(|c| c.eligible).collect();
    for c in &obs.candidates {
        require(
            positions.insert(c.pos)
                && (row.context_start..row.context_end).contains(&c.pos)
                && row.prefix[c.pos] == c.id,
            "candidate outside evidence or wrong original ID",
        )?;
        if c.eligible {
            let features = c
                .features
                .as_ref()
                .ok_or("missing active feature indices")?;
            require(
                features.len() == 16
                    && features
                        .iter()
                        .enumerate()
                        .all(|(level, &key)| key / 32 == level),
                "invalid typed feature IDs",
            )?;
            let score: i32 = features.iter().map(|&key| i32::from(weights[key])).sum();
            require(
                c.score == Some(score),
                "device score differs from saved native weights",
            )?;
            let rank = 1 + eligible
                .iter()
                .filter(|other| other.score.is_some_and(|s| s > score))
                .count();
            require(
                c.rank == Some(rank),
                "full-occurrence competition rank differs",
            )?;
        }
    }
    require(
        obs.no_read_score == i32::from(weights[512]),
        "NoRead word disagrees with state",
    )?;
    let maximum = eligible.iter().filter_map(|c| c.score).max();
    let maxima: Vec<_> = eligible.iter().filter(|c| c.score == maximum).collect();
    require(
        obs.best_score == maximum && obs.best_ties == maxima.len(),
        "maximum/tie count disagrees",
    )?;
    let winner = match maxima.as_slice() {
        [winner] if maximum.is_some_and(|s| s > obs.no_read_score) => Some(winner.pos),
        _ => None,
    };
    require(winner == obs.winner_pos, "winner or abstention disagrees")?;
    for (pos, id) in [(row.gold_pos, row.target), (row.rival_pos, row.rival)] {
        if let Some(pos) = pos {
            require(
                (row.context_start..row.context_end).contains(&pos) && row.prefix[pos] == id,
                "incorrect annotated position",
            )?;
        }
    }
    for name in ARMS {
        let choice = obs
            .arms
            .get(name)
            .ok_or_else(|| format!("missing arm {name}"))?;
        if let Some(pos) = choice.source_pos {
            require(
                (row.context_start..row.context_end).contains(&pos)
                    && row.prefix[pos] == choice.gate_id,
                "writer did not receive exact original ID from indicated source",
            )?;
        }
    }
    let baseline = &obs.arms["marker"];
    let absent = &obs.arms["absent"];
    require(
        absent.source_pos.is_none()
            && baseline.source_pos.is_none()
            && absent.gate_id == baseline.gate_id
            && absent.emitted_id == baseline.emitted_id,
        "missing source did not use normal marker fallback",
    )?;
    let learned = &obs.arms["learned"];
    require(
        learned.source_pos == winner,
        "reported learned source changed on way to writer",
    )?;
    if winner.is_none() {
        require(
            learned.gate_id == baseline.gate_id && learned.emitted_id == baseline.emitted_id,
            "learned abstention failed normal fallback",
        )?;
    }
    Ok(())
}

fn count(summary: &mut Stratum, obs: &Observation, flags: Flags) {
    let row = &obs.row;
    let gold = candidate(&obs.candidates, row.gold_pos).and_then(|c| c.score);
    let rival = candidate(&obs.candidates, row.rival_pos).and_then(|c| c.score);
    let has_gold = row.gold_pos.is_some();
    summary.rows += 1;
    summary.unique_gold += usize::from(has_gold);
    summary.absent_gold += usize::from(row.source_status == "absent");
    summary.gold_eligible += usize::from(gold.is_some());
    summary.factual_positions_scored += obs.candidates.iter().filter(|c| c.eligible).count();
    summary.learned_no_read += usize::from(obs.winner_pos.is_none());
    summary.gold_above_no_read += usize::from(gold.is_some_and(|s| s > obs.no_read_score));
    if let (Some(g), Some(r)) = (gold, rival) {
        summary.two_candidates_eligible += 1;
        summary.gold_gt_rival += usize::from(g > r);
        summary.gold_eq_rival += usize::from(g == r);
        summary.gold_lt_rival += usize::from(g < r);
        summary.higher_other_than_gold += usize::from(obs.candidates.iter().any(|c| {
            Some(c.pos) != row.gold_pos
                && Some(c.pos) != row.rival_pos
                && c.eligible
                && c.score.is_some_and(|s| s > g)
        }));
    }
    summary.learned_source_alias += usize::from(flags.source && !flags.first);
    summary.correct_source_given_oracle_emits += usize::from(flags.source && flags.oracle);
    summary.oracle_first_correct_on_unique += usize::from(has_gold && flags.oracle);
    summary.learned_first_correct_given_oracle_emits +=
        usize::from(has_gold && flags.oracle && flags.first);
    for name in ARMS {
        let x = &obs.arms[name];
        let m = summary.arms.entry(name.into()).or_default();
        let exact_source = has_gold && x.source_pos == row.gold_pos;
        let comparable = !matches!(row.cohort.as_str(), "subject_unbound" | "value_omitted");
        m.rows += 1;
        m.comparable_first_ids += usize::from(comparable);
        m.no_answer_emitted_counterfactual_id +=
            usize::from(!comparable && x.emitted_id == row.raw_target);
        m.reads += usize::from(x.source_pos.is_some());
        m.correct_source += usize::from(exact_source);
        m.gold_selected_id += usize::from(comparable && x.gate_id == row.raw_target);
        m.correct_first_id += usize::from(comparable && x.emitted_id == row.raw_target);
        m.source_and_emission += usize::from(exact_source && x.emitted_id == row.raw_target);
        m.emitted_space += usize::from(x.emitted_id == SPACE);
        m.emitted_eos += usize::from(x.emitted_id == 1);
        m.writer_changed_gate += usize::from(x.gate_id != x.emitted_id);
        if let Some(lexical) = &x.lexical {
            m.continue_to_lexical += 1;
            m.subsequent_lexical_correct += usize::from(
                lexical.get("emitted_id").and_then(Value::as_u64) == Some(u64::from(row.target)),
            );
        }
    }
}

fn group_summary(groups: &BTreeMap<String, BTreeMap<(usize, usize), Flags>>) -> R<Value> {
    let mut out = BTreeMap::new();
    type Flag = fn(&Flags) -> bool;
    let endpoints: [(&str, Flag); 6] = [
        ("gold_strictly_above_rival", |f| f.pref),
        ("correct_all_position_source", |f| f.source),
        ("learned_first_id", |f| f.first),
        ("learned_correct_source_and_first_id", |f| f.joint),
        ("oracle_first_id", |f| f.oracle),
        ("marker_first_id", |f| f.marker),
    ];
    for (label, endpoint) in endpoints {
        let mut pairs = 0;
        let mut whole = 0;
        for cells in groups.values() {
            require(
                cells.len() == 4 && (0..2).all(|a| (0..2).all(|s| cells.contains_key(&(a, s)))),
                "incomplete crossed group",
            )?;
            for a in 0..2 {
                pairs += usize::from(endpoint(&cells[&(a, 0)]) && endpoint(&cells[&(a, 1)]));
            }
            whole += usize::from(cells.values().all(endpoint));
        }
        out.insert(
            label,
            json!({"both_questions_per_assignment":pairs,"both_assignments_per_group":whole}),
        );
    }
    Ok(json!({"groups":groups.len(),"assignments":groups.len()*2,"endpoints":out}))
}

fn csv(out: &mut impl Write, fields: impl IntoIterator<Item = String>) -> R<()> {
    for (index, v) in fields.into_iter().enumerate() {
        if index != 0 {
            write!(out, ",")?;
        }
        write!(out, "\"{}\"", v.replace('"', "\"\""))?;
    }
    writeln!(out)?;
    Ok(())
}

fn opt<T: ToString>(x: Option<T>) -> String {
    x.map(|v| v.to_string()).unwrap_or_default()
}

fn main() -> R<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [panel, observations, words, output] = args.as_slice() else {
        return Err(
            "usage: audit-shared-source FULL_PANEL.jsonl OBSERVATIONS.jsonl WEIGHTS.i16le NEW_DIR"
                .into(),
        );
    };
    let raw = fs::read(words)?;
    require(raw.len() == 1026, "expected 513 little-endian i16 words")?;
    let weights = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect::<Vec<_>>();
    require(
        weights.iter().all(|x| (-4096..=4096).contains(x)),
        "weight outside registered clip",
    )?;
    let panels = json_rows::<Row>(Path::new(panel))?;
    let traces = json_rows::<Observation>(Path::new(observations))?;
    require(
        !panels.is_empty() && panels.len() == traces.len(),
        "panel and observations differ in length",
    )?;
    let out_dir = Path::new(output);
    fs::create_dir(out_dir)?;
    let mut rows = BufWriter::new(File::create_new(out_dir.join("opening-scores.csv"))?);
    let mut header = [
        "cohort",
        "group",
        "pair",
        "assignment",
        "question_side",
        "source_status",
        "gold_id",
        "rival_id",
        "raw_gold",
        "raw_rival",
        "gold_position",
        "rival_position",
        "gold_score",
        "rival_score",
        "gold_rank_all",
        "rival_rank_all",
        "NoRead",
        "best_score",
        "best_ties",
        "learned_position",
        "eligible_candidates",
    ]
    .map(String::from)
    .to_vec();
    for name in ARMS {
        for col in ["source", "gate_id", "emitted_id", "next_lexical_emitted"] {
            header.push(format!("{name}_{col}"));
        }
    }
    csv(&mut rows, header)?;
    let mut unique = BTreeSet::new();
    let mut by_cohort = BTreeMap::<String, Stratum>::new();
    let mut groups = BTreeMap::<String, BTreeMap<(usize, usize), Flags>>::new();
    for (panel, obs) in panels.iter().zip(&traces) {
        let r = &obs.row;
        require(
            panel == r,
            "panel content or digest alignment differs from observed row",
        )?;
        require(
            unique.insert((&r.cohort, &r.group_id, r.assignment, r.side, &r.pair_id)),
            "duplicate observation",
        )?;
        audit(obs, &weights).map_err(|e| format!("{} / {}: {e}", r.pair_id, r.side))?;
        let g = candidate(&obs.candidates, r.gold_pos);
        let v = candidate(&obs.candidates, r.rival_pos);
        let source = r.gold_pos.is_some() && obs.winner_pos == r.gold_pos;
        let first = obs.arms["learned"].emitted_id == r.raw_target;
        let pref = g
            .and_then(|x| x.score)
            .zip(v.and_then(|x| x.score))
            .is_some_and(|(a, b)| a > b);
        let flags = Flags {
            source,
            first,
            pref,
            joint: source && first,
            oracle: obs.arms["correct"].emitted_id == r.raw_target,
            marker: obs.arms["marker"].emitted_id == r.raw_target,
        };
        count(by_cohort.entry(r.cohort.clone()).or_default(), obs, flags);
        if r.cohort == "controlled" {
            require(
                groups
                    .entry(r.group_id.clone())
                    .or_default()
                    .insert((r.assignment, r.side), flags)
                    .is_none(),
                "duplicate controlled cell",
            )?;
        }
        let mut record = vec![
            r.cohort.clone(),
            r.group_id.clone(),
            r.pair_id.clone(),
            r.assignment.to_string(),
            r.side.to_string(),
            r.source_status.clone(),
            r.target.to_string(),
            r.rival.to_string(),
            r.raw_target.to_string(),
            r.raw_rival.to_string(),
            opt(r.gold_pos),
            opt(r.rival_pos),
            opt(g.and_then(|x| x.score)),
            opt(v.and_then(|x| x.score)),
            opt(g.and_then(|x| x.rank)),
            opt(v.and_then(|x| x.rank)),
            obs.no_read_score.to_string(),
            opt(obs.best_score),
            obs.best_ties.to_string(),
            opt(obs.winner_pos),
            obs.candidates
                .iter()
                .filter(|c| c.eligible)
                .count()
                .to_string(),
        ];
        for name in ARMS {
            let arm = &obs.arms[name];
            record.extend([
                opt(arm.source_pos),
                arm.gate_id.to_string(),
                arm.emitted_id.to_string(),
                opt(arm
                    .lexical
                    .as_ref()
                    .and_then(|v| v.get("emitted_id"))
                    .and_then(Value::as_u64)),
            ]);
        }
        csv(&mut rows, record)?;
    }
    rows.flush()?;
    let stats = json!({"schema":"uor-r4.shared-occurrence-analysis/1","distinct_openings":panels.len(),"cohorts":by_cohort,"crossed":group_summary(&groups)?});
    let mut out = BufWriter::new(File::create_new(out_dir.join("summary.json"))?);
    serde_json::to_writer_pretty(&mut out, &stats)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}
