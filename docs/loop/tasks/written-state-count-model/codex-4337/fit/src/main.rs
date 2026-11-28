#![forbid(unsafe_code)]

mod device;
mod solve;
mod wire;

use serde::{Deserialize, Serialize};
use solve::{Comparison, Tools, Verdict};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Write,
    path::Path,
};
use wire::{
    Arm, BOUND, MARGIN, Result, Row, Span, WORDS, Words, gold_index, invalid, score, winner,
};

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct CaseReport {
    case_id: String,
    group: String,
    bounds: wire::Bounds,
    choices: usize,
    gold: Option<Span>,
    gold_present: bool,
    selected: Option<Span>,
    selected_score: i32,
    no_read_score: i32,
    minimum_gold_margin: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    strongest_gold_rival: Option<ScoredChoice>,
    swapped_gold_present: Option<bool>,
    swapped_selected: Option<Span>,
    swapped_differs: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    both_matched_gold_selected: Option<bool>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct ScoredChoice {
    span: Option<Span>,
    score: i32,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct Replay {
    exact: usize,
    positive: usize,
    positive_selected: usize,
    no_read: usize,
    false_reads: usize,
    paired: usize,
    paired_changes: usize,
    strictly_positive: usize,
    minimum_margin: Option<i32>,
    cases: Vec<CaseReport>,
}

struct Panel {
    rows: Vec<Row>,
    mates: Vec<Option<usize>>,
}

impl Panel {
    fn read(features: &Path, gold: &Path) -> Result<Self> {
        Self::from_rows(wire::read_rows(features, Some(gold))?)
    }

    fn unlabelled(features: &Path) -> Result<Self> {
        Self::from_rows(wire::read_rows(features, None)?)
    }

    fn from_rows(rows: Vec<Row>) -> Result<Self> {
        let index: BTreeMap<_, _> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.case_id.as_str(), i))
            .collect();
        if index.len() != rows.len() {
            return Err(invalid("duplicate gold case"));
        }
        let mut mates = Vec::new();
        for row in &rows {
            let mate = row
                .swapped_case_id
                .as_deref()
                .map(|id| {
                    let i = *index
                        .get(id)
                        .ok_or_else(|| invalid(format!("missing swap {id}")))?;
                    let other = &rows[i];
                    if other.swapped_case_id.as_deref() != Some(&row.case_id)
                        || other.group != row.group
                        || other.gold == row.gold
                    {
                        return Err(invalid(format!(
                            "{}: invalid registered mate {id}",
                            row.case_id
                        )));
                    }
                    Ok(i)
                })
                .transpose()?;
            mates.push(mate);
        }
        Ok(Self { rows, mates })
    }

    fn comparisons(&self, arm: Arm) -> Vec<Comparison> {
        let mut output = Vec::new();
        for row in &self.rows {
            let Some(gold) = gold_index(row) else {
                continue;
            };
            let candidates = &row.keys[&arm];
            for (rival, keys) in candidates.iter().enumerate() {
                if rival != gold {
                    output.push(Comparison::new(
                        &row.case_id,
                        (gold, row.gold, &candidates[gold]),
                        (rival, row.choices[rival], keys),
                    ));
                }
            }
        }
        output
    }

    fn replay(&self, arm: Arm, words: &Words, record_rivals: bool) -> Replay {
        let mut replay = Replay {
            exact: 0,
            positive: 0,
            positive_selected: 0,
            no_read: 0,
            false_reads: 0,
            paired: 0,
            paired_changes: 0,
            strictly_positive: 0,
            minimum_margin: None,
            cases: Vec::new(),
        };
        for (row, mate) in self.rows.iter().zip(&self.mates) {
            let keys = &row.keys[&arm];
            let (index, selected_score) = winner(words, keys);
            let selected = row.choices[index];
            let gold = gold_index(row);
            let rival = gold.and_then(|gold| {
                keys.iter()
                    .enumerate()
                    .filter(|(index, _)| *index != gold)
                    .map(|(index, keys)| (index, score(words, keys)))
                    .max_by_key(|&(index, value)| (value, std::cmp::Reverse(index)))
            });
            let margin = gold
                .zip(rival)
                .map(|(gold, (_, rival_score))| score(words, &keys[gold]) - rival_score);
            let swapped_selected = mate.map(|mate| {
                let swapped = &self.rows[mate];
                let (index, _) = winner(words, &swapped.keys[&arm]);
                swapped.choices[index]
            });
            let swapped_differs = swapped_selected.map(|swapped| selected != swapped);
            let exact = selected == row.gold;
            replay.exact += usize::from(exact);
            if row.gold.is_some() {
                replay.positive += 1;
                replay.positive_selected += usize::from(exact);
            } else {
                replay.no_read += 1;
                replay.false_reads += usize::from(selected.is_some());
            }
            replay.paired += usize::from(mate.is_some());
            replay.paired_changes += usize::from(swapped_differs == Some(true));
            if let Some(margin) = margin {
                replay.strictly_positive += usize::from(margin >= MARGIN);
                replay.minimum_margin =
                    Some(replay.minimum_margin.map_or(margin, |old| old.min(margin)));
            }
            replay.cases.push(CaseReport {
                case_id: row.case_id.clone(),
                group: row.group.clone(),
                bounds: row.bounds,
                choices: row.choices.len(),
                gold: row.gold,
                gold_present: gold.is_some(),
                selected,
                selected_score,
                no_read_score: score(words, &keys[0]),
                minimum_gold_margin: margin,
                strongest_gold_rival: if record_rivals {
                    rival.map(|(index, score)| ScoredChoice {
                        span: row.choices[index],
                        score,
                    })
                } else {
                    None
                },
                swapped_gold_present: mate
                    .map(|mate| row.choices.binary_search(&self.rows[mate].gold).is_ok()),
                swapped_selected: swapped_selected.flatten(),
                swapped_differs,
                both_matched_gold_selected: if record_rivals {
                    mate.map(|mate| exact && swapped_selected == Some(self.rows[mate].gold))
                } else {
                    None
                },
            });
        }
        replay
    }
}

#[derive(Deserialize, Serialize)]
struct GateReport {
    arm: Arm,
    cases: usize,
    comparisons: usize,
    authored_pair_count: usize,
    registered_groups: Vec<String>,
    missing_targets: Vec<String>,
    missing_swap_targets: Vec<String>,
    bound: i16,
    margin: i32,
    payload_bytes: usize,
    witness_nonzero_words: Option<usize>,
    witness_maximum_abs_word: Option<u16>,
    zero: Replay,
    witness: Option<Replay>,
    verdict: Verdict,
}

fn coverage(panel: &Panel) -> Result<()> {
    let zero = panel.replay(Arm::Geometry, &[0; WORDS], false);
    serde_json::to_writer(std::io::stdout().lock(), &zero)?;
    println!();
    Ok(())
}

fn gate_report(
    panel: &Panel,
    arm: Arm,
    comparisons: &[Comparison],
    verdict: Verdict,
) -> Result<GateReport> {
    solve::verify(&verdict, comparisons)?;
    let zero = panel.replay(arm, &[0; WORDS], false);
    let missing_targets: Vec<_> = zero
        .cases
        .iter()
        .filter(|case| !case.gold_present)
        .map(|case| case.case_id.clone())
        .collect();
    let registered_groups: BTreeSet<_> = zero.cases.iter().map(|case| case.group.clone()).collect();
    let missing_swap_targets: Vec<_> = zero
        .cases
        .iter()
        .filter(|row| row.swapped_gold_present == Some(false))
        .map(|row| row.case_id.clone())
        .collect();
    let (witness, witness_nonzero_words, witness_maximum_abs_word) = match &verdict {
        Verdict::Feasible { words } => {
            if !missing_targets.is_empty() || !missing_swap_targets.is_empty() || zero.paired == 0 {
                return Err(invalid(
                    "witness omitted candidate coverage or matched contrasts",
                ));
            }
            let words = <&Words>::try_from(words.as_slice())?;
            let replay = panel.replay(arm, words, false);
            if replay.exact != panel.rows.len()
                || replay.strictly_positive != panel.rows.len()
                || replay.paired_changes != replay.paired
                || replay.minimum_margin != solve::minimum(words, comparisons)
            {
                return Err(invalid(
                    "direct candidate scoring or matched counterfactual failed",
                ));
            }
            (
                Some(replay),
                Some(words.iter().filter(|&&word| word != 0).count()),
                words.iter().map(|word| word.unsigned_abs()).max(),
            )
        }
        _ => (None, None, None),
    };
    Ok(GateReport {
        arm,
        cases: panel.rows.len(),
        comparisons: comparisons.len(),
        authored_pair_count: zero.paired / 2,
        registered_groups: registered_groups.into_iter().collect(),
        missing_targets,
        missing_swap_targets,
        bound: BOUND,
        margin: MARGIN,
        payload_bytes: 2 * WORDS,
        witness_nonzero_words,
        witness_maximum_abs_word,
        zero,
        witness,
        verdict,
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

fn gate(panel: &Panel, output: &Path, z3: &Path, highs: &Path) -> Result<()> {
    fs::create_dir(output)?;
    for (arm, name) in [(Arm::Geometry, "geometry"), (Arm::Count, "count")] {
        let scratch = output.join(name);
        fs::create_dir(&scratch)?;
        let scratch = scratch.canonicalize()?;
        let tools = Tools {
            z3,
            highs,
            scratch: &scratch,
        };
        let comparisons = panel.comparisons(arm);
        let missing = panel.rows.iter().zip(&panel.mates).any(|(row, mate)| {
            gold_index(row).is_none()
                || mate.is_some_and(|i| row.choices.binary_search(&panel.rows[i].gold).is_err())
        });
        let verdict = if missing {
            Verdict::Unknown {
                reason: "the label-blind candidate tape omitted one or more targets".into(),
            }
        } else {
            solve::solve(&tools, &comparisons)?
        };
        let report = gate_report(panel, arm, &comparisons, verdict)?;
        let path = output.join(format!("{name}.json"));
        write_new(&path, &serde_json::to_vec_pretty(&report)?)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn verify(panel: &Panel, input: &Path) -> Result<()> {
    for (arm, name) in [(Arm::Geometry, "geometry"), (Arm::Count, "count")] {
        let report: GateReport =
            serde_json::from_slice(&fs::read(input.join(format!("{name}.json")))?)?;
        let comparisons = panel.comparisons(arm);
        let replay = gate_report(panel, arm, &comparisons, report.verdict)?;
        if replay.arm != report.arm
            || replay.cases != report.cases
            || replay.comparisons != report.comparisons
            || replay.authored_pair_count != report.authored_pair_count
            || replay.registered_groups != report.registered_groups
            || replay.missing_targets != report.missing_targets
            || replay.missing_swap_targets != report.missing_swap_targets
            || replay.bound != report.bound
            || replay.margin != report.margin
            || replay.payload_bytes != report.payload_bytes
            || replay.witness_nonzero_words != report.witness_nonzero_words
            || replay.witness_maximum_abs_word != report.witness_maximum_abs_word
            || replay.zero != report.zero
            || replay.witness != report.witness
        {
            return Err(invalid(format!("changed replay for {name}")));
        }
        eprintln!(
            "verified {name}: {} original comparisons",
            replay.comparisons
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let raw: Vec<_> = env::args_os().skip(1).collect();
    let args: Vec<_> = raw.iter().map(Path::new).collect();
    match args.as_slice() {
        [command, features, gold, output, z3, highs] if *command == Path::new("gate") => gate(
            &Panel::read(features, gold)?,
            output,
            &z3.canonicalize()?,
            &highs.canonicalize()?,
        ),
        [command, features, gold, input] if *command == Path::new("verify") => {
            verify(&Panel::read(features, gold)?, input)
        }
        [command, features, gold] if *command == Path::new("coverage") => {
            coverage(&Panel::read(features, gold)?)
        }
        [command, features, gold, exclusions, arm, output]
            if *command == Path::new("prepare-train") =>
        {
            let panel = device::eligible_train(Panel::read(features, gold)?, exclusions)?;
            device::prepare(&panel, device::arm_name(arm)?, output, true)
        }
        [command, features, arm, output] if *command == Path::new("prepare-eval") => {
            device::prepare(
                &Panel::unlabelled(features)?,
                device::arm_name(arm)?,
                output,
                false,
            )
        }
        [command, output] if *command == Path::new("zero-device") => device::zero_payload(output),
        [command, features, cold, arm, weights, output]
            if *command == Path::new("verify-device") =>
        {
            device::verify(
                &Panel::unlabelled(features)?,
                &Panel::unlabelled(cold)?,
                device::arm_name(arm)?,
                weights,
                output,
            )
        }
        [command, features, gold, arm, weights] if *command == Path::new("replay-device") => {
            let panel = Panel::read(features, gold)?;
            let replay = panel.replay(device::arm_name(arm)?, &device::read_words(weights)?, true);
            serde_json::to_writer(std::io::stdout().lock(), &replay)?;
            println!();
            Ok(())
        }
        _ => Err(invalid(
            "expected gate, verify, coverage, prepare-train, prepare-eval, zero-device, verify-device or replay-device",
        )),
    }
}
