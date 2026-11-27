use super::*;
use serde_json::{json, Value};

const ARMS: [&str; 5] = ["native", "learned", "swapped", "correct", "opposite"];
const CAP: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Action {
    Copy,
    Eos,
    Native,
}

fn action(score: i32) -> Action {
    match score.cmp(&0) {
        std::cmp::Ordering::Greater => Action::Copy,
        std::cmp::Ordering::Less => Action::Eos,
        std::cmp::Ordering::Equal => Action::Native,
    }
}

#[derive(Debug, Deserialize)]
struct Common {
    stage: u8,
    own_score: i32,
    peer_score: i32,
    own_choice: Action,
    peer_choice: Action,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
struct Step {
    selected_source: String,
    stage: Option<u8>,
    action_score: Option<i32>,
    requested_pos: Option<u32>,
    requested_id: Option<u32>,
    refusal: u32,
    native_proposal: u32,
    committed_id: u32,
    copied: bool,
    source_id: Option<u32>,
    source_admitted: bool,
    read_pos: Option<u32>,
    claim_pos: Option<u32>,
    #[serde(flatten)]
    writer: BTreeMap<String, Value>,
}

impl Step {
    fn same_writer(&self, other: &Self) -> bool {
        self.native_proposal == other.native_proposal
            && self.source_id == other.source_id
            && self.source_admitted == other.source_admitted
            && self.read_pos == other.read_pos
            && self.claim_pos == other.claim_pos
            && self.writer == other.writer
    }
}

#[derive(Debug, Deserialize)]
struct Arm {
    steps: Vec<Step>,
    terminal: bool,
    first_mismatch: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Observation {
    #[serde(flatten)]
    row: Row,
    common: Common,
    arms: BTreeMap<String, Arm>,
}

#[derive(Default, Serialize)]
struct ArmSummary {
    exact_rows: usize,
    exact_pairs: usize,
    native_eos: usize,
    selected_eos: usize,
    open_at_cap: usize,
    verified_copies: usize,
    copy_refusals: usize,
    first_mismatch: BTreeMap<usize, usize>,
}

#[derive(Default, Serialize)]
struct Summary {
    pairs: usize,
    directed_common_flips: usize,
    reverse_common_flips: usize,
    common_outcomes: BTreeMap<String, usize>,
    arms: BTreeMap<String, ArmSummary>,
}

fn reference(ids: [u32; 4], contrast: u8, side: u8) -> Vec<u32> {
    let mut out = vec![ids[0], PERIOD];
    out.extend_from_slice(&ids[1..1 + usize::from(contrast + side)]);
    out.push(EOS);
    out
}

fn check_row(row: &Row) -> Result<()> {
    require(
        row.side <= 1
            && row.contrast <= 1
            && row.expected == reference(row.ids, row.contrast, row.side)
            && row.prefix.len() + CAP < LIMIT
            && row.ids.into_iter().collect::<BTreeSet<_>>().len() == 4,
        "illegally shaped sealed row",
    )?;
    let delimiter = row
        .prefix
        .windows(2)
        .position(|ids| ids == [NEWLINE, NEWLINE])
        .ok_or("missing factual block delimiter")?;
    for (id, pos) in row.ids.into_iter().zip(row.positions) {
        require(
            (pos as usize) < delimiter
                && row.prefix.get(pos as usize) == Some(&id)
                && row
                    .prefix
                    .iter()
                    .filter(|&&candidate| candidate == id)
                    .count()
                    == 1,
            "sealed source occurrence drift",
        )?;
    }
    Ok(())
}

fn check_pair(left: &Observation, right: &Observation) -> Result<()> {
    let a = &left.row;
    let b = &right.row;
    let query = a
        .prefix
        .iter()
        .position(|&id| id == QUESTION)
        .ok_or("missing query")?;
    require(
        a.side == 0
            && b.side == 1
            && a.pair_id == b.pair_id
            && a.family == b.family
            && a.contrast == b.contrast
            && a.ids == b.ids
            && a.positions == b.positions
            && b.prefix.get(..=query) == Some(&a.prefix[..=query])
            && a.prefix != b.prefix
            && left.common.stage == a.contrast
            && right.common.stage == a.contrast
            && left.common.own_score == right.common.peer_score
            && right.common.own_score == left.common.peer_score,
        "paired row or common-condition swap drift",
    )?;
    for obs in [left, right] {
        require(
            obs.common.own_choice == action(obs.common.own_score)
                && obs.common.peer_choice == action(obs.common.peer_score),
            "common action does not follow signed score",
        )?;
    }
    Ok(())
}

fn first_mismatch(expected: &[u32], actual: &[u32]) -> Option<usize> {
    expected
        .iter()
        .zip(actual)
        .position(|(want, got)| want != got)
        .or_else(|| (expected.len() != actual.len()).then(|| expected.len().min(actual.len())))
        .map(|index| index + 1)
}

fn check_stop(boundary: &Arm, expected: &[u32]) -> Result<(Vec<u32>, Option<usize>)> {
    let actual: Vec<_> = boundary
        .steps
        .iter()
        .map(|step| step.committed_id)
        .collect();
    let observed_eos = actual.iter().position(|&id| id == EOS);
    require(
        !actual.is_empty()
            && actual.len() <= CAP
            && observed_eos.map_or(actual.len() == CAP, |i| i + 1 == actual.len())
            && boundary.terminal == observed_eos.is_some(),
        "terminal/cap or token after EOS violation",
    )?;
    let mismatch = first_mismatch(expected, &actual);
    require(
        mismatch == boundary.first_mismatch,
        "inconsistent first wrong committed token",
    )?;
    Ok((actual, mismatch))
}

fn oracle(branch: &str, row: &Row, stage: u8) -> Action {
    let side = if branch == "correct" {
        row.side
    } else {
        1 - row.side
    };
    if stage < row.contrast + side {
        Action::Copy
    } else {
        Action::Eos
    }
}

fn check_arm(
    name: &str,
    obs: &Observation,
    arm: &Arm,
    writers: &mut BTreeMap<Vec<u32>, Step>,
) -> Result<(Vec<u32>, Option<usize>)> {
    let row = &obs.row;
    let (actual, mismatch) = check_stop(arm, &row.expected)?;
    let mut release_stage = None;
    for (i, step) in arm.steps.iter().enumerate() {
        let prefix = actual[..i].to_vec();
        if let Some(other) = writers.get(&prefix) {
            require(
                step.same_writer(other),
                "writer changed at identical committed prefix",
            )?;
        } else {
            writers.insert(prefix, step.clone());
        }

        let request = step.requested_pos.zip(step.requested_id);
        require(
            step.requested_pos.is_some() == step.requested_id.is_some()
                && step.stage == release_stage,
            "unpaired source request or wrong stage",
        )?;
        let chosen = match (name, release_stage) {
            ("learned" | "swapped", Some(stage)) => {
                let score = step.action_score.ok_or("missing active action score")?;
                if stage == row.contrast && i == 2 + usize::from(stage) {
                    let expected = if name == "learned" {
                        obs.common.own_score
                    } else {
                        obs.common.peer_score
                    };
                    require(
                        score == expected,
                        "self-fed common-prefix action differs from conditional score",
                    )?;
                }
                Some(action(score))
            }
            ("correct" | "opposite", Some(stage)) => {
                require(
                    step.action_score.is_none(),
                    "oracle exposes learned action score",
                )?;
                Some(oracle(name, row, stage))
            }
            _ => {
                require(
                    step.action_score.is_none(),
                    "inactive action exposes learned score",
                )?;
                None
            }
        };
        if i > 0 {
            require(
                !step.copied && step.claim_pos.is_none(),
                "unexpected writer copy",
            )?;
        }
        match step.selected_source.as_str() {
            "copy" => {
                require(
                    i == 0
                        && step.copied
                        && step.source_admitted
                        && step.source_id == Some(row.ids[0])
                        && step.claim_pos == Some(row.positions[0])
                        && step.read_pos == Some(row.positions[0])
                        && step.committed_id == row.ids[0]
                        && request.is_none()
                        && step.refusal == 0,
                    "opening copy lacks authentication",
                )?;
            }
            "supplied_period" => {
                require(
                    i == 1
                        && !step.copied
                        && step.committed_id == PERIOD
                        && request.is_none()
                        && step.refusal == 0,
                    "unexpected period intervention",
                )?;
                release_stage = (name != "native").then_some(0);
            }
            "verified_copy" => {
                let stage = release_stage.ok_or("verified copy before release stage")?;
                let index = usize::from(stage) + 1;
                require(
                    chosen == Some(Action::Copy)
                        && index < row.ids.len()
                        && request == Some((row.positions[index], row.ids[index]))
                        && step.committed_id == row.ids[index]
                        && step.refusal == 0,
                    "selected token is not the verified original source",
                )?;
                release_stage = (stage < 2).then_some(stage + 1);
            }
            "selected_eos" => {
                require(
                    chosen == Some(Action::Eos)
                        && step.committed_id == EOS
                        && request.is_none()
                        && step.refusal == 0,
                    "explicit EOS lacks a negative or oracle selection",
                )?;
            }
            "native" => {
                let valid_fallback = if chosen == Some(Action::Copy) {
                    let index = usize::from(release_stage.ok_or("missing copy stage")?) + 1;
                    index < row.ids.len()
                        && request == Some((row.positions[index], row.ids[index]))
                        && step.refusal == 3
                } else {
                    chosen != Some(Action::Eos) && request.is_none() && step.refusal == 0
                };
                require(
                    step.committed_id == step.native_proposal && valid_fallback,
                    "native output or refusal differs from the fresh proposal",
                )?;
                release_stage = None;
            }
            _ => return Err("unknown selected source".into()),
        }
        if i < 2 {
            let expected_source = if i == 0 { "copy" } else { "supplied_period" };
            require(
                step.selected_source == expected_source,
                "intervention arm lost supplied prefix",
            )?;
        }
    }
    Ok((actual, mismatch))
}

fn summarize(
    summary: &mut Summary,
    sides: [&Observation; 2],
    exact: &[BTreeMap<String, bool>; 2],
) -> Result<()> {
    summary.pairs += 1;
    let choices = (&sides[0].common.own_choice, &sides[1].common.own_choice);
    summary.directed_common_flips += usize::from(choices == (&Action::Eos, &Action::Copy));
    summary.reverse_common_flips += usize::from(choices == (&Action::Copy, &Action::Eos));
    *summary
        .common_outcomes
        .entry(format!("{:?}/{:?}", choices.0, choices.1))
        .or_default() += 1;
    for name in ARMS {
        let counts = summary.arms.entry(name.to_owned()).or_default();
        counts.exact_pairs += usize::from(exact[0][name] && exact[1][name]);
        for (idx, obs) in sides.into_iter().enumerate() {
            let arm = obs.arms.get(name).ok_or("missing observed arm")?;
            counts.exact_rows += usize::from(exact[idx][name]);
            counts.open_at_cap += usize::from(!arm.terminal);
            if let Some(step) = arm.first_mismatch {
                *counts.first_mismatch.entry(step).or_default() += 1;
            }
            for step in &arm.steps {
                counts.verified_copies += usize::from(step.selected_source == "verified_copy");
                counts.copy_refusals += usize::from(step.refusal != 0);
                if step.committed_id == EOS {
                    match step.selected_source.as_str() {
                        "selected_eos" => counts.selected_eos += 1,
                        "native" => counts.native_eos += 1,
                        _ => return Err("unaccounted EOS provenance".into()),
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn run(
    vocab: &ByteBpeTokenizer,
    panel: &Path,
    split: &str,
    observations: &Path,
    out: &Path,
) -> Result<()> {
    require(
        ["train", "finite", "held-primary", "held-secondary"].contains(&split),
        "unknown split",
    )?;
    verify_manifest(panel, &panel.join("SHA256SUMS"))?;
    let dir = panel.join(split);
    verify_manifest(&dir, &dir.join("SHA256SUMS"))?;
    let mut sealed = BTreeMap::new();
    for row in read_jsonl::<Row>(&dir.join("rows.jsonl"))? {
        check_row(&row)?;
        require(
            sealed
                .insert((row.pair_id.clone(), row.side), row)
                .is_none(),
            "duplicate sealed row",
        )?;
    }
    let mut observed = BTreeMap::<String, BTreeMap<u8, Observation>>::new();
    for obs in read_jsonl::<Observation>(observations)? {
        let id = obs.row.pair_id.clone();
        let side = obs.row.side;
        require(
            sealed.remove(&(id.clone(), side)) == Some(obs.row.clone()),
            "observation substituted a sealed row",
        )?;
        require(
            observed.entry(id).or_default().insert(side, obs).is_none(),
            "duplicate observation",
        )?;
    }
    require(
        sealed.is_empty() && !observed.is_empty(),
        "missing observed rows",
    )?;

    let mut summaries = BTreeMap::<String, Summary>::new();
    let mut details = Vec::new();
    for (id, pair) in observed {
        let sides = [
            pair.get(&0).ok_or("missing stop side")?,
            pair.get(&1).ok_or("missing continue side")?,
        ];
        check_pair(sides[0], sides[1])?;
        let mut pair_exact = [BTreeMap::new(), BTreeMap::new()];
        for (side, obs) in sides.into_iter().enumerate() {
            require(obs.arms.len() == ARMS.len(), "wrong arm count")?;
            let mut outputs = BTreeMap::new();
            let mut writers = BTreeMap::new();
            for name in ARMS {
                let arm = obs.arms.get(name).ok_or("missing observed arm")?;
                let (ids, mismatch) = check_arm(name, obs, arm, &mut writers)
                    .map_err(|err| format!("{id} side {side} arm {name}: {err}"))?;
                pair_exact[side].insert(name.to_owned(), mismatch.is_none());
                outputs.insert(
                    name,
                    json!({
                        "ids": ids, "text": vocab.decode(&ids), "terminal": arm.terminal,
                        "first_mismatch": mismatch,
                    }),
                );
            }
            details.push(json!({
                "pair_id": id, "side": side, "family": obs.row.family, "contrast": obs.row.contrast,
                "reference_ids": obs.row.expected, "reference_text": vocab.decode(&obs.row.expected),
                "common_stage": obs.common.stage, "common_own_score": obs.common.own_score,
                "common_peer_score": obs.common.peer_score, "arms": outputs,
            }));
        }
        let contrast = sides[0].row.contrast;
        let family = &sides[0].row.family;
        for group in [
            "all".to_owned(),
            format!("contrast-{contrast}"),
            format!("family-{family}"),
        ] {
            summarize(summaries.entry(group).or_default(), sides, &pair_exact)?;
        }
    }

    fs::create_dir(out)?;
    let mut report = fresh_writer(&out.join("report.json"))?;
    serde_json::to_writer_pretty(
        &mut report,
        &json!({
            "schema": "codex-4321.independent-score/1", "split": split,
            "observations_sha256": hash(observations)?, "panel_sha256": hash(&panel.join("SHA256SUMS"))?,
            "summary": summaries,
        }),
    )?;
    writeln!(report)?;
    report.flush()?;
    let mut rows = fresh_writer(&out.join("rollouts.jsonl"))?;
    for detail in details {
        serde_json::to_writer(&mut rows, &detail)?;
        writeln!(rows)?;
    }
    rows.flush()?;
    write_manifest(out, &["report.json", "rollouts.jsonl"])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(side: u8, contrast: u8) -> Observation {
        let ids = [2000, 2001, 2002, 2003];
        let score = if side == 0 { -2 } else { 2 };
        Observation {
            row: Row {
                pair_id: "fixture".to_owned(),
                side,
                contrast,
                family: "planted".to_owned(),
                positions: [1, 2, 3, 4],
                ids,
                prefix: vec![
                    0,
                    2000,
                    2001,
                    2002,
                    2003,
                    NEWLINE,
                    NEWLINE,
                    QUESTION,
                    77 + u32::from(side),
                ],
                expected: reference(ids, contrast, side),
            },
            common: Common {
                stage: contrast,
                own_score: score,
                peer_score: -score,
                own_choice: action(score),
                peer_choice: action(-score),
            },
            arms: BTreeMap::new(),
        }
    }

    fn step(source: &str, id: u32, native: u32, stage: Option<u8>) -> Step {
        Step {
            selected_source: source.to_owned(),
            stage,
            action_score: None,
            requested_pos: None,
            requested_id: None,
            refusal: 0,
            native_proposal: native,
            committed_id: id,
            copied: false,
            source_id: None,
            source_admitted: false,
            read_pos: None,
            claim_pos: None,
            writer: BTreeMap::new(),
        }
    }

    fn forced(row: &Row) -> Vec<Step> {
        let mut open = step("copy", row.ids[0], 900, None);
        open.copied = true;
        open.source_admitted = true;
        open.source_id = Some(row.ids[0]);
        open.read_pos = Some(row.positions[0]);
        open.claim_pos = Some(row.positions[0]);
        vec![open, step("supplied_period", PERIOD, 900, None)]
    }

    fn oracle_fixture(row: &Row, side: u8) -> Arm {
        let mut steps = forced(row);
        let requested = row.contrast + side;
        for stage in 0..requested {
            let index = usize::from(stage) + 1;
            let mut copy = step("verified_copy", row.ids[index], 900, Some(stage));
            copy.requested_pos = Some(row.positions[index]);
            copy.requested_id = Some(row.ids[index]);
            steps.push(copy);
        }
        steps.push(step("selected_eos", EOS, 900, Some(requested)));
        let actual: Vec<_> = steps.iter().map(|s| s.committed_id).collect();
        Arm {
            terminal: true,
            steps,
            first_mismatch: first_mismatch(&row.expected, &actual),
        }
    }

    #[test]
    fn correct_and_actual_opposite_are_distinct_through_eos() -> Result<()> {
        for contrast in 0..=1 {
            for side in 0..=1 {
                let obs = example(side, contrast);
                check_row(&obs.row)?;
                let mut writers = BTreeMap::new();
                let correct = oracle_fixture(&obs.row, side);
                let opposite = oracle_fixture(&obs.row, 1 - side);
                assert_eq!(check_arm("correct", &obs, &correct, &mut writers)?.1, None);
                assert_eq!(
                    check_arm("opposite", &obs, &opposite, &mut writers)?.1,
                    Some(usize::from(contrast) + 3)
                );
                let mut invalid = oracle_fixture(&obs.row, side);
                invalid
                    .steps
                    .last_mut()
                    .ok_or("missing terminal step")?
                    .requested_id = Some(22);
                assert!(check_arm("correct", &obs, &invalid, &mut writers).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn conditional_direction_does_not_imply_self_fed_exact() -> Result<()> {
        let left = example(0, 1);
        let right = example(1, 1);
        check_pair(&left, &right)?;
        let mut steps = forced(&right.row);
        let mut early = step("selected_eos", EOS, 900, Some(0));
        early.action_score = Some(-2);
        steps.push(early);
        let exact = check_arm(
            "learned",
            &right,
            &Arm {
                steps,
                terminal: true,
                first_mismatch: Some(3),
            },
            &mut BTreeMap::new(),
        )?;
        assert_eq!(exact.1, Some(3));
        assert_eq!(right.common.own_choice, Action::Copy);
        Ok(())
    }

    #[test]
    fn native_eos_and_open_cap_are_scored_from_committed_tokens() -> Result<()> {
        let obs = example(0, 0);
        let mut steps = forced(&obs.row);
        steps.push(step("native", EOS, EOS, None));
        let mut native = Arm {
            steps,
            terminal: true,
            first_mismatch: None,
        };
        assert_eq!(
            check_arm("native", &obs, &native, &mut BTreeMap::new())?.1,
            None
        );
        native.terminal = false;
        assert!(check_stop(&native, &obs.row.expected).is_err());
        native.steps.pop();
        for _ in 2..CAP {
            native.steps.push(step("native", 900, 900, None));
        }
        native.first_mismatch = Some(3);
        assert_eq!(
            check_arm("native", &obs, &native, &mut BTreeMap::new())?.1,
            Some(3)
        );
        native.steps.pop();
        assert!(check_stop(&native, &obs.row.expected).is_err());
        Ok(())
    }
}
