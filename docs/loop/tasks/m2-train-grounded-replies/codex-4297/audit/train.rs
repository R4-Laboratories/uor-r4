use std::{cmp::Reverse, fs, path::Path, thread};

use serde_json::{json, Value};

use crate::identity::{Ids, Pairs, Reference};
use crate::packet::{self, Bank, Row, Sign};
use crate::replay::Arm;
use crate::signed::{self, Edge, Objective, Totals, Votes};
use crate::{
    check, edge_list, equal_rows, read_json, read_lines, replay, same, sha, write_json, Result,
};

const PROPOSALS: usize = 68;

struct Input<'a> {
    rows: &'a [Row],
    banks: &'a [Bank; 3],
    base: &'a [u8],
    state: &'a [Sign; 2],
    edges: &'a [Edge<'a>],
    directions: &'a Votes,
    physical: &'a Path,
    far_rows: usize,
    finite: bool,
}

fn read_state(raw: &[u8], finite: bool) -> Result<[Sign; 2]> {
    if !finite {
        return Ok(packet::state(raw)?);
    }
    check(raw.len() == 13_336, "wrong finite native state length")?;
    let mut active_view = raw.to_vec();
    for start in [6_156, 12_820] {
        let count = u32::from_le_bytes(raw[start..start + 4].try_into()?) as usize;
        check(count <= 8, "wrong finite active transport slot count")?;
        active_view[start + 4 + 64 * count..start + 4 + 64 * 8].fill(0);
    }
    // The fixture deliberately writes sentinel bytes in inactive transport slots.
    // Decode only active slots. Exact replay compares the full original bytes.
    Ok(packet::state(&active_view)?)
}

struct Candidate {
    id: Option<usize>,
    totals: Totals,
    changed: usize,
    state: Vec<u8>,
    joint: Vec<Value>,
    fields: [u64; 2],
}

impl Candidate {
    fn read(input: &Input<'_>, id: Option<usize>, observed: &Value) -> Result<Self> {
        let stem = id.map_or_else(|| "base".into(), |i| i.to_string());
        let (state, changed) = match id {
            Some(i) => signed::proposal(input.base, input.state, input.directions, i)?,
            None => (input.base.to_vec(), 0),
        };
        check(
            state == fs::read(input.physical.join(format!("{stem}.state")))?,
            format!("{stem}: physical native state differs"),
        )?;
        let signs = read_state(&state, input.finite)?;
        let mut totals = Totals::default();
        let mut expected = Vec::with_capacity(input.edges.len());
        for (i, edge) in input.edges.iter().enumerate() {
            let row = i / 2;
            let read = edge.read(&signs)?;
            if row < input.far_rows {
                totals.accumulate(read.totals());
            }
            expected.push(edge.record(row, &read, false));
        }
        let signed = equal_rows(
            &expected,
            &read_lines(&input.physical.join(format!("{stem}.edges.jsonl")))?,
            &format!("{stem} signed"),
        )?;
        same(
            &totals.value(id, changed),
            observed,
            &format!("{stem} score"),
        )?;
        let joint = replay(input.rows, &signs, input.banks, Arm::Learned);
        let joint_fields = equal_rows(
            &joint,
            &read_lines(&input.physical.join(format!("{stem}.reader.jsonl")))?,
            &format!("{stem} joint"),
        )?;
        Ok(Self {
            id,
            totals,
            changed,
            state,
            joint,
            fields: [signed, joint_fields],
        })
    }
}

pub struct Round {
    candidates: Vec<Candidate>,
    pub value: Value,
}

pub fn round(
    arm: &str,
    pooled: &Path,
    geometry: &Path,
    labels: &Path,
    base: &Path,
    far_rows: usize,
    physical: &Path,
) -> Result<Round> {
    let objective = Objective::read(arm)?;
    let rows = packet::text(&fs::read(geometry)?)?;
    let banks = packet::pooled(&fs::read(pooled)?)?;
    let labels = signed::labels(&fs::read(labels)?, rows.len())?;
    let base = fs::read(base)?;
    let finite = rows.len() < 256;
    let state = read_state(&base, finite)?;
    check(
        far_rows > 0 && far_rows < rows.len(),
        "invalid far-row split",
    )?;
    let completed = json!({"schema":"uor-r4.codex-4297.bridge-fit/1","mode":arm,"rows":rows.len(),
        "common_far_rows":far_rows,"proposals":PROPOSALS,"state_bytes":base.len()});
    same(
        &completed,
        &read_json(&physical.join("complete.json"))?,
        "round completion",
    )?;
    let edges = edge_list(&rows, &labels, &state)?;
    let mut directions = Votes::default();
    for edge in &edges[..2 * far_rows] {
        let read = edge.read(&state)?;
        check(read.eligible, "TRAIN endorsement edge is not admitted")?;
        edge.vote(&read, &state, objective, &mut directions);
    }
    let directions_log = directions
        .iter()
        .enumerate()
        .flat_map(|(sign, slots)| {
            slots
                .iter()
                .enumerate()
                .map(move |(slot, votes)| json!({"sign":sign,"slot":slot,"votes":votes}))
        })
        .collect::<Vec<_>>();
    equal_rows(
        &directions_log,
        &read_lines(&physical.join("directions.jsonl"))?,
        "per-axis votes",
    )?;
    let menu = (0..PROPOSALS).map(|i| {
        let group = i / 4;
        let center = group != 16;
        json!({"id":i,"group":group,"step":([1,2,4,8][i % 4]),"kind":if center {"center"} else {"all"},
            "sign":center.then_some(group / 8),"slot":center.then_some(group % 8)})
    }).collect::<Vec<_>>();
    equal_rows(
        &menu,
        &read_lines(&physical.join("menu.jsonl"))?,
        "canonical full proposal schedule",
    )?;
    let scores = read_lines(&physical.join("scores.jsonl"))?;
    check(
        scores.len() == PROPOSALS + 1,
        "physical score count changed",
    )?;
    let input = Input {
        rows: &rows,
        banks: &banks,
        base: &base,
        state: &state,
        edges: &edges,
        directions: &directions,
        physical,
        far_rows,
        finite,
    };
    let jobs = scores.iter().enumerate().collect::<Vec<_>>();
    let candidates = thread::scope(|scope| -> Result<Vec<Candidate>> {
        let mut work = Vec::new();
        for group in jobs.chunks(jobs.len().div_ceil(4)) {
            let input = &input;
            work.push(scope.spawn(move || {
                group
                    .iter()
                    .map(|(index, observed)| {
                        Candidate::read(input, index.checked_sub(1), observed)
                            .map_err(|err| err.to_string())
                    })
                    .collect::<std::result::Result<Vec<_>, String>>()
            }));
        }
        let mut result = Vec::with_capacity(jobs.len());
        for job in work {
            result.extend(
                job.join()
                    .map_err(|_| "independent replay worker failed")??,
            );
        }
        Ok(result)
    })?;
    let signed_fields = candidates.iter().map(|c| c.fields[0]).sum::<u64>();
    let joint_fields = candidates.iter().map(|c| c.fields[1]).sum::<u64>();
    let value = json!({"pass":true,"rows":rows.len(),"common_far_rows":far_rows,"states":candidates.len(),"joint_fields":joint_fields,
        "signed_fields":signed_fields,"votes":directions_log,"changed_words":candidates.iter().map(|c|c.changed).collect::<Vec<_>>(),
        "state_sha256":candidates.iter().map(|c|sha(&c.state)).collect::<Vec<_>>()});
    Ok(Round { candidates, value })
}

fn loss(totals: Totals, objective: Objective) -> u64 {
    match objective {
        Objective::Strongest => totals.strongest_loss,
        Objective::All => totals.all_loss,
    }
}

struct Graded {
    pairs: Pairs,
    lost: Ids,
}

fn record(candidate: &Candidate, grade: &Graded, historical_far: &Ids) -> Value {
    let mut value = candidate.totals.value(candidate.id, candidate.changed);
    value["near_full"] = json!(grade.pairs.full_near);
    value["far_full"] = json!(grade.pairs.full_far);
    value["lost_protected"] = json!(grade.lost);
    value["far_gained"] = json!(grade
        .pairs
        .full_far
        .difference(historical_far)
        .collect::<Vec<_>>());
    value["far_lost"] = json!(historical_far
        .difference(&grade.pairs.full_far)
        .collect::<Vec<_>>());
    value
}

fn election(round: &Round, arm: &str, grades: &[Graded]) -> Result<(usize, bool, usize)> {
    let objective = Objective::read(arm)?;
    let feasible = round
        .candidates
        .iter()
        .zip(grades)
        .enumerate()
        .filter(|(_, (_, grade))| grade.lost.is_empty());
    let safe = feasible.clone().count();
    let winner = feasible
        .min_by_key(|&(index, (candidate, grade))| {
            (
                loss(candidate.totals, objective),
                Reverse(grade.pairs.full_far.len()),
                candidate.changed,
                index,
            )
        })
        .map(|(index, _)| index);
    Ok((winner.unwrap_or(0), winner.is_some(), safe))
}

fn same_record(wanted: &Value, got: &Value, where_at: &str) -> Result<()> {
    let mut normalized = got.clone();
    for field in [
        "near_full",
        "far_full",
        "lost_protected",
        "far_gained",
        "far_lost",
    ] {
        let actual: Vec<String> = serde_json::from_value(got[field].clone())?;
        let set: Ids = actual.iter().cloned().collect();
        check(
            set.len() == actual.len(),
            format!("repeated original pair in {where_at}.{field}"),
        )?;
        normalized[field] = json!(set);
    }
    same(wanted, &normalized, where_at)?;
    Ok(())
}

fn examine_round(
    result: &Round,
    arm: &str,
    refs: &[Reference],
    protected: &Ids,
    historical_far: &Ids,
    physical_grade: &Path,
) -> Result<(usize, bool, Value)> {
    let grades = result
        .candidates
        .iter()
        .map(|candidate| {
            let pairs = Pairs::read(refs, &candidate.joint, protected)?;
            let lost = protected.difference(&pairs.full_near).cloned().collect();
            Ok(Graded { pairs, lost })
        })
        .collect::<Result<Vec<_>>>()?;
    let (winner, qualified, safe) = election(result, arm, &grades)?;
    let physical = read_json(&physical_grade.join("election.json"))?;
    let physical_candidates = read_lines(&physical_grade.join("candidates.jsonl"))?;
    check(
        physical_candidates.len() == result.candidates.len(),
        "training candidate grade count differs",
    )?;
    for (i, ((candidate, grade), actual)) in result
        .candidates
        .iter()
        .zip(&grades)
        .zip(&physical_candidates)
        .enumerate()
    {
        same_record(
            &record(candidate, grade, historical_far),
            actual,
            &format!("{arm} election candidate {i}"),
        )?;
    }
    let chose = record(&result.candidates[winner], &grades[winner], historical_far);
    let base = record(&result.candidates[0], &grades[0], historical_far);
    same_record(&base, &physical["base"], &format!("{arm} election base"))?;
    same_record(
        &chose,
        &physical["chosen"],
        &format!("{arm} election winner"),
    )?;
    let objective = Objective::read(arm)?;
    let before = loss(result.candidates[0].totals, objective);
    let after = loss(result.candidates[winner].totals, objective);
    let first_rejection = (1..grades.len()).find(|&i| !grades[i].lost.is_empty());
    let base_safe = usize::from(grades[0].lost.is_empty());
    check(
        physical["arm"] == arm
            && physical["proposals"] == PROPOSALS
            && physical["qualified"] == qualified
            && physical["eligible_proposals"] == safe - base_safe
            && physical["objective_before"] == before
            && physical["objective_after"] == after
            && physical["first_near_rejection"]["id"] == json!(first_rejection.map(|i| i - 1)),
        format!("{arm} election decision differs"),
    )?;
    let first_rejection = first_rejection.map(
        |i| json!({"id":i-1,"lost":grades[i].lost,"first":grades[i].pairs.first_lost_protected}),
    );
    Ok((
        winner,
        qualified,
        json!({"independent":result.value,"protected_feasible":safe,"qualified":qualified,"chosen":chose,
        "base":base,"chosen_pairs":grades[winner].pairs.value(),"objective_before":before,"objective_after":after,"first_near_rejection":first_rejection}),
    ))
}

fn bundle(tag: u8, bytes: &[u8], state: &[u8]) -> Result<()> {
    check(
        bytes.len() == 16 + state.len()
            && &bytes[..8] == b"UORBRG01"
            && bytes[8] == tag
            && bytes[9..16] == [0; 7]
            && &bytes[16..] == state,
        "lossless operator-bound state differs",
    )
}

pub fn study(
    historical: &Path,
    pooled: &Path,
    panel: &Path,
    physical: &Path,
    out: &Path,
) -> Result<()> {
    let train = panel.join("mixed");
    let premise = read_json(&historical.join("premise.json"))?;
    let protected: Ids = serde_json::from_value(premise["protected"].clone())?;
    let historical_far: Ids = serde_json::from_value(premise["far"].clone())?;
    check(
        protected.len() == 24 && historical_far.len() == 40,
        "original historical TRAIN pair sets changed",
    )?;
    let selection = read_json(&physical.join("selection.json"))?;
    let mut arms = serde_json::Map::new();
    for (arm, repr, tag) in [
        ("strongest", "blind", 0),
        ("all", "blind", 0),
        ("ordered-all", "ordered", 1),
    ] {
        let shape = train.join(repr).join("text.bin");
        let targets = train.join("links.bin");
        let packet = packet::text(&fs::read(&shape)?)?;
        let labels = signed::labels(&fs::read(&targets)?, packet.len())?;
        let refs = Reference::read(
            &train,
            &train.join(repr).join("coordinates.jsonl"),
            &packet,
            &labels,
        )?;
        check(
            refs.len() == 256
                && refs
                    .iter()
                    .enumerate()
                    .all(|(i, r)| (i < 128) == (r.distance == "far")),
            "TRAIN far/near panel changed",
        )?;
        let root = physical.join("train").join(arm);
        let first_path = root.join("round-1");
        let first = round(
            arm,
            pooled,
            &shape,
            &targets,
            &historical.join("states/protected.state"),
            128,
            &first_path.join("device"),
        )?;
        let (i, _, first_grade) = examine_round(
            &first,
            arm,
            &refs,
            &protected,
            &historical_far,
            &first_path.join("grade"),
        )?;
        let first_stem = first.candidates[i]
            .id
            .map_or_else(|| "base".into(), |id| id.to_string());
        let carried = first_path
            .join("device")
            .join(format!("{first_stem}.state"));
        let second_path = root.join("round-2");
        let second = round(
            arm,
            pooled,
            &shape,
            &targets,
            &carried,
            128,
            &second_path.join("device"),
        )?;
        let (j, qualified, second_grade) = examine_round(
            &second,
            arm,
            &refs,
            &protected,
            &historical_far,
            &second_path.join("grade"),
        )?;
        let retained = &second.candidates[j].state;
        let final_path = physical.join("states").join(format!("{arm}.state"));
        check(
            *retained == fs::read(&final_path)?,
            format!("{arm} elected state bytes differ"),
        )?;
        bundle(
            tag,
            &fs::read(final_path.with_extension("bridge"))?,
            retained,
        )?;
        same_record(
            &first_grade["base"],
            &selection[arm]["initial"],
            "initial selection",
        )?;
        same_record(
            &first_grade["chosen"],
            &selection[arm]["first"],
            "first selection",
        )?;
        same_record(
            &second_grade["chosen"],
            &selection[arm]["final"],
            "final selection",
        )?;
        check(
            selection[arm]["qualified"] == qualified
                && selection[arm]["objective_before"] == first_grade["objective_before"]
                && selection[arm]["objective_after"] == second_grade["objective_after"]
                && selection[arm]["round_ids"]
                    == json!([first.candidates[i].id, second.candidates[j].id]),
            format!("{arm} final election or scope differs"),
        )?;
        arms.insert(arm.into(), json!({"rounds":[first_grade, second_grade],"qualified":qualified,"state_sha256":sha(retained)}));
    }
    write_json(
        out,
        &json!({"pass":true,"batches":6,"arms":arms,"protected":protected,"historical_far":historical_far}),
    )
}

pub fn fixture(pooled: &Path, physical: &Path, out: &Path) -> Result<()> {
    let mut summaries = serde_json::Map::new();
    let text = physical.join("text.bin");
    let labels = physical.join("labels.bin");
    let parent = physical.join("parent.state");
    for (name, arm, geometry, base) in [
        ("strongest", "strongest", text.clone(), parent.clone()),
        ("all", "all", text.clone(), parent.clone()),
        ("ordered-all", "ordered-all", text.clone(), parent.clone()),
        ("repeated", "all", text.clone(), parent.clone()),
        (
            "remine",
            "strongest",
            text.clone(),
            physical.join("strongest/1.state"),
        ),
        ("zero", "all", text.clone(), physical.join("all/1.state")),
        (
            "boundary",
            "all",
            physical.join("boundary-text.bin"),
            physical.join("boundary-parent.state"),
        ),
    ] {
        let result = round(
            arm,
            pooled,
            &geometry,
            &labels,
            &base,
            2,
            &physical.join(name),
        )?;
        summaries.insert(name.into(), result.value);
    }
    let votes = &summaries["strongest"]["votes"][0]["votes"];
    let all = &summaries["all"]["votes"][0]["votes"];
    let remine = &summaries["remine"]["votes"][0]["votes"];
    check(
        [votes[0].as_i64(), votes[1].as_i64()] == [Some(0), Some(2)]
            && [all[0].as_i64(), all[1].as_i64()] == [Some(2), Some(2)]
            && [remine[0].as_i64(), remine[1].as_i64()] == [Some(2), Some(4)]
            && summaries["all"] == summaries["repeated"],
        "finite strict-vote directions or retry disagree",
    )?;
    let rows = packet::text(&fs::read(text)?)?;
    let labels = signed::labels(&fs::read(labels)?, rows.len())?;
    crate::controls::aid(&fs::read(physical.join("aid.bin"))?, &rows, &labels)?;
    let banks = packet::pooled(&fs::read(pooled)?)?;
    let mut control_fields = 0;
    for arm in ["correct", "exchanged"] {
        let wanted = rows
            .iter()
            .zip(&labels)
            .enumerate()
            .map(|(i, (row, &label))| crate::controls::frozen(i, row, &banks, Some(label), arm))
            .collect::<Result<Vec<_>>>()?;
        let recorded = read_lines(&physical.join(format!("official/{arm}.reader.jsonl")))?;
        control_fields += equal_rows(&wanted, &recorded, arm)?;
    }
    let original_pairs = |predictions: Vec<Value>| -> Vec<usize> {
        predictions
            .chunks_exact(2)
            .enumerate()
            .filter(|(pair, rows)| {
                rows.iter().enumerate().all(|(offset, raw)| {
                    let i = 2 * pair + offset;
                    let [a, e] = labels[i];
                    raw["tuple"]["ties"] == 1
                        && raw["tuple"]["sites"] == json!([a[1], e[1], 6])
                        && raw["links"][0]["word"] == a[0]
                        && raw["links"][1]["word"] == e[0]
                })
            })
            .map(|(pair, _)| pair)
            .collect()
    };
    let protected = original_pairs(read_lines(&physical.join("all/base.reader.jsonl"))?);
    let substitution = original_pairs(read_lines(&physical.join("all/1.reader.jsonl"))?);
    let kept: Vec<_> = substitution.into_iter().filter(|&pair| pair > 0).collect();
    check(
        protected == [1] && kept == [2],
        "fixture fails to catch count-preserving substitution",
    )?;
    write_json(
        out,
        &json!({"pass":true,"rounds":summaries,"protected_near":protected,"substituted_near":kept,"supplied_control_fields":control_fields}),
    )
}
