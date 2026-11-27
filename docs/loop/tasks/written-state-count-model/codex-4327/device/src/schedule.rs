use super::*;
use std::collections::VecDeque;

#[derive(Deserialize)]
struct Key {
    row_id: String,
    pair_id: String,
    side: u32,
    contrast: u32,
    family: String,
    cohort: String,
    split: String,
    source_issue: u32,
    context_id: String,
    expected_word_count: u32,
    expected_first_period_action: String,
    expected_post_b_action: Option<String>,
    pair_axis: Option<String>,
    selector_swap_id: Option<String>,
    one_word_clause_ordinal: Option<u32>,
    operative_clause_ordinal: Option<u32>,
}

impl Key {
    fn validate(&self, row: &Input) -> Result<()> {
        let core = &row.core;
        let count = core.requested();
        let expected_post_b = match count {
            1 => Some("stop"),
            2 => Some("copy"),
            _ => None,
        };
        if row
            .metadata
            .get("row_id")
            .and_then(serde_json::Value::as_str)
            != Some(self.row_id.as_str())
            || self.row_id != format!("{}:{}", self.pair_id, self.side)
            || self.pair_id != core.pair_id
            || self.side != core.side
            || self.contrast != core.contrast
            || self.family != core.family
            || self.split != "train"
            || self.expected_word_count != count + 1
            || self.expected_first_period_action != if count == 0 { "stop" } else { "copy" }
            || self.expected_post_b_action.as_deref() != expected_post_b
        {
            return Err(format!("TRAIN key disagrees with typed row {}", self.row_id).into());
        }
        match (
            self.source_issue,
            self.cohort.as_str(),
            self.pair_axis.as_deref(),
        ) {
            (4325, "fresh", Some("clause-order-swap")) => {
                let (Some(order @ (1 | 2)), Some(operative @ (1 | 2))) =
                    (self.one_word_clause_ordinal, self.operative_clause_ordinal)
                else {
                    return Err(format!("invalid selector ordinals on {}", self.row_id).into());
                };
                if core.contrast != 0
                    || (operative != order) != (count == 1)
                    || self.selector_swap_id.as_deref()
                        != Some(format!("{}-o{order}", self.context_id).as_str())
                {
                    return Err(format!("invalid fixed-order selector key {}", self.row_id).into());
                }
            }
            (4325, "fresh", Some("direct-count-control"))
                if self.selector_swap_id.is_none()
                    && self.one_word_clause_ordinal.is_none()
                    && self.operative_clause_ordinal.is_none() => {}
            (4322, "rehearsal-4322", _) | (4323, "rehearsal-4323", _) => {}
            _ => return Err(format!("unknown TRAIN identity {}", self.row_id).into()),
        }
        if core.boundaries()?.2.len() < 3 {
            return Err(format!("TRAIN row {} has no distant root pair", self.row_id).into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    Selector,
    DirectOneTwo,
    DirectTwoThree,
    Rehearsal,
    SameAction,
    NoEligibleSibling,
    UnmatchedExposure,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Selector => "selector",
            Self::DirectOneTwo => "direct_one_two",
            Self::DirectTwoThree => "direct_two_three",
            Self::Rehearsal => "rehearsal",
            Self::SameAction => "same_action",
            Self::NoEligibleSibling => "no_eligible_sibling",
            Self::UnmatchedExposure => "unmatched_exposure",
        }
    }
}

#[derive(Serialize)]
pub(super) struct Member<'a> {
    pub(super) train_index: u32,
    row_id: &'a str,
    pair_id: &'a str,
    side: u32,
    copy: bool,
    original_offset: usize,
}

#[derive(Serialize)]
pub(super) struct Group<'a> {
    epoch: u32,
    pub(super) stage: u32,
    pub(super) kind: Kind,
    pub(super) first: Member<'a>,
    pub(super) second: Option<Member<'a>>,
}

impl Group<'_> {
    pub(super) fn device_words(&self) -> [u32; 5] {
        [
            self.first.train_index,
            self.second
                .as_ref()
                .map_or(NONE, |member| member.train_index),
            self.stage,
            u32::from(self.first.copy),
            self.epoch,
        ]
    }
}

fn fixed_facts(rows: &[Input], a: usize, b: usize) -> Result<bool> {
    let (left, right) = (&rows[a].core, &rows[b].core);
    let (_, left_start, _) = left.boundaries()?;
    let (_, right_start, _) = right.boundaries()?;
    Ok(left.prefix[..left_start] == right.prefix[..right_start]
        && left.positions == right.positions
        && left.ids == right.ids
        && left.family == right.family)
}

fn siblings(rows: &[Input], keys: &[Key]) -> Result<[Vec<Option<(usize, Kind)>>; 2]> {
    let mut pairs = BTreeMap::new();
    for (index, key) in keys.iter().enumerate() {
        if key.cohort != "fresh" {
            continue;
        }
        let (stage, kind, id) = if let Some(selector) = &key.selector_swap_id {
            (0, Kind::Selector, selector)
        } else if key.contrast == 0 {
            (0, Kind::DirectOneTwo, &key.pair_id)
        } else {
            (1, Kind::DirectTwoThree, &key.pair_id)
        };
        let (members, _) = pairs
            .entry((stage, kind.name(), id))
            .or_insert((Vec::new(), kind));
        members.push(index);
    }
    let mut result = [vec![None; rows.len()], vec![None; rows.len()]];
    for ((stage, _, id), (members, kind)) in pairs {
        let &[a, b] = members.as_slice() else {
            return Err(format!("TRAIN pair {id} has {} members", members.len()).into());
        };
        let (left, right) = (&keys[a], &keys[b]);
        if left.context_id != right.context_id
            || left.one_word_clause_ordinal != right.one_word_clause_ordinal
            || !fixed_facts(rows, a, b)?
            || [rows[a].core.requested(), rows[b].core.requested()]
                .into_iter()
                .min()
                != Some(stage as u32)
            || [rows[a].core.requested(), rows[b].core.requested()]
                .into_iter()
                .max()
                != Some(stage as u32 + 1)
        {
            return Err(format!("TRAIN pair {id} changes facts or lacks opposed labels").into());
        }
        result[stage][a] = Some((b, kind));
        result[stage][b] = Some((a, kind));
    }
    Ok(result)
}

fn group_exposures<'a>(
    tape: &mut Vec<Group<'a>>,
    rows: &'a [Input],
    keys: &[Key],
    links: &[Option<(usize, Kind)>],
    exposures: &[usize],
    epoch: u32,
    stage: u32,
) {
    let mut pending = vec![VecDeque::new(); rows.len()];
    for (offset, &index) in exposures.iter().enumerate() {
        pending[index].push_back(offset);
    }
    let member = |index: usize, original_offset| {
        let row = &rows[index];
        Member {
            train_index: index as u32,
            row_id: row.metadata["row_id"].as_str().unwrap_or_default(),
            pair_id: &row.core.pair_id,
            side: row.core.side,
            copy: row.core.requested() > stage,
            original_offset,
        }
    };
    for (offset, &index) in exposures.iter().enumerate() {
        if pending[index].front() != Some(&offset) {
            continue;
        }
        pending[index].pop_front();
        let (kind, second) = if let Some((other, opposed_kind)) = links[index] {
            match pending[other].pop_front() {
                Some(peer_offset) => (opposed_kind, Some(member(other, peer_offset))),
                None => (Kind::UnmatchedExposure, None),
            }
        } else if keys[index].cohort != "fresh" {
            (Kind::Rehearsal, None)
        } else if stage == 0 {
            (Kind::SameAction, None)
        } else {
            (Kind::NoEligibleSibling, None)
        };
        tape.push(Group {
            epoch,
            stage,
            kind,
            first: member(index, offset),
            second,
        });
    }
}

pub(super) fn load<'a>(rows: &'a [Input], path: &str) -> Result<(String, Vec<Group<'a>>)> {
    // The manifest binds the immutable keys file as well as TRAIN rows.
    let bytes = fs::read(path)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let keys: Vec<Key> = BufReader::new(bytes.as_slice())
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;
    if keys.len() != rows.len() || rows.len() % 2 != 0 || rows.len() > u32::MAX as usize {
        return Err("TRAIN rows and keys must align and have an even tape".into());
    }
    let mut stage_one: [Vec<usize>; 2] = Default::default();
    for (index, (key, row)) in keys.iter().zip(rows).enumerate() {
        key.validate(row)?;
        if row.core.requested() > 0 {
            stage_one[(row.core.requested() - 1) as usize].push(index);
        }
    }
    if stage_one.iter().any(Vec::is_empty) {
        return Err("both post-B action classes are required".into());
    }
    let siblings = siblings(rows, &keys)?;
    let stage_zero: Vec<_> = (0..rows.len()).collect();
    let mut tape = Vec::new();
    for epoch in 0..EPOCHS {
        let post_b: Vec<_> = (0..rows.len())
            .map(|slot| {
                let pool = &stage_one[slot % 2];
                let offset = (epoch * (rows.len() / 2) + slot / 2) % pool.len();
                pool[offset]
            })
            .collect();
        for (stage, exposures) in [&stage_zero, &post_b].into_iter().enumerate() {
            group_exposures(
                &mut tape,
                rows,
                &keys,
                &siblings[stage],
                exposures,
                epoch as u32,
                stage as u32,
            );
        }
    }
    Ok((hash, tape))
}

pub(super) fn summary(tape: &[Group<'_>]) -> serde_json::Value {
    let mut groups = [BTreeMap::new(), BTreeMap::new()];
    let mut classes = [[0, 0], [0, 0]];
    for group in tape {
        let stage = group.stage as usize;
        *groups[stage].entry(group.kind.name()).or_insert(0) += 1;
        classes[stage][usize::from(group.first.copy)] += 1;
        if let Some(second) = &group.second {
            classes[stage][usize::from(second.copy)] += 1;
        }
    }
    serde_json::json!({"groups_by_kind": groups, "stop_copy_rows": classes})
}

pub(super) fn emit(panel: &str, keys: &str, output: &Path) -> Result<()> {
    let (input_sha256, rows) = run::load_panel(panel)?;
    let (keys_sha256, tape) = load(&rows, keys)?;
    write_new(output, &tape)?;
    println!(
        "{}",
        serde_json::json!({
            "rows": rows.len(), "groups": tape.len(), "row_exposures": rows.len() * EPOCHS * 2,
            "input_sha256": input_sha256, "keys_sha256": keys_sha256,
            "schedule_sha256": file_hash(output)?, "schedule": summary(&tape),
        })
    );
    Ok(())
}
