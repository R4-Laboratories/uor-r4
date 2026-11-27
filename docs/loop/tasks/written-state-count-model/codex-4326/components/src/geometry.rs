//! Independent inventory of the recorded operator's addresses and signed contributions.
//! Reads archived GPU scores for comparison. Runs neither writer nor model.

use super::*;
use std::collections::BTreeMap;

const ROOTS_SHA: &str = "875a0857dfef840d88a64c90a5777a6154a1674cde08a71429e2cca93fb7bb5a";
const PERIOD_SHA: &str = "521e35bffabc060a5b0be12078c371a639e55870e07cfef678ee4708539c4bcd";
const POST_B_SHA: &str = "92daec94c940a6120cec86e8155bc5bd592e4c2a8379978f6eded43deb75c03b";
const EOS: u32 = 1;
const PERIOD: u32 = 16;
const LF: u32 = 201;
const QUESTION: u32 = 33;
const MARKER: [u32; 6] = [LF, 35, 560, 652, 714, 28];
const CONTEXT: usize = 192;

struct Split {
    name: &'static str,
    digests: [&'static str; 4],
    rows: usize,
    scoped_contexts: usize,
    post_b_pairs: usize,
}

const SPLITS: [Split; 2] = [
    Split {
        name: "train",
        digests: [
            "ed20c8b0789d86585baee78dd26cff0bfa188ba28b83e74ecca88ff38213f698",
            "e943be8c98e985e3608720eb93522b7da802495c843392f86a203d3e733eeb3b",
            "56b81e09ac2d36b25838071867103024082b329249b50b6a1d601d2b8ea60980",
            "adb913fde5829003ebac03dc3c2849edbbaf2f2c00638604da987565f58657b9",
        ],
        rows: 768,
        scoped_contexts: 64,
        post_b_pairs: 112,
    },
    Split {
        name: "held",
        digests: [
            "62a4099f172eea49a8fa8e5231744a45f81bcca15023271ffc8a0254cd76fe44",
            "1349c5e3fc4fde9c40e7b041edf20ae493a71080d46d960921848a008c5b8a7f",
            "f63549493d00594a0985ddf4cf335966376d3b09fc500213a1ef46822a7eaac9",
            "dbd24cf070c507ba43c2483ad60d0f417838eb9fcecf078c0b42974ba0f8a09f",
        ],
        rows: 512,
        scoped_contexts: 64,
        post_b_pairs: 64,
    },
];

#[derive(Deserialize, PartialEq, Eq)]
struct Panel {
    row_id: String,
    pair_id: String,
    side: usize,
    contrast: usize,
    family: String,
    prefix: Vec<u32>,
    positions: [u32; 4],
    ids: [u32; 4],
    expected: Vec<u32>,
}

#[derive(Deserialize)]
struct Key {
    row_id: String,
    pair_id: String,
    side: usize,
    contrast: usize,
    context_id: String,
    operative_clause_ordinal: Option<usize>,
    one_word_clause_ordinal: Option<usize>,
    selector_swap_id: Option<String>,
    pair_axis: Option<String>,
}

#[derive(Deserialize)]
struct GpuRow {
    #[serde(flatten)]
    panel: Panel,
    arms: Arms,
}

#[derive(Deserialize)]
struct Arms {
    learned: Trajectory,
    first: Trajectory,
}

#[derive(Deserialize)]
struct Trajectory {
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    stage: Option<usize>,
    action_score: Option<i32>,
    committed_id: u32,
    selected_source: String,
    requested_pos: Option<u32>,
    requested_id: Option<u32>,
    refusal: u32,
}

struct Geometry {
    roots: Vec<u16>,
    weights: [Vec<i16>; 3],
    touched: [[bool; WORDS]; 2],
}

struct Row {
    panel: Panel,
    key: Key,
    features: Vec<usize>,
    stages: [Option<Contribution>; 2],
}

#[derive(Clone, Copy, Serialize)]
struct Contribution {
    frozen_gpu: i32,
    period_component: i32,
    post_b_component: i32,
    full_gpu: i32,
}

impl Contribution {
    fn add(self, other: Self) -> Self {
        Self {
            frozen_gpu: self.frozen_gpu + other.frozen_gpu,
            period_component: self.period_component + other.period_component,
            post_b_component: self.post_b_component + other.post_b_component,
            full_gpu: self.full_gpu + other.full_gpu,
        }
    }

    fn signed(self, sign: i32) -> Self {
        Self {
            frozen_gpu: self.frozen_gpu * sign,
            period_component: self.period_component * sign,
            post_b_component: self.post_b_component * sign,
            full_gpu: self.full_gpu * sign,
        }
    }
}

#[derive(Serialize)]
struct Difference {
    support: usize,
    l1: usize,
    coefficients_minus_2_to_plus_2_excluding_zero: [usize; 4],
    untouched_by_stage: [usize; 2],
    untouched_by_either_stage: usize,
    zero_net_by_component: [usize; 3],
    derived_residual: [i32; 3],
}

#[derive(Serialize)]
struct Pair<'a> {
    id: &'a str,
    context_id: &'a str,
    source_axis: &'a str,
    contrast: usize,
    stop_row: &'a str,
    copy_row: &'a str,
    raw_visits_stop_copy: [usize; 2],
    unique_support_stop_copy: [usize; 2],
    common_support: usize,
    stop_only_support: usize,
    copy_only_support: usize,
    difference: Difference,
    copy_minus_stop: Contribution,
}

#[derive(Serialize)]
struct Interaction<'a> {
    context_id: &'a str,
    row_ids_selector_major_one_clause_minor: [&'a str; 4],
    difference: Difference,
    copy_diagonal_minus_stop_diagonal: Contribution,
    positive_signed_scores_impossible_for_any_residual: bool,
    coefficients: Vec<(usize, i8)>,
}

#[derive(Serialize)]
struct Totals {
    records: usize,
    zero_difference: usize,
    matched_difference_support_min_max: [usize; 2],
    difference_support_sum: usize,
    difference_l1_sum: usize,
    untouched_by_stage_sum: [usize; 2],
    untouched_by_either_stage_sum: usize,
    no_stage_write_on_any_difference: [usize; 2],
    no_write_from_either_stage_on_any_difference: usize,
    zero_net_by_component_sum: [usize; 3],
    frozen_margin_sign_negative_zero_positive: [usize; 3],
    period_delta_sign_negative_zero_positive: [usize; 3],
    post_b_delta_sign_negative_zero_positive: [usize; 3],
    full_margin_sign_negative_zero_positive: [usize; 3],
    full_flip_for_positive_copy: usize,
    frozen_flip_for_positive_copy: usize,
    counterfactual_component_only_flips_positive_copy: [usize; 2],
    impossible_for_any_residual: usize,
}

impl Default for Totals {
    fn default() -> Self {
        Self {
            records: 0,
            zero_difference: 0,
            matched_difference_support_min_max: [usize::MAX, 0],
            difference_support_sum: 0,
            difference_l1_sum: 0,
            untouched_by_stage_sum: [0; 2],
            untouched_by_either_stage_sum: 0,
            no_stage_write_on_any_difference: [0; 2],
            no_write_from_either_stage_on_any_difference: 0,
            zero_net_by_component_sum: [0; 3],
            frozen_margin_sign_negative_zero_positive: [0; 3],
            period_delta_sign_negative_zero_positive: [0; 3],
            post_b_delta_sign_negative_zero_positive: [0; 3],
            full_margin_sign_negative_zero_positive: [0; 3],
            full_flip_for_positive_copy: 0,
            frozen_flip_for_positive_copy: 0,
            counterfactual_component_only_flips_positive_copy: [0; 2],
            impossible_for_any_residual: 0,
        }
    }
}

impl Totals {
    fn record(&mut self, difference: &Difference, margin: Contribution) {
        self.records += 1;
        self.zero_difference += usize::from(difference.support == 0);
        let [min, max] = self.matched_difference_support_min_max;
        self.matched_difference_support_min_max =
            [min.min(difference.support), max.max(difference.support)];
        self.difference_support_sum += difference.support;
        self.difference_l1_sum += difference.l1;
        for stage in 0..2 {
            self.untouched_by_stage_sum[stage] += difference.untouched_by_stage[stage];
            self.no_stage_write_on_any_difference[stage] += usize::from(
                difference.support != 0
                    && difference.untouched_by_stage[stage] == difference.support,
            );
        }
        self.untouched_by_either_stage_sum += difference.untouched_by_either_stage;
        self.no_write_from_either_stage_on_any_difference += usize::from(
            difference.support != 0 && difference.untouched_by_either_stage == difference.support,
        );
        for (count, zeros) in self
            .zero_net_by_component_sum
            .iter_mut()
            .zip(difference.zero_net_by_component)
        {
            *count += zeros;
        }
        let bin = |value: i32| (value.signum() + 1) as usize;
        self.frozen_margin_sign_negative_zero_positive[bin(margin.frozen_gpu)] += 1;
        self.period_delta_sign_negative_zero_positive[bin(margin.period_component)] += 1;
        self.post_b_delta_sign_negative_zero_positive[bin(margin.post_b_component)] += 1;
        self.full_margin_sign_negative_zero_positive[bin(margin.full_gpu)] += 1;
    }

    fn record_pair(&mut self, difference: &Difference, stop: Contribution, copy: Contribution) {
        self.record(difference, copy.add(stop.signed(-1)));
        self.frozen_flip_for_positive_copy +=
            usize::from(stop.frozen_gpu < 0 && copy.frozen_gpu > 0);
        self.full_flip_for_positive_copy += usize::from(stop.full_gpu < 0 && copy.full_gpu > 0);
        self.counterfactual_component_only_flips_positive_copy[0] += usize::from(
            stop.frozen_gpu + stop.period_component < 0
                && copy.frozen_gpu + copy.period_component > 0,
        );
        self.counterfactual_component_only_flips_positive_copy[1] += usize::from(
            stop.frozen_gpu + stop.post_b_component < 0
                && copy.frozen_gpu + copy.post_b_component > 0,
        );
    }
}

fn instruction(row: &Panel) -> Result<&[u32]> {
    if row.prefix.len() > CONTEXT - 5 || row.prefix.iter().any(|&id| id >= WORDS as u32) {
        return Err("panel prefix exceeds registered bounds".into());
    }
    let user = row
        .prefix
        .strip_suffix(&MARKER)
        .ok_or("missing canonical assistant marker")?;
    let fence = user
        .windows(2)
        .rposition(|pair| pair == [LF, LF])
        .ok_or("missing fact/question fence")?;
    let question = user[fence + 2..]
        .iter()
        .rposition(|&id| id == QUESTION)
        .ok_or("missing question mark")?;
    let instruction = &user[fence + 3 + question..];
    if instruction.len() < 3 {
        return Err("instruction has no distant root pair".into());
    }
    for (&position, &id) in row.positions.iter().zip(&row.ids) {
        if position as usize >= fence || row.prefix[position as usize] != id {
            return Err("source identity differs from original prefix".into());
        }
    }
    if row.side > 1 || row.contrast > 1 {
        return Err("invalid panel contrast or side".into());
    }
    let requested = row.side + row.contrast;
    let expected: Vec<_> = [row.ids[0], PERIOD]
        .into_iter()
        .chain(row.ids[1..1 + requested].iter().copied())
        .chain([EOS])
        .collect();
    if row.expected != expected {
        return Err("expected trajectory differs from bound contrast and source IDs".into());
    }
    Ok(instruction)
}

fn raw_visits(tokens: usize) -> usize {
    (2..tokens).map(|later| 2 * (later - 1).min(15)).sum()
}

fn link_addresses(roots: &[u16], directive: &[u32]) -> Vec<usize> {
    fn positioned(earlier: u16, later: u16, delta: usize) -> usize {
        let ordered = earlier.rotate_left(5) ^ later;
        let salt = (delta as u16).wrapping_mul(0x9e37);
        let code = ordered ^ salt;
        usize::from((code ^ (code >> 7)) & 2047)
    }

    let mut seen = [false; WORDS];
    let mut addresses = Vec::new();
    for later in 2..directive.len() {
        let tail = roots[directive[later] as usize];
        for delta in 2..=later.min(16) {
            let head = roots[directive[later - delta] as usize];
            let forward = positioned(head, tail, delta);
            let backward = positioned(tail, head, delta) + WORDS / 2;
            for address in [forward, backward] {
                if !seen[address] {
                    addresses.push(address);
                    seen[address] = true;
                }
            }
        }
    }
    addresses
}

fn score_from_trace(record: &GpuRow, stage: usize) -> Result<i32> {
    let score = if stage == 0 {
        record
            .arms
            .learned
            .steps
            .get(2)
            .ok_or("missing GPU period choice")?
    } else {
        let steps = &record.arms.first.steps;
        let [a, period, b, score, ..] = steps.as_slice() else {
            return Err("missing GPU choice after supplied first action".into());
        };
        if a.committed_id != record.panel.ids[0]
            || period.committed_id != PERIOD
            || b.stage != Some(0)
            || b.selected_source != "verified_copy"
            || b.refusal != 0
            || b.requested_id != Some(record.panel.ids[1])
            || b.requested_pos != Some(record.panel.positions[1])
            || b.committed_id != record.panel.ids[1]
        {
            return Err("GPU post-B score lacks a common authenticated B".into());
        }
        score
    };
    if score.stage != Some(stage) {
        return Err("GPU trace choice has an unexpected stage index".into());
    }
    score
        .action_score
        .ok_or_else(|| "GPU trace choice missing score".into())
}

fn residual_from_state(components: &Path, state: &str, digest: &str) -> Result<Vec<i16>> {
    let bytes = pinned(components, state, digest)?;
    if bytes.len() != STATE_BYTES {
        return Err("component width differs from native state width".into());
    }
    Ok(bytes[PREFIX_BYTES..]
        .chunks_exact(2)
        .map(|word| i16::from_le_bytes([word[0], word[1]]))
        .collect())
}

impl Geometry {
    fn verify_contribution(
        &self,
        addresses: &[usize],
        stage: usize,
        frozen: &GpuRow,
        full: &GpuRow,
    ) -> Result<Contribution> {
        let phase = stage * (WORDS / 2);
        let [period_component, post_b_component, direct_full] =
            self.weights.each_ref().map(|weights| {
                addresses
                    .iter()
                    .map(|&address| i32::from(weights[address ^ phase]))
                    .sum::<i32>()
            });
        let frozen_gpu = score_from_trace(frozen, stage)?;
        let full_gpu = score_from_trace(full, stage)?;
        if period_component + post_b_component != direct_full
            || full_gpu - frozen_gpu != direct_full
        {
            return Err(format!("independent addresses differ from historical GPU delta on {} stage {stage}: frozen {frozen_gpu}, full {full_gpu}, residual {direct_full}", frozen.panel.row_id).into());
        }
        Ok(Contribution {
            frozen_gpu,
            period_component,
            post_b_component,
            full_gpu,
        })
    }

    fn difference(&self, coeffs: &[i8; WORDS]) -> Difference {
        let mut out = Difference {
            support: 0,
            l1: 0,
            coefficients_minus_2_to_plus_2_excluding_zero: [0; 4],
            untouched_by_stage: [0; 2],
            untouched_by_either_stage: 0,
            zero_net_by_component: [0; 3],
            derived_residual: [0; 3],
        };
        for (address, &coeff) in coeffs.iter().enumerate().filter(|(_, coeff)| **coeff != 0) {
            out.support += 1;
            out.l1 += usize::from(coeff.unsigned_abs());
            let bin = if coeff < 0 { coeff + 2 } else { coeff + 1 };
            out.coefficients_minus_2_to_plus_2_excluding_zero[bin as usize] += 1;
            for stage in 0..2 {
                out.untouched_by_stage[stage] += usize::from(!self.touched[stage][address]);
            }
            out.untouched_by_either_stage +=
                usize::from(!self.touched[0][address] && !self.touched[1][address]);
            for (component, weights) in self.weights.iter().enumerate() {
                out.zero_net_by_component[component] += usize::from(weights[address] == 0);
                out.derived_residual[component] += i32::from(coeff) * i32::from(weights[address]);
            }
        }
        out
    }
}

fn signed_features(coeffs: &mut [i8; WORDS], row: &Row, stage: usize, sign: i8) {
    let phase = stage * (WORDS / 2);
    for &address in &row.features {
        coeffs[address ^ phase] += sign;
    }
}

fn verify_residual_margin(diff: &Difference, margin: Contribution) -> Result<()> {
    if diff.derived_residual
        != [
            margin.period_component,
            margin.post_b_component,
            margin.full_gpu - margin.frozen_gpu,
        ]
    {
        return Err(
            "signed address difference does not reproduce residual margin contribution".into(),
        );
    }
    Ok(())
}

fn pair<'a>(
    geometry: &Geometry,
    id: &'a str,
    stop: &'a Row,
    copy: &'a Row,
    stage: usize,
) -> Result<(Pair<'a>, Contribution, Contribution)> {
    if stop.panel.side != 0
        || copy.panel.side != 1
        || stop.panel.ids != copy.panel.ids
        || stop.panel.positions != copy.panel.positions
        || stop.key.context_id != copy.key.context_id
        || stop.panel.contrast != stage
        || copy.panel.contrast != stage
    {
        return Err(format!("unmatched pair identity or labels for {id}").into());
    }
    let mut coeffs = [0; WORDS];
    signed_features(&mut coeffs, stop, stage, -1);
    signed_features(&mut coeffs, copy, stage, 1);
    let difference = geometry.difference(&coeffs);
    let stop_only = difference.coefficients_minus_2_to_plus_2_excluding_zero[1];
    let copy_only = difference.coefficients_minus_2_to_plus_2_excluding_zero[2];
    let common = stop.features.len() - stop_only;
    if copy.features.len() - copy_only != common {
        return Err("inconsistent common support".into());
    }
    let stop_score = stop.stages[stage].ok_or("missing audited STOP score")?;
    let copy_score = copy.stages[stage].ok_or("missing audited COPY score")?;
    let margin = copy_score.add(stop_score.signed(-1));
    verify_residual_margin(&difference, margin)?;
    Ok((
        Pair {
            id,
            context_id: &stop.key.context_id,
            source_axis: if stage == 0 {
                "selector-swap"
            } else {
                stop.key.pair_axis.as_deref().unwrap_or("archive")
            },
            contrast: stage,
            stop_row: &stop.panel.row_id,
            copy_row: &copy.panel.row_id,
            raw_visits_stop_copy: [
                raw_visits(instruction(&stop.panel)?.len()),
                raw_visits(instruction(&copy.panel)?.len()),
            ],
            unique_support_stop_copy: [stop.features.len(), copy.features.len()],
            common_support: common,
            stop_only_support: stop_only,
            copy_only_support: copy_only,
            difference,
            copy_minus_stop: margin,
        },
        stop_score,
        copy_score,
    ))
}

fn interaction<'a>(
    geometry: &Geometry,
    id: &'a str,
    rows: [&'a Row; 4],
) -> Result<Interaction<'a>> {
    let mut coeffs = [0; WORDS];
    let mut margin: Option<Contribution> = None;
    for (index, row) in rows.iter().enumerate() {
        let stop = index / 2 == index % 2;
        if stop != (row.panel.side == 0)
            || row.panel.ids != rows[0].panel.ids
            || row.panel.positions != rows[0].panel.positions
        {
            return Err(format!("unmatched four-cell identity or labels for {id}").into());
        }
        let sign = if stop { -1 } else { 1 };
        signed_features(&mut coeffs, row, 0, sign);
        let score = row.stages[0]
            .ok_or("missing audited four-cell score")?
            .signed(i32::from(sign));
        margin = Some(margin.map_or(score, |previous| previous.add(score)));
    }
    let margin = margin.ok_or("empty interaction")?;
    let difference = geometry.difference(&coeffs);
    verify_residual_margin(&difference, margin)?;
    Ok(Interaction {
        context_id: id,
        row_ids_selector_major_one_clause_minor: rows.map(|row| row.panel.row_id.as_str()),
        positive_signed_scores_impossible_for_any_residual: difference.support == 0
            && margin.frozen_gpu <= 0,
        difference,
        copy_diagonal_minus_stop_diagonal: margin,
        coefficients: coeffs
            .into_iter()
            .enumerate()
            .filter(|(_, coeff)| *coeff != 0)
            .collect(),
    })
}

fn load_panel(prior: &Path, split: &Split, geometry: &Geometry) -> Result<Vec<Row>> {
    let panel_path = prior.join("panel-v1").join(split.name);
    let raw_path = prior.join("measured-v1/raw").join(split.name);
    let panels: Vec<Panel> = rows(&pinned(&panel_path, "rows.jsonl", split.digests[0])?)?;
    let keys: Vec<Key> = rows(&pinned(&panel_path, "keys.jsonl", split.digests[1])?)?;
    let frozen: Vec<GpuRow> = rows(&pinned(
        &raw_path,
        "frozen/candidate.jsonl",
        split.digests[2],
    )?)?;
    let full: Vec<GpuRow> = rows(&pinned(&raw_path, "trained/joint.jsonl", split.digests[3])?)?;
    if [panels.len(), keys.len(), frozen.len(), full.len()] != [split.rows; 4] {
        return Err("panel and archived GPU rows have differing counts".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    panels
        .into_iter()
        .zip(keys)
        .zip(frozen.iter().zip(full.iter()))
        .map(|((panel, key), (frozen, full))| {
            if !seen.insert(panel.row_id.clone())
                || panel.row_id != key.row_id
                || panel.pair_id != key.pair_id
                || panel.side != key.side
                || panel.contrast != key.contrast
                || panel != frozen.panel
                || panel != full.panel
            {
                return Err(
                    format!("panel/key/GPU row identity mismatch for {}", panel.row_id).into(),
                );
            }
            let features = link_addresses(&geometry.roots, instruction(&panel)?);
            let stage_zero = geometry.verify_contribution(&features, 0, frozen, full)?;
            let stage_one = if panel.expected.len() >= 4 {
                Some(geometry.verify_contribution(&features, 1, frozen, full)?)
            } else {
                None
            };
            Ok(Row {
                panel,
                key,
                features,
                stages: [Some(stage_zero), stage_one],
            })
        })
        .collect()
}

#[derive(Default, Serialize)]
struct MatchedWrites {
    matched_slots: usize,
    slots_that_wrote: usize,
    writes_to_shared_addresses: usize,
    writes_to_distinguishing_addresses: usize,
    writing_slots_without_a_distinguishing_write: usize,
}

fn training_sibling_key(row: &Row, stage: usize) -> Option<(&'static str, &str)> {
    match stage {
        0 => row
            .key
            .selector_swap_id
            .as_deref()
            .map(|id| ("period_selector", id))
            .or_else(|| {
                (row.key.pair_axis.as_deref() == Some("selector-swap"))
                    .then_some(("period_archive_selector", row.panel.pair_id.as_str()))
            }),
        1 if row.panel.contrast == 1 && row.key.one_word_clause_ordinal.is_none() => {
            let cohort = if row.key.pair_axis.as_deref() == Some("direct-count-control") {
                "post_b_direct"
            } else {
                "post_b_archive"
            };
            Some((cohort, row.panel.pair_id.as_str()))
        }
        _ => None,
    }
}

fn verify_audit_addresses(
    tape: &[Slot],
    audit: &[Audit],
    training: &[Row],
) -> Result<BTreeMap<&'static str, MatchedWrites>> {
    let mut matched_siblings = BTreeMap::<_, [Option<&Row>; 2]>::new();
    for row in training {
        for stage in 0..2 {
            if let Some(key) = training_sibling_key(row, stage) {
                if matched_siblings.entry(key).or_default()[row.panel.side]
                    .replace(row)
                    .is_some()
                {
                    return Err(format!("duplicate TRAIN sibling side for {key:?}").into());
                }
            }
        }
    }
    let mut write_counts = BTreeMap::<_, MatchedWrites>::new();
    for (index, record) in audit.iter().enumerate() {
        let (stage, _, _, _) = check_slot(index, tape, record)?;
        let slot = &tape[index];
        let row = &training[slot.train_index];
        if slot.pair_id != row.panel.pair_id
            || slot.side as usize != row.panel.side
            || slot.joint_copy != (row.panel.side + row.panel.contrast > stage)
        {
            return Err(format!("scheduled target differs from TRAIN row at slot {index}").into());
        }
        let support = row.features.len();
        let visit_count = raw_visits(instruction(&row.panel)?.len());
        let phase = 17 * slot.epoch;
        if record.link_visits != visit_count || record.unique_coordinate_visits != support {
            return Err(format!(
                "address reconstruction disagrees with audited visits at slot {index}"
            )
            .into());
        }
        for (visit, &actual) in record.changed_coordinates.iter().enumerate() {
            let offset = if support < COORDINATE_CAP {
                visit
            } else {
                visit * support / COORDINATE_CAP
            };
            let expected = row.features[(phase + offset) % support] ^ (stage * (WORDS / 2));
            if actual != expected {
                return Err(
                    format!("selected address or visit order disagrees at slot {index}").into(),
                );
            }
        }
        if let Some(key) = training_sibling_key(row, stage) {
            let sibling = matched_siblings[&key][1 - row.panel.side]
                .ok_or_else(|| format!("missing opposed TRAIN sibling for {key:?}"))?;
            if sibling.panel.ids != row.panel.ids || sibling.panel.positions != row.panel.positions
            {
                return Err(format!("TRAIN sibling source did not match for {key:?}").into());
            }
            let stats = write_counts.entry(key.0).or_default();
            let writes = record.changed_coordinates.len();
            let shared = record
                .changed_coordinates
                .iter()
                .filter(|&&address| {
                    sibling
                        .features
                        .contains(&(address ^ (stage * (WORDS / 2))))
                })
                .count();
            stats.matched_slots += 1;
            stats.slots_that_wrote += usize::from(writes != 0);
            stats.writes_to_shared_addresses += shared;
            stats.writes_to_distinguishing_addresses += writes - shared;
            stats.writing_slots_without_a_distinguishing_write +=
                usize::from(writes != 0 && shared == writes);
        }
    }
    Ok(write_counts)
}

fn stats(
    geometry: &Geometry,
    split: &Split,
    rows: &[Row],
    dir: &Path,
) -> Result<serde_json::Value> {
    let mut contexts: BTreeMap<&str, [Option<&Row>; 4]> = BTreeMap::new();
    let mut post_b_pairs: BTreeMap<&str, [Option<&Row>; 2]> = BTreeMap::new();
    let mut archive_selector_pairs: BTreeMap<&str, [Option<&Row>; 2]> = BTreeMap::new();
    for row in rows {
        let key = &row.key;
        match (key.operative_clause_ordinal, key.one_word_clause_ordinal) {
            (Some(selector @ 1..=2), Some(one @ 1..=2)) if key.selector_swap_id.is_some() => {
                let index = (selector - 1) * 2 + one - 1;
                let cells = contexts.entry(&key.context_id).or_default();
                if cells[index].replace(row).is_some() {
                    return Err(format!("duplicate scoped cell for {}", key.context_id).into());
                }
            }
            (Some(_), Some(_)) if key.pair_axis.as_deref() == Some("selector-swap") => {
                if archive_selector_pairs
                    .entry(&row.panel.pair_id)
                    .or_default()[row.panel.side]
                    .replace(row)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate historical selector pair side for {}",
                        row.panel.pair_id
                    )
                    .into());
                }
            }
            (Some(_), Some(_)) if key.pair_axis.as_deref() == Some("clause-order-swap") => {}
            (None, None) if row.panel.contrast == 1 => {
                if post_b_pairs.entry(&row.panel.pair_id).or_default()[row.panel.side]
                    .replace(row)
                    .is_some()
                {
                    return Err(format!("duplicate post-B side for {}", row.panel.pair_id).into());
                }
            }
            (None, None) => {}
            _ => return Err(format!("malformed scoped key for {}", key.row_id).into()),
        }
    }
    if contexts.len() != split.scoped_contexts || post_b_pairs.len() != split.post_b_pairs {
        return Err("incomplete scoped interaction or post-B pair census".into());
    }

    let mut selector_ledger = Vec::new();
    let mut archive_selector_ledger = Vec::new();
    let mut direct_ledger = Vec::new();
    let mut interactions_ledger = Vec::new();
    let mut selectors = Totals::default();
    let mut archive_selectors = Totals::default();
    let mut post_b = Totals::default();
    let mut interactions = Totals::default();
    let mut direct_axes = BTreeMap::<&str, Totals>::new();
    let mut common_support_sum = [0; 2];
    let mut support_stop_copy_sum = [[0; 2]; 2];
    let archived_gpu_scores_checked =
        rows.len() + rows.iter().filter(|row| row.stages[1].is_some()).count();
    for (id, cells) in contexts {
        let [Some(s1o1), Some(s1o2), Some(s2o1), Some(s2o2)] = cells else {
            return Err(format!("incomplete four-cell context for {id}").into());
        };
        for (stop, copy) in [(s1o1, s2o1), (s2o2, s1o2)] {
            let selector_id = stop
                .key
                .selector_swap_id
                .as_deref()
                .ok_or("missing selector pair identifier")?;
            if copy.key.selector_swap_id.as_deref() != Some(selector_id) {
                return Err("paired selector swap ID differs".into());
            }
            let (record, stop_score, copy_score) = pair(geometry, selector_id, stop, copy, 0)?;
            common_support_sum[0] += record.common_support;
            for (sum, support) in support_stop_copy_sum[0]
                .iter_mut()
                .zip(record.unique_support_stop_copy)
            {
                *sum += support;
            }
            selectors.record_pair(&record.difference, stop_score, copy_score);
            serde_json::to_writer(&mut selector_ledger, &record)?;
            selector_ledger.push(b'\n');
        }
        let record = interaction(geometry, id, [s1o1, s1o2, s2o1, s2o2])?;
        interactions.record(&record.difference, record.copy_diagonal_minus_stop_diagonal);
        interactions.impossible_for_any_residual +=
            usize::from(record.positive_signed_scores_impossible_for_any_residual);
        serde_json::to_writer(&mut interactions_ledger, &record)?;
        interactions_ledger.push(b'\n');
    }
    for (id, cells) in post_b_pairs {
        let [Some(stop), Some(copy)] = cells else {
            return Err(format!("incomplete direct post-B pair for {id}").into());
        };
        if stop.key.pair_axis != copy.key.pair_axis {
            return Err(format!("mismatched direct pair provenance for {id}").into());
        }
        let (record, stop_score, copy_score) = pair(geometry, id, stop, copy, 1)?;
        common_support_sum[1] += record.common_support;
        for (sum, support) in support_stop_copy_sum[1]
            .iter_mut()
            .zip(record.unique_support_stop_copy)
        {
            *sum += support;
        }
        post_b.record_pair(&record.difference, stop_score, copy_score);
        direct_axes
            .entry(record.source_axis)
            .or_default()
            .record_pair(&record.difference, stop_score, copy_score);
        serde_json::to_writer(&mut direct_ledger, &record)?;
        direct_ledger.push(b'\n');
    }
    for (id, cells) in archive_selector_pairs {
        let [Some(stop), Some(copy)] = cells else {
            return Err(format!("incomplete historical selector pair for {id}").into());
        };
        if stop.key.one_word_clause_ordinal != copy.key.one_word_clause_ordinal
            || stop.key.operative_clause_ordinal == copy.key.operative_clause_ordinal
        {
            return Err(format!("historical selector pair changed clause order for {id}").into());
        }
        let (record, stop_score, copy_score) = pair(geometry, id, stop, copy, 0)?;
        archive_selectors.record_pair(&record.difference, stop_score, copy_score);
        serde_json::to_writer(&mut archive_selector_ledger, &record)?;
        archive_selector_ledger.push(b'\n');
    }
    for (name, bytes) in [
        ("selector-pairs", selector_ledger),
        ("post-b-pairs", direct_ledger),
        ("selector-contexts", interactions_ledger),
    ] {
        write_new(&dir.join(format!("{}-{name}.jsonl", split.name)), &bytes)?;
    }
    if !archive_selector_ledger.is_empty() {
        write_new(
            &dir.join(format!("{}-archive-selector-pairs.jsonl", split.name)),
            &archive_selector_ledger,
        )?;
    }
    Ok(serde_json::json!({
        "archived_gpu_row_stage_deltas_verified": archived_gpu_scores_checked,
        "selector_pairs_at_period": selectors,
        "archive_selector_pairs_at_period": (archive_selectors.records != 0).then_some(archive_selectors),
        "post_b_two_three_pairs": post_b,
        "post_b_two_three_by_provenance": direct_axes,
        "selector_by_clause_order_interaction": interactions,
        "per_axis_common_support_sum": common_support_sum,
        "per_axis_support_stop_copy_sum": support_stop_copy_sum
    }))
}

pub(super) fn run(prior: &Path, roots: &Path, components: &Path, dest: &Path) -> Result<()> {
    let roots = fs::read(roots)?;
    verify_sha(&roots, ROOTS_SHA, "packed frozen roots")?;
    if roots.len() != WORDS * 4 {
        return Err("unexpected root table width".into());
    }
    let roots: Vec<_> = roots
        .chunks_exact(4)
        .map(|root| u16::try_from(u32::from_le_bytes([root[0], root[1], root[2], root[3]])))
        .collect::<std::result::Result<_, _>>()?;
    let fit = prior.join("measured-v1/raw/fit");
    let tape: Vec<Slot> = rows(&pinned(&fit, "schedule.jsonl", SCHEDULE_SHA)?)?;
    let audit: Vec<Audit> = rows(&pinned(&fit, "joint-slots.jsonl", AUDIT_SHA)?)?;
    if tape.len() != SLOTS || audit.len() != SLOTS {
        return Err("invalid slots for feature inventory".into());
    }
    let mut touched = [[false; WORDS]; 2];
    for (index, record) in audit.iter().enumerate() {
        let (stage, _, _, _) = check_slot(index, &tape, record)?;
        for &address in &record.changed_coordinates {
            touched[stage][address] = true;
        }
    }
    let weights = [
        residual_from_state(components, "period.i16le", PERIOD_SHA)?,
        residual_from_state(components, "post-b.i16le", POST_B_SHA)?,
        residual_from_state(components, "full.i16le", JOINT_SHA)?,
    ];
    let geometry = Geometry {
        roots,
        weights,
        touched,
    };
    let train = load_panel(prior, &SPLITS[0], &geometry)?;
    let writes = verify_audit_addresses(&tape, &audit, &train)?;
    let held = load_panel(prior, &SPLITS[1], &geometry)?;
    fs::create_dir(dest)?;
    let receipt = serde_json::json!({
        "source": "independent reconstruction of original indexed feature supports and decomposition of archived CUDA score differences",
        "hardware": "Wick bookkeeping only",
        "fit_slots_with_verified_ordered_addresses": SLOTS,
        "train_matched_slot_writes": writes,
        "component_order_for_array_fields": ["period", "post_b", "full"],
        "axis_order_for_array_fields": ["selector_period", "direct_post_b"],
        "train": stats(&geometry, &SPLITS[0], &train, dest)?,
        "held": stats(&geometry, &SPLITS[1], &held, dest)?
    });
    write_new(
        &dest.join("geometry.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}
