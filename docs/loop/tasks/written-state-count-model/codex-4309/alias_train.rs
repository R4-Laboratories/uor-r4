//! Paired P5 training with a frozen, per-step distance-zero alias cost.
use crate::{
    engine::{EvalResult, MoveMode, P5Gpu, PickResult, StepResult},
    fail, measure,
    rows::{RowsGpu, Sample},
    sample::{self, Corpus},
    Result,
};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, Ptx, PushKernelArg};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Arc, time::Instant};
use uor_r4_discrete::p5_brier_core::{
    p5_orbit_tables, p5_schedule_block, P5BatchBrier, P5RowBrier, P5S1State, P5_BLOCKS,
    P5_CANDIDATES, P5_VOCAB,
};

const CUBIN: &[u8] = include_bytes!(env!("P5_ALIAS_ADJUST_CUBIN"));
const UNIT_Q32: u64 = 1 << 32;
const BATCHES: usize = 16;
const ROWS: usize = 4096;
const K: usize = 64;
const SEED: u64 = 4308;
const BASELINE_SHA: &str = "f681fad83f59376bf5b6a3598c1af0e89834bc6b98e6645c1b918544c990c0ac";

struct AdjustGpu {
    stream: Arc<CudaStream>,
    adjust: CudaFunction,
}

struct TableBand {
    words: Vec<u64>,
    protected: Vec<usize>,
    penalized_entries: usize,
}

fn literal(window: &[u16], position: usize) -> bool {
    let length = window.len();
    window[position - 1..=position] == window[length - 2..]
}

fn protected_position(row: &P5RowBrier, pick: &PickResult) -> Result<usize> {
    let position = usize::try_from(pick.position)?;
    if position == 0 {
        if row.window.len() >= 3
            || pick.distance != u32::MAX
            || pick.successor != 0
            || pick.loss_q32 != row.no_pick_brier_q32
        {
            return Err(fail("inconsistent incoming no-Pick"));
        }
        return Ok(0);
    }
    if position >= row.window.len().saturating_sub(1)
        || row.window[position + 1] != pick.successor
        || pick.distance > 128
        || pick.loss_q32 != row.brier_for_pick(Some((position, pick.distance)))
    {
        return Err(fail("incoming Pick disagrees with the ordinary loss table"));
    }
    Ok(
        if pick.distance == 0 && pick.successor == row.target && literal(&row.window, position) {
            position
        } else {
            0
        },
    )
}

fn protected_positions(batch: &P5BatchBrier, picks: &[PickResult]) -> Result<Vec<usize>> {
    if batch.rows.len() != picks.len() {
        return Err(fail("unexpected incoming Pick cardinality"));
    }
    batch
        .rows
        .iter()
        .zip(picks)
        .map(|(row, pick)| protected_position(row, pick))
        .collect()
}

fn penalized_position(row: &P5RowBrier, protected: usize, position: usize) -> bool {
    protected != 0 && position > protected && !literal(&row.window, position)
}

fn collision_count(
    batch: &P5BatchBrier,
    protected: &[usize],
    picks: &[PickResult],
) -> Result<usize> {
    if batch.rows.len() != protected.len() || batch.rows.len() != picks.len() {
        return Err(fail("unexpected candidate Pick cardinality"));
    }
    let mut count = 0;
    for ((row, &p), pick) in batch.rows.iter().zip(protected).zip(picks) {
        // Check the actual evaluated address and loss before indexing its pair.
        protected_position(row, pick)?;
        if p != 0 && (pick.distance != 0 || pick.position < p as u32) {
            return Err(fail(
                "a protected literal pair ceased to be a zero-distance candidate",
            ));
        }
        count +=
            usize::from(pick.distance == 0 && penalized_position(row, p, pick.position as usize));
    }
    Ok(count)
}

fn row_ceiling(vocab: usize, rows: usize) -> Result<u64> {
    let ceiling = (vocab as u64)
        .checked_sub(1)
        .and_then(|n| n.checked_mul(UNIT_Q32))
        .ok_or_else(|| fail("training row loss bound overflow"))?;
    ceiling
        .checked_mul(rows as u64)
        .ok_or_else(|| fail("training batch loss bound overflow"))?;
    Ok(ceiling)
}

fn expected_word(brier_q32: u64, affected: bool, bound: u64) -> Result<u64> {
    brier_q32
        .checked_add(if affected { UNIT_Q32 } else { 0 })
        .filter(|&word| word <= bound)
        .ok_or_else(|| fail("training alias cost exceeds the scorer loss bound"))
}

impl TableBand {
    // The first band of each row is d=0. No-Pick and every other distance class
    // stay in their original buffers. Swapping twice restores the exact bits.
    fn swap_into(&mut self, batch: &mut P5BatchBrier) {
        let mut offset = 0;
        for row in &mut batch.rows {
            let positions = row.window.len().saturating_sub(2);
            row.pick_brier_q32[..positions]
                .swap_with_slice(&mut self.words[offset..offset + positions]);
            offset += positions;
        }
    }
}

impl AdjustGpu {
    fn new() -> Result<Self> {
        let context = CudaContext::new(0)?;
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        Ok(Self {
            stream: context.default_stream(),
            adjust: module.load_function("p5_alias_adjust")?,
        })
    }

    fn band(
        &self,
        batch: &P5BatchBrier,
        incoming: &[PickResult],
        vocab: usize,
    ) -> Result<TableBand> {
        batch.validate_for_vocab(vocab)?;
        let protected = protected_positions(batch, incoming)?;
        let bound = row_ceiling(vocab, batch.rows.len())?;
        let mut tokens = Vec::new();
        let mut windows = vec![0u32];
        let mut targets = Vec::with_capacity(batch.rows.len());
        let mut offsets = Vec::with_capacity(batch.rows.len());
        let mut words = Vec::new();
        for row in &batch.rows {
            tokens.extend_from_slice(&row.window);
            windows.push(u32::try_from(tokens.len())?);
            targets.push(row.target);
            offsets.push(u32::try_from(words.len())?);
            words.extend_from_slice(&row.pick_brier_q32[..row.window.len().saturating_sub(2)]);
        }
        // A batch with no historical pair has no table band to adjust.
        if words.is_empty() {
            return Ok(TableBand {
                words,
                protected,
                penalized_entries: 0,
            });
        }
        u32::try_from(words.len())?;
        let picks: Vec<u32> = incoming
            .iter()
            .flat_map(|pick| [pick.position, pick.distance])
            .collect();
        let d_windows = self.stream.clone_htod(&windows)?;
        let d_offsets = self.stream.clone_htod(&offsets)?;
        let d_tokens = self.stream.clone_htod(&tokens)?;
        let d_targets = self.stream.clone_htod(&targets)?;
        let d_picks = self.stream.clone_htod(&picks)?;
        let mut d_words = self.stream.clone_htod(&words)?;
        let mut d_protected = self.stream.alloc_zeros::<u32>(batch.rows.len())?;
        let mut d_status = self.stream.alloc_zeros::<u32>(1)?;
        let mut launch = self.stream.launch_builder(&self.adjust);
        launch
            .arg(&d_windows)
            .arg(&d_offsets)
            .arg(&d_tokens)
            .arg(&d_targets)
            .arg(&d_picks)
            .arg(&bound)
            .arg(&mut d_words)
            .arg(&mut d_protected)
            .arg(&mut d_status);
        // SAFETY: The validated row lengths and flattened offsets bound each
        // disjoint class-zero band. One block owns a row; a thread owns a word.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (u32::try_from(batch.rows.len())?, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        if self.stream.clone_dtoh(&d_status)? != [0] {
            return Err(fail(
                "CUDA training alias cost exceeds the scorer loss bound",
            ));
        }
        if !self
            .stream
            .clone_dtoh(&d_protected)?
            .iter()
            .map(|&pos| pos as usize)
            .eq(protected.iter().copied())
        {
            return Err(fail(
                "CUDA alias protection mask disagrees with exact TRAIN labels",
            ));
        }
        let adjusted = self.stream.clone_dtoh(&d_words)?;
        let mut penalized_entries = 0;
        for (i, row) in batch.rows.iter().enumerate() {
            let begin = offsets[i] as usize;
            for (word, &loss) in row.pick_brier_q32[..row.window.len().saturating_sub(2)]
                .iter()
                .enumerate()
            {
                let affected = penalized_position(row, protected[i], word + 1);
                if adjusted[begin + word] != expected_word(loss, affected, bound)? {
                    return Err(fail(format!(
                        "CUDA alias table mismatch at row {i}, position {}",
                        word + 1
                    )));
                }
                penalized_entries += usize::from(affected);
            }
        }
        Ok(TableBand {
            words: adjusted,
            protected,
            penalized_entries,
        })
    }
}

fn objective_after(brier_q32: u64, collisions: usize) -> Result<u64> {
    (collisions as u64)
        .checked_mul(UNIT_Q32)
        .and_then(|cost| brier_q32.checked_add(cost))
        .ok_or_else(|| fail("alias-adjusted Brier overflow"))
}

fn candidate_exposure(standard: &StepResult, penalty: &StepResult) -> Result<Value> {
    let mut lanes_with_collisions = 0u64;
    let mut lane_row_events = 0u64;
    let mut max_collisions_on_lane = 0u64;
    for (ordinary, adjusted) in standard.scores.iter().zip(&penalty.scores) {
        if ordinary.coord != adjusted.coord || ordinary.losses[0] != adjusted.losses[0] {
            return Err(fail("same-state coordinates or incoming losses disagree"));
        }
        for (&raw, &priced) in ordinary.losses[1..].iter().zip(&adjusted.losses[1..]) {
            let delta = priced
                .checked_sub(raw)
                .filter(|delta| delta % UNIT_Q32 == 0)
                .ok_or_else(|| {
                    fail("same-state candidate cost is not a nonnegative whole Q32 unit")
                })?;
            let collisions = delta / UNIT_Q32;
            lanes_with_collisions += u64::from(collisions != 0);
            lane_row_events += collisions;
            max_collisions_on_lane = max_collisions_on_lane.max(collisions);
        }
    }
    Ok(
        json!({"lanes_with_collisions":lanes_with_collisions,"lane_row_events":lane_row_events,
        "max_collisions_on_lane":max_collisions_on_lane}),
    )
}

fn changed_roots(before: &P5S1State, after: &P5S1State) -> usize {
    before
        .roots
        .iter()
        .zip(&after.roots)
        .filter(|(a, b)| a != b)
        .count()
}

fn step_receipt(
    step: &StepResult,
    incoming: &EvalResult,
    outgoing: &EvalResult,
    protected: &[usize],
    collisions: usize,
    own_start: &P5S1State,
    committed: &P5S1State,
) -> Value {
    let record = &step.record;
    let moves: Vec<_> = record.commit_coords.iter().zip(&record.commit_lanes)
        .map(|(&coord, &lane)| json!({"coordinate":coord,"lane":lane,"before_root":own_start.roots[coord as usize],"after_root":committed.roots[coord as usize]}))
        .collect();
    let scored_destinations: u64 = step
        .evaluated_destinations
        .iter()
        .zip(&step.affected_rows)
        .filter(|&(_, &rows)| rows != 0)
        .map(|(&destinations, _)| u64::from(destinations))
        .sum();
    let destination_rows: u64 = step
        .evaluated_destinations
        .iter()
        .zip(&step.affected_rows)
        .map(|(&destinations, &rows)| u64::from(destinations) * u64::from(rows))
        .sum();
    json!({
        "protected_rows":protected.iter().filter(|&&p| p != 0).count(),
        "committed_alias_collisions":collisions,
        "ordinary_before_q32":incoming.loss_q32,"ordinary_after_q32":outgoing.loss_q32,
        "objective_before_q32":record.l_before,"objective_after_q32":record.committed_loss,
        "objective_best_single_q32":record.l_best,"objective_joint_q32":record.l_joint,
        "joint_committed":record.joint,"improving_coordinates":record.s_len,
        "logical_proposals":step.evaluated_destinations.iter().sum::<u32>(),
        "dispatched_proposals":scored_destinations,"destination_row_evaluations":destination_rows,
        "committed_moves":moves,"epoch":record.epoch_after,
        "scorer_us":step.score_micros,"step_device_us":step.device_micros,
        "plan_us":step.plan_micros,"upload_allocate_us":step.upload_allocate_micros,
        "readback_validate_us":step.readback_validate_micros,
    })
}

/// Both arms begin at the same roots and see the same registered TRAIN draw.
pub fn train(
    gpu: &P5Gpu,
    reader: &RowsGpu,
    base: &P5S1State,
    corpus: &Corpus,
    out: &Path,
) -> Result<(P5S1State, P5S1State)> {
    if (corpus.record_count(), corpus.target_count()) != (4060, 15947584)
        || (base.vocab, base.blocks) != (P5_VOCAB, P5_BLOCKS)
    {
        return Err(fail("wrong registered R census or S1 geometry"));
    }
    let states = out.join("states");
    fs::create_dir_all(&states)?;
    let start = Instant::now();
    let adjust = AdjustGpu::new()?;
    let schedule = corpus.train_sample(BATCHES, ROWS, SEED)?;
    let mut standard = base.clone();
    let mut penalty = base.clone();
    let mut receipts = Vec::with_capacity(BATCHES);
    for (b, pos) in schedule.iter().enumerate() {
        let prep = Instant::now();
        let samples: Vec<_> = pos
            .iter()
            .map(|p| Sample {
                record: p.record,
                target_at: p.target_at,
            })
            .collect();
        let mut prepared = reader.prepare_for_training(&corpus.ids, &samples)?;
        if prepared.batch.rows.len() != ROWS
            || prepared.oracle_rows != 0
            || !prepared.oracle_time.is_zero()
        {
            return Err(fail(
                "wrong training batch cardinality or unexpected training oracle",
            ));
        }
        let prep_us = prep.elapsed().as_micros();
        let batch = &mut prepared.batch;
        let coords = p5_schedule_block(base.roots.len(), K, b as u64)?;
        let standard_start = standard.clone();
        let penalty_start = penalty.clone();
        let same_incoming_roots = standard_start.roots == penalty_start.roots;
        let before_standard = gpu.evaluate(&standard_start, batch)?;
        let before_penalty = if same_incoming_roots {
            None
        } else {
            Some(gpu.evaluate(&penalty_start, batch)?)
        };
        let before_penalty = before_penalty.as_ref().unwrap_or(&before_standard);
        let standard_mask = protected_positions(batch, &before_standard.rows)?;
        let adjust_started = Instant::now();
        let mut band = adjust.band(batch, &before_penalty.rows, base.vocab)?;
        let adjust_us = adjust_started.elapsed().as_micros();
        let (step_standard, step_penalty) = if b % 2 == 0 {
            let step_standard = gpu.step(&mut standard, batch, &coords, MoveMode::AllGenerators)?;
            let step_penalty = penalty_step(gpu, &mut penalty, batch, &coords, &mut band)?;
            (step_standard, step_penalty)
        } else {
            let step_penalty = penalty_step(gpu, &mut penalty, batch, &coords, &mut band)?;
            let step_standard = gpu.step(&mut standard, batch, &coords, MoveMode::AllGenerators)?;
            (step_standard, step_penalty)
        };
        let after_standard = gpu.evaluate(&standard, batch)?;
        let after_penalty = gpu.evaluate(&penalty, batch)?;
        let standard_collisions = collision_count(batch, &standard_mask, &after_standard.rows)?;
        let penalty_collisions = collision_count(batch, &band.protected, &after_penalty.rows)?;
        if step_standard.affected_rows != step_penalty.affected_rows
            || step_standard.evaluated_destinations != vec![(P5_CANDIDATES - 1) as u32; K]
            || step_standard.evaluated_destinations != step_penalty.evaluated_destinations
            || step_standard.record.l_before != before_standard.loss_q32
            || step_standard.record.committed_loss != after_standard.loss_q32
            || step_penalty.record.l_before != before_penalty.loss_q32
            || step_penalty.record.committed_loss
                != objective_after(after_penalty.loss_q32, penalty_collisions)?
        {
            return Err(fail(format!(
                "paired work or raw GPU objective disagrees at batch {b}"
            )));
        }
        let shared_state_candidate_exposure = if same_incoming_roots {
            Some(candidate_exposure(&step_standard, &step_penalty)?)
        } else {
            None
        };
        let mut standard_log = step_receipt(
            &step_standard,
            &before_standard,
            &after_standard,
            &standard_mask,
            standard_collisions,
            &standard_start,
            &standard,
        );
        let mut penalty_log = step_receipt(
            &step_penalty,
            before_penalty,
            &after_penalty,
            &band.protected,
            penalty_collisions,
            &penalty_start,
            &penalty,
        );
        for (name, state, log) in [
            ("standard", &mut standard, &mut standard_log),
            ("penalty", &mut penalty, &mut penalty_log),
        ] {
            let (saved, checkpoint_us, sha) =
                measure::checkpoint(state, &states.join(format!("{name}-{b:02}.s1")))?;
            *state = saved;
            log["checkpoint_us"] = json!(checkpoint_us);
            log["checkpoint_sha256"] = json!(sha);
        }
        let matching_moves = coords
            .iter()
            .filter(|&&coordinate| {
                let i = coordinate as usize;
                standard_start.roots[i] != standard.roots[i]
                    && standard_start.roots[i] == penalty_start.roots[i]
                    && standard.roots[i] == penalty.roots[i]
            })
            .count();
        let receipt = json!({"batch":b,"sample_sha256":sample::sample_digest(corpus,pos),
            "rows":pos.len(),"coordinates":coords,"standard_first":b%2==0,
            "row_prep_us":prep_us,"row_gather_upload_us":prepared.prepare_time.as_micros(),
            "row_gpu_us":prepared.gpu_time.as_micros(),"row_materialize_us":prepared.materialize_time.as_micros(),
            "adjust_and_verify_us":adjust_us,"penalized_distance_zero_entries":band.penalized_entries,
            "shared_state_candidate_exposure":shared_state_candidate_exposure,
            "coordinates_with_identical_committed_moves":matching_moves,
            "different_roots_after_batch":changed_roots(&standard,&penalty),
            "standard":standard_log,"penalty":penalty_log});
        measure::json_file(&out.join(format!("alias-train-{b:02}.json")), &receipt)?;
        receipts.push(receipt);
    }
    if receipts[BATCHES - 1]["standard"]["checkpoint_sha256"] != BASELINE_SHA {
        return Err(fail(
            "ordinary Brier arm did not reproduce the registered 4308 checkpoint",
        ));
    }
    measure::json_file(
        &out.join("alias-training.json"),
        &json!({
            "schema":"uor-r4.p5-train-alias/1","train_sha256":sample::TRAIN_SHA,"sample_seed":SEED,
            "p4_sha256":reader.artifact_sha256,"rows_cubin_sha256":reader.cubin_sha256,
            "adjust_cubin_sha256":hex::encode(Sha256::digest(CUBIN)),
            "initial_s1_sha256":hex::encode(Sha256::digest(base.encode_bytes()?)),
            "penalty_q32":UNIT_Q32,"batches":BATCHES,"rows_per_arm":BATCHES*ROWS,"coordinates_per_batch":K,
            "standard_changed_roots":changed_roots(base,&standard),"penalty_changed_roots":changed_roots(base,&penalty),
            "final_arm_root_differences":changed_roots(&standard,&penalty),"wall_us":start.elapsed().as_micros(),
            "paired_steps":receipts,
        }),
    )?;
    Ok((standard, penalty))
}

fn penalty_step(
    gpu: &P5Gpu,
    state: &mut P5S1State,
    batch: &mut P5BatchBrier,
    coords: &[u32],
    band: &mut TableBand,
) -> Result<StepResult> {
    band.swap_into(batch);
    let step = gpu.step(state, batch, coords, MoveMode::AllGenerators);
    band.swap_into(batch);
    step
}

fn planted_row(window: Vec<u16>, target: u16) -> P5RowBrier {
    let positions = window.len().saturating_sub(2);
    P5RowBrier {
        window,
        target,
        no_pick_brier_q32: 400,
        pick_brier_q32: vec![300; positions * 4],
    }
}

fn planted() -> Result<(P5S1State, P5BatchBrier)> {
    let (moves, codes) = p5_orbit_tables();
    let (literal_root, alias_root) = (0..codes.len())
        .flat_map(|a| (0..codes.len()).map(move |b| (a, b)))
        .find(|&(a, b)| codes[a] != codes[b] && moves[b][1..].contains(&(a as u8)))
        .ok_or_else(|| fail("finite fixture has no reachable alias root"))?;
    let mut used_codes = vec![codes[literal_root], codes[alias_root]];
    let mut roots = vec![literal_root as u8; 7 * P5_BLOCKS];
    for token in [1, 3, 6] {
        let root = codes
            .iter()
            .position(|c| !used_codes.contains(c))
            .ok_or_else(|| fail("finite fixture requires three unrelated root codes"))?;
        used_codes.push(codes[root]);
        roots[token * P5_BLOCKS..(token + 1) * P5_BLOCKS].fill(root as u8);
    }
    roots[4 * P5_BLOCKS] = alias_root as u8;
    roots[5 * P5_BLOCKS] = alias_root as u8;
    let state = P5S1State::from_roots(7, P5_BLOCKS, 7, [42; 32], roots)?;
    // p=1 is literal and correct; p=4 and p=7 are later aliases with matching
    // and different successors respectively. Ordinary Brier rewards either.
    let mut positive = planted_row(vec![1, 2, 3, 1, 4, 3, 1, 5, 6, 1, 2], 3);
    positive.pick_brier_q32[0] = 100;
    positive.pick_brier_q32[3] = 10;
    positive.pick_brier_q32[6] = 5;
    let mut wrong_target = positive.clone();
    wrong_target.target = 6;
    let earlier_alias = planted_row(vec![1, 4, 3, 1, 2, 3, 1, 2], 3);
    let positive_distance = planted_row(vec![3, 6, 1, 2], 3);
    let no_pick = planted_row(vec![1, 2], 3);
    let duplicate_literal = planted_row(vec![1, 2, 3, 1, 2, 3, 1, 4, 3, 1, 2], 3);
    Ok((
        state,
        P5BatchBrier::new(vec![
            positive,
            wrong_target,
            earlier_alias,
            positive_distance,
            no_pick,
            duplicate_literal,
        ])?,
    ))
}

fn finite_step(
    gpu: &P5Gpu,
    start: &P5S1State,
    batch: &P5BatchBrier,
    coords: &[u32],
) -> Result<(P5S1State, StepResult)> {
    use uor_r4_discrete::p5_brier_core::{p5_bvgk_step_cpu, p5_score_block_cpu};
    let mut reference = start.clone();
    let expected = p5_bvgk_step_cpu(&mut reference, batch, coords)?;
    let mut device = start.clone();
    let step = gpu.step(&mut device, batch, coords, MoveMode::AllGenerators)?;
    if step.scores != p5_score_block_cpu(start, batch, coords)?
        || step.record != expected
        || device != reference
    {
        return Err(fail(
            "finite adjusted lane, commit or native root bytes disagree",
        ));
    }
    Ok((device, step))
}

pub fn finite(gpu: &P5Gpu) -> Result<()> {
    let adjustment = AdjustGpu::new()?;
    let (start, original) = planted()?;
    let initial = gpu.evaluate(&start, &original)?;
    let mut band = adjustment.band(&original, &initial.rows, start.vocab)?;
    let mut adjusted = original.clone();
    band.swap_into(&mut adjusted);
    if band.protected != [1, 0, 4, 0, 0, 4]
        || initial.rows[3].distance == 0
        || initial.rows[4].position != 0
        || adjusted.rows[0].brier_for_pick(Some((1, 0))) != 100
        || adjusted.rows[0].brier_for_pick(Some((4, 0))) != UNIT_Q32 + 10
        || adjusted.rows[0].brier_for_pick(Some((7, 0))) != UNIT_Q32 + 5
        || adjusted.rows[0].brier_for_pick(Some((4, 1))) != 300
        || adjusted.rows[1] != original.rows[1]
        || adjusted.rows[2].brier_for_pick(Some((1, 0))) != 300
        || adjusted.rows[3..5] != original.rows[3..5]
        || adjusted.rows[5].brier_for_pick(Some((1, 0))) != 300
        || adjusted.rows[5].brier_for_pick(Some((4, 0))) != 300
        || adjusted.rows[5].brier_for_pick(Some((7, 0))) != UNIT_Q32 + 300
    {
        return Err(fail(
            "finite exact, same-successor, earlier, or positive-distance control disagrees",
        ));
    }
    let protected_only = P5BatchBrier::new(vec![original.rows[0].clone()])?;
    let adjusted_only = P5BatchBrier::new(vec![adjusted.rows[0].clone()])?;
    for (coord, position, successor, loss) in [(16, 4, 3, 10), (20, 7, 6, 5)] {
        let (ordinary, ordinary_result) = finite_step(gpu, &start, &protected_only, &[coord])?;
        let (penalized, penalty_result) = finite_step(gpu, &start, &adjusted_only, &[coord])?;
        let picked = &gpu.evaluate(&ordinary, &protected_only)?.rows[0];
        let already_aliased =
            adjustment.band(&protected_only, std::slice::from_ref(picked), start.vocab)?;
        if picked.position != position
            || picked.distance != 0
            || picked.successor != successor
            || ordinary_result.record.committed_loss != loss
            || penalized != start
            || penalty_result.record.committed_loss != 100
            || already_aliased.protected != [0]
            || already_aliased.penalized_entries != 0
        {
            return Err(fail(
                "finite later-alias rejection or old-alias control disagrees",
            ));
        }
    }

    // Preserve useful nonliteral reads on unprotected rows. Neither move alone
    // is as useful as both in this fixture, so the adjusted joint scan runs.
    let mut joint = protected_only.clone();
    for token in [4, 5] {
        let mut gain = planted_row(vec![1, token, 6, 1, 2], 6);
        let positions = gain.window.len() - 2;
        gain.pick_brier_q32[..positions].fill(0);
        gain.pick_brier_q32[positions..].fill(2 * UNIT_Q32);
        joint.rows.push(gain);
    }
    let joint_incoming = gpu.evaluate(&start, &joint)?;
    let mut joint_band = adjustment.band(&joint, &joint_incoming.rows, start.vocab)?;
    if joint_band.protected != [1, 0, 0] {
        return Err(fail(
            "finite joint controls must start with positive-distance nonliteral reads",
        ));
    }
    joint_band.swap_into(&mut joint);
    let (committed, step) = finite_step(gpu, &start, &joint, &[16, 20])?;
    joint_band.swap_into(&mut joint);
    let joint_outgoing = gpu.evaluate(&committed, &joint)?;
    if !step.record.joint
        || step.record.committed_loss != UNIT_Q32 + 5
        || joint_outgoing.loss_q32 != 5
        || collision_count(&joint, &joint_band.protected, &joint_outgoing.rows)? != 1
    {
        return Err(fail(
            "finite joint scan did not price the committed alias exactly once",
        ));
    }

    let mut overflow = protected_only;
    let bound = row_ceiling(start.vocab, overflow.rows.len())?;
    overflow.rows[0].pick_brier_q32[3] = bound;
    let before = gpu.evaluate(&start, &overflow)?;
    let error = adjustment
        .band(&overflow, &before.rows, start.vocab)
        .err()
        .ok_or_else(|| fail("finite GPU accepted an out-of-bound alias loss"))?;
    if !error
        .to_string()
        .contains("CUDA training alias cost exceeds")
    {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uor_r4_discrete::p5_brier_core::pick_with_virtual_move;

    #[test]
    fn exact_pairs_and_later_aliases_are_distinct_even_with_the_same_successor() {
        let (state, batch) = planted().unwrap();
        let (moves, codes) = p5_orbit_tables();
        let picks: Vec<_> = batch
            .rows
            .iter()
            .map(|row| {
                let picked =
                    pick_with_virtual_move(&state.roots, &moves, &codes, &row.window, None);
                PickResult {
                    position: picked.map_or(0, |(p, _)| p as u32),
                    distance: picked.map_or(u32::MAX, |(_, d)| d),
                    successor: picked.map_or(0, |(p, _)| row.window[p + 1]),
                    loss_q32: row.brier_for_pick(picked),
                }
            })
            .collect();
        let mask = protected_positions(&batch, &picks).unwrap();
        assert_eq!(mask, [1, 0, 4, 0, 0, 4]);
        let affected = |r: usize, p| penalized_position(&batch.rows[r], mask[r], p);
        assert!(!affected(0, 1));
        assert!(affected(0, 4));
        assert!(affected(0, 7));
        assert!(!affected(1, 7));
        assert!(!affected(2, 1));
        assert!(!affected(5, 1));
        assert!(!affected(5, 4));
        assert!(affected(5, 7));
    }

    #[test]
    fn loss_bounds_refuse_overflow_and_keep_boundary_value() {
        let bound = row_ceiling(P5_VOCAB, ROWS).unwrap();
        assert_eq!(expected_word(bound - UNIT_Q32, true, bound).unwrap(), bound);
        assert!(expected_word(bound - UNIT_Q32 + 1, true, bound).is_err());
        assert!(expected_word(u64::MAX, true, u64::MAX).is_err());
        assert!(objective_after(u64::MAX, 1).is_err());
        assert!(row_ceiling(1 << 28, 1 << 8).is_err());
    }
}
