//! CUDA implementation of the merged P5 integer Pick and BVG-K transition.
use crate::{fail, Result};
use cudarc::driver::{
    sys::CUevent_flags, CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
    Ptx, PushKernelArg,
};
use std::sync::Arc;
use std::time::Instant;
use uor_r4_discrete::p5_brier_core::{
    p5_bvgk_step_cpu, p5_orbit_tables, p5_score_block_cpu, pick_with_virtual_move, P5BatchBrier,
    P5BlockRecord, P5CoordScore, P5RowBrier, P5S1State, P5_BLOCKS, P5_CANDIDATES, P5_MAX_K,
};

const CUBIN: &[u8] = include_bytes!(env!("P5_PILOT_CUBIN"));
const BLOCK_THREADS: u32 = 128;
const WARPS_PER_BLOCK: u32 = 4;
const ROWS_PER_SCORE_JOB: u32 = 32;
const JOB_WORDS: usize = 3;
const TRANSITION_WORDS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveMode {
    AllGenerators,
    DistinctDestinations,
}

#[derive(Debug)]
pub struct StepResult {
    pub record: P5BlockRecord,
    pub scores: Vec<P5CoordScore>,
    /// Window membership counts, in the order of the coordinate block.
    pub affected_rows: Vec<u32>,
    /// Enumerated move representatives per coordinate, excluding the identity lane.
    /// Coordinates with empty membership reuse the baseline for every representative.
    pub evaluated_destinations: Vec<u32>,
    /// CUDA event time for baseline, scoring, choose, joint verification and apply.
    pub device_micros: u64,
    /// CUDA event time for the proposal-scoring kernel alone.
    pub score_micros: u64,
    pub plan_micros: u64,
    pub upload_allocate_micros: u64,
    pub readback_validate_micros: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PickResult {
    /// Position of the historical pair's second token. Zero means no Pick.
    pub position: u32,
    /// UINT32_MAX means no Pick.
    pub distance: u32,
    pub successor: u16,
    pub loss_q32: u64,
}

#[derive(Debug)]
pub struct EvalResult {
    pub loss_q32: u64,
    pub rows: Vec<PickResult>,
    pub device_micros: u64,
}

pub struct P5Gpu {
    stream: Arc<CudaStream>,
    baseline: CudaFunction,
    score: CudaFunction,
    choose: CudaFunction,
    joint: CudaFunction,
    apply: CudaFunction,
    orbit_moves: [[u8; P5_CANDIDATES]; 16],
    device_moves: CudaSlice<u8>,
    device_distances: CudaSlice<u8>,
}

struct DeviceBatch {
    rows: u32,
    tokens: CudaSlice<u16>,
    windows: CudaSlice<u32>,
    loss_offsets: CudaSlice<u32>,
    losses: CudaSlice<u64>,
}

struct PackedBatch {
    tokens: Vec<u16>,
    windows: Vec<u32>,
    loss_offsets: Vec<u32>,
    losses: Vec<u64>,
}

struct Baseline {
    distance: CudaSlice<u8>,
    picks: CudaSlice<u32>,
    row_losses: CudaSlice<u64>,
    total: CudaSlice<u64>,
}

struct Proposals {
    jobs: Vec<u32>,
    row_indices: Vec<u32>,
    representative: Vec<u8>,
    representative_counts: Vec<u32>,
    affected_counts: Vec<u32>,
}

fn validate_batch(state: &P5S1State, batch: &P5BatchBrier) -> Result<()> {
    state.validate()?;
    if state.blocks != P5_BLOCKS {
        return Err(fail("P5 CUDA Pick requires four blocks per token"));
    }
    batch.validate_for_vocab(state.vocab)?;
    let row_max = ((state.vocab - 1) as u64) << 32;
    row_max
        .checked_mul(batch.rows.len() as u64)
        .ok_or_else(|| fail("P5 batch can overflow full-vocabulary Brier accumulation"))?;
    for (index, row) in batch.rows.iter().enumerate() {
        // The longer inner loop and flattened u32 offsets remain in bounds.
        if row.window.len() > usize::from(u16::MAX)
            || row.no_pick_brier_q32 > row_max
            || row.pick_brier_q32.iter().any(|&loss| loss > row_max)
        {
            return Err(fail(format!(
                "P5 CUDA row {index} exceeds window or Brier bounds"
            )));
        }
    }
    Ok(())
}

fn validate_coords(state: &P5S1State, coords: &[u32]) -> Result<()> {
    if coords.is_empty() || coords.len() > P5_MAX_K {
        return Err(fail("P5 coordinate block must hold 1 to 64 coordinates"));
    }
    if coords
        .iter()
        .enumerate()
        .any(|(ord, &coord)| coord as usize >= state.roots.len() || coords[..ord].contains(&coord))
    {
        return Err(fail(
            "P5 coordinate block contains an out-of-range or duplicated coordinate",
        ));
    }
    Ok(())
}

impl Proposals {
    fn new(
        state: &P5S1State,
        batch: &P5BatchBrier,
        coordinates: &[u32],
        moves: &[[u8; P5_CANDIDATES]; 16],
        mode: MoveMode,
    ) -> Self {
        let mut row_indices_by_token = vec![Vec::new(); state.vocab];
        let mut tracked_tokens = vec![false; state.vocab];
        for &coordinate in coordinates {
            tracked_tokens[coordinate as usize / P5_BLOCKS] = true;
        }
        for (row_index, row) in batch.rows.iter().enumerate() {
            let row_index = row_index as u32;
            for &token in &row.window {
                let token = usize::from(token);
                if tracked_tokens[token] {
                    let indices = &mut row_indices_by_token[token];
                    if indices.last() != Some(&row_index) {
                        indices.push(row_index);
                    }
                }
            }
        }

        let mut row_indices = Vec::new();
        let mut ranges = vec![(0u32, 0u32); state.vocab];
        for (range, indices) in ranges.iter_mut().zip(row_indices_by_token) {
            let start = row_indices.len() as u32;
            row_indices.extend(indices);
            *range = (start, row_indices.len() as u32);
        }
        let mut result = Self {
            jobs: Vec::new(),
            row_indices,
            representative: Vec::with_capacity(coordinates.len() * P5_CANDIDATES),
            representative_counts: Vec::with_capacity(coordinates.len()),
            affected_counts: Vec::with_capacity(coordinates.len()),
        };
        for (ord, &coordinate) in coordinates.iter().enumerate() {
            let coord = coordinate as usize;
            let root = usize::from(state.roots[coord]);
            let (start, end) = ranges[coord / P5_BLOCKS];
            result.affected_counts.push(end - start);

            let mut first_lane_for_root = [None; 16];
            first_lane_for_root[root] = Some(0);
            let mut representatives = 0;
            for (lane, &destination) in moves[root].iter().enumerate() {
                let first_lane =
                    first_lane_for_root[usize::from(destination)].get_or_insert(lane as u8);
                let representative = match mode {
                    MoveMode::AllGenerators => lane as u8,
                    MoveMode::DistinctDestinations => *first_lane,
                };
                result.representative.push(representative);
                if lane == 0 || representative != lane as u8 {
                    continue;
                }
                representatives += 1;
                let lane_index = (ord * P5_CANDIDATES + lane) as u32;
                for tile in (start..end).step_by(ROWS_PER_SCORE_JOB as usize) {
                    result
                        .jobs
                        .extend([lane_index, tile, end.min(tile + ROWS_PER_SCORE_JOB)]);
                }
            }
            result.representative_counts.push(representatives);
        }
        result
    }
}

fn launch_config(blocks: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

// CUDA events use a floating-point duration. Model values and decisions use integers.
fn elapsed_micros(start: &CudaEvent, end: &CudaEvent) -> Result<u64> {
    Ok((start.elapsed_ms(end)? * 1000.0).round() as u64)
}

impl PackedBatch {
    fn new(batch: &P5BatchBrier) -> Result<Self> {
        let mut windows = Vec::with_capacity(batch.rows.len() + 1);
        let mut loss_offsets = Vec::with_capacity(batch.rows.len());
        let mut tokens = Vec::new();
        let mut losses = Vec::new();
        windows.push(0);
        for row in &batch.rows {
            tokens.extend_from_slice(&row.window);
            windows.push(u32::try_from(tokens.len())?);
            loss_offsets.push(u32::try_from(losses.len())?);
            losses.push(row.no_pick_brier_q32);
            losses.extend_from_slice(&row.pick_brier_q32);
        }
        u32::try_from(losses.len())?;
        Ok(Self {
            tokens,
            windows,
            loss_offsets,
            losses,
        })
    }
}

impl P5Gpu {
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(ordinal)?;
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        let (orbit_moves, codes) = p5_orbit_tables();
        let distances: Vec<u8> = codes
            .iter()
            .flat_map(|a| codes.iter().map(move |b| (a ^ b).count_ones() as u8))
            .collect();
        Ok(Self {
            baseline: module.load_function("p5_baseline")?,
            score: module.load_function("p5_score")?,
            choose: module.load_function("p5_choose")?,
            joint: module.load_function("p5_joint")?,
            apply: module.load_function("p5_apply")?,
            device_moves: stream.clone_htod(orbit_moves.as_flattened())?,
            device_distances: stream.clone_htod(&distances)?,
            orbit_moves,
            stream,
        })
    }

    fn timed_event(&self) -> Result<CudaEvent> {
        Ok(self
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?)
    }

    fn upload_batch(&self, packed: &PackedBatch) -> Result<DeviceBatch> {
        Ok(DeviceBatch {
            rows: packed.loss_offsets.len() as u32,
            tokens: self.stream.clone_htod(&packed.tokens)?,
            windows: self.stream.clone_htod(&packed.windows)?,
            loss_offsets: self.stream.clone_htod(&packed.loss_offsets)?,
            losses: self.stream.clone_htod(&packed.losses)?,
        })
    }

    fn allocate_baseline(&self, batch: &DeviceBatch) -> Result<Baseline> {
        Ok(Baseline {
            distance: self.stream.alloc_zeros(batch.tokens.len())?,
            picks: self.stream.alloc_zeros(batch.rows as usize * 2)?,
            row_losses: self.stream.alloc_zeros(batch.rows as usize)?,
            total: self.stream.alloc_zeros(1)?,
        })
    }

    fn scan(&self, roots: &CudaSlice<u8>, batch: &DeviceBatch, base: &mut Baseline) -> Result<()> {
        let mut launch = self.stream.launch_builder(&self.baseline);
        launch
            .arg(&batch.rows)
            .arg(roots)
            .arg(&self.device_distances)
            .arg(&batch.windows)
            .arg(&batch.tokens)
            .arg(&batch.loss_offsets)
            .arg(&batch.losses)
            .arg(&mut base.distance)
            .arg(&mut base.picks)
            .arg(&mut base.row_losses)
            .arg(&mut base.total);
        // SAFETY: Row indices, flattened offsets, and data widths match p5_baseline.
        unsafe {
            launch.launch(launch_config(
                batch.rows.div_ceil(WARPS_PER_BLOCK),
                BLOCK_THREADS,
            ))?;
        }
        Ok(())
    }

    pub fn evaluate(&self, state: &P5S1State, batch: &P5BatchBrier) -> Result<EvalResult> {
        validate_batch(state, batch)?;
        let device_batch = self.upload_batch(&PackedBatch::new(batch)?)?;
        let roots = self.stream.clone_htod(&state.roots)?;
        let mut base = self.allocate_baseline(&device_batch)?;
        let start = self.timed_event()?;
        self.scan(&roots, &device_batch, &mut base)?;
        let stop = self.timed_event()?;
        let loss_q32 = self.stream.clone_dtoh(&base.total)?[0];
        let picks = self.stream.clone_dtoh(&base.picks)?;
        let losses = self.stream.clone_dtoh(&base.row_losses)?;
        let mut rows = Vec::with_capacity(batch.rows.len());
        for ((row, pick), loss_q32) in batch.rows.iter().zip(picks.chunks_exact(2)).zip(losses) {
            let position = pick[0];
            let distance = pick[1];
            let successor = if position == 0 {
                if distance != u32::MAX || row.window.len() >= 3 {
                    return Err(fail("P5 CUDA returned inconsistent no-Pick record"));
                }
                0
            } else {
                if distance > 128 {
                    return Err(fail("P5 CUDA returned an out-of-range Pick distance"));
                }
                *row.window
                    .get(position as usize + 1)
                    .ok_or_else(|| fail("P5 CUDA Pick returned an out-of-range source"))?
            };
            rows.push(PickResult {
                position,
                distance,
                successor,
                loss_q32,
            });
        }
        let sum = rows
            .iter()
            .try_fold(0u64, |sum, row| sum.checked_add(row.loss_q32));
        if sum != Some(loss_q32) {
            return Err(fail("P5 CUDA per-row losses disagree with batch reduction"));
        }
        Ok(EvalResult {
            loss_q32,
            rows,
            device_micros: elapsed_micros(&start, &stop)?,
        })
    }

    pub fn step(
        &self,
        state: &mut P5S1State,
        batch: &P5BatchBrier,
        coords: &[u32],
        mode: MoveMode,
    ) -> Result<StepResult> {
        let plan_start = Instant::now();
        validate_batch(state, batch)?;
        validate_coords(state, coords)?;
        let plan = Proposals::new(state, batch, coords, &self.orbit_moves, mode);
        let packed = PackedBatch::new(batch)?;
        let plan_micros = plan_start.elapsed().as_micros() as u64;

        let upload_start = Instant::now();
        let device_batch = self.upload_batch(&packed)?;
        let k = coords.len() as u32;
        let coordinates = self.stream.clone_htod(coords)?;
        let mut roots = self.stream.clone_htod(&state.roots)?;
        let mut scratch = self.stream.clone_htod(&state.roots)?;
        let reps = self.stream.clone_htod(&plan.representative)?;
        let mut base = self.allocate_baseline(&device_batch)?;
        let lanes = coords.len() * P5_CANDIDATES;
        let mut deltas: CudaSlice<u64> = self.stream.alloc_zeros(lanes)?;
        let mut losses: CudaSlice<u64> = self.stream.alloc_zeros(lanes)?;
        let mut selected: CudaSlice<u8> = self.stream.alloc_zeros(coords.len())?;
        let mut committed: CudaSlice<u8> = self.stream.alloc_zeros(coords.len())?;
        let mut transition: CudaSlice<u64> = self.stream.alloc_zeros(TRANSITION_WORDS)?;
        let mut joint_loss: CudaSlice<u64> = self.stream.alloc_zeros(1)?;
        let score_inputs = if plan.jobs.is_empty() {
            None
        } else {
            Some((
                self.stream.clone_htod(&plan.jobs)?,
                self.stream.clone_htod(&plan.row_indices)?,
            ))
        };

        let start = self.timed_event()?;
        start.synchronize()?;
        let upload_allocate_micros = upload_start.elapsed().as_micros() as u64;
        self.scan(&roots, &device_batch, &mut base)?;
        let start_score = self.timed_event()?;
        if let Some((jobs, membership)) = &score_inputs {
            let mut launch = self.stream.launch_builder(&self.score);
            launch
                .arg(&roots)
                .arg(&self.device_distances)
                .arg(&self.device_moves)
                .arg(&coordinates)
                .arg(jobs)
                .arg(membership)
                .arg(&device_batch.windows)
                .arg(&device_batch.tokens)
                .arg(&device_batch.loss_offsets)
                .arg(&device_batch.losses)
                .arg(&base.distance)
                .arg(&base.row_losses)
                .arg(&mut deltas);
            // SAFETY: Each job refers to one valid lane and a contiguous membership tile.
            unsafe {
                launch.launch(launch_config(
                    (plan.jobs.len() / JOB_WORDS) as u32,
                    BLOCK_THREADS,
                ))?;
            }
        }
        let stop_score = self.timed_event()?;
        let mut choose = self.stream.launch_builder(&self.choose);
        choose
            .arg(&k)
            .arg(&coordinates)
            .arg(&self.device_moves)
            .arg(&reps)
            .arg(&base.total)
            .arg(&deltas)
            .arg(&mut losses)
            .arg(&mut scratch)
            .arg(&mut selected)
            .arg(&mut transition);
        // SAFETY: Unique coordinate indices, full 16-lane storage, and one decision thread.
        unsafe {
            choose.launch(launch_config(1, 1))?;
        }

        let mut joint = self.stream.launch_builder(&self.joint);
        joint
            .arg(&device_batch.rows)
            .arg(&scratch)
            .arg(&self.device_distances)
            .arg(&device_batch.windows)
            .arg(&device_batch.tokens)
            .arg(&device_batch.loss_offsets)
            .arg(&device_batch.losses)
            .arg(&transition)
            .arg(&mut joint_loss);
        // SAFETY: Same row storage and bounds as scan. Scratch contains in-orbit roots.
        unsafe {
            joint.launch(launch_config(
                device_batch.rows.div_ceil(WARPS_PER_BLOCK),
                BLOCK_THREADS,
            ))?;
        }

        let mut apply = self.stream.launch_builder(&self.apply);
        apply
            .arg(&k)
            .arg(&coordinates)
            .arg(&scratch)
            .arg(&selected)
            .arg(&joint_loss)
            .arg(&state.epoch)
            .arg(&mut transition)
            .arg(&mut roots)
            .arg(&mut committed);
        // SAFETY: Validated coordinates, in-orbit scratch and one application thread.
        unsafe {
            apply.launch(launch_config(1, 1))?;
        }
        let stop = self.timed_event()?;
        let device_micros = elapsed_micros(&start, &stop)?;
        let score_micros = if score_inputs.is_some() {
            elapsed_micros(&start_score, &stop_score)?
        } else {
            0
        };
        let readback_start = Instant::now();
        let scores = self.stream.clone_dtoh(&losses)?;
        let transition = self.stream.clone_dtoh(&transition)?;
        let committed = self.stream.clone_dtoh(&committed)?;
        let next_roots = self.stream.clone_dtoh(&roots)?;
        let l_before = self.stream.clone_dtoh(&base.total)?[0];
        let member_count = u32::try_from(transition[0])?;
        let l_joint = if member_count >= 2 {
            Some(self.stream.clone_dtoh(&joint_loss)?[0])
        } else {
            None
        };
        let l_best = transition[2];
        let joint = transition[3] == 1;
        let committed_loss = transition[4];
        let epoch_after = transition[5];
        let expected_epoch = state.epoch.checked_add(u64::from(member_count != 0));
        if member_count > k
            || transition[3] > 1
            || expected_epoch != Some(epoch_after)
            || joint != l_joint.is_some_and(|loss| loss < l_best)
            || committed_loss
                != if joint {
                    l_joint.unwrap_or(l_best)
                } else {
                    l_best
                }
            || (member_count != 0 && committed_loss >= l_before)
            || (member_count == 0 && committed_loss != l_before)
        {
            return Err(fail("P5 CUDA transition record violated BVG-K invariants"));
        }
        let mut record = P5BlockRecord {
            k,
            s_len: member_count,
            joint,
            l_before,
            l_joint,
            l_best,
            committed_loss,
            commit_ordinals: Vec::new(),
            commit_coords: Vec::new(),
            commit_lanes: Vec::new(),
            epoch_after,
        };
        let mut expected_roots = state.roots.clone();
        for (ord, (&coord, lane)) in coords.iter().zip(committed).enumerate() {
            if lane == 0 {
                continue;
            }
            let root = expected_roots[coord as usize];
            expected_roots[coord as usize] = *self.orbit_moves[usize::from(root)]
                .get(usize::from(lane))
                .ok_or_else(|| fail("P5 CUDA returned out-of-range committed lane"))?;
            record.commit_ordinals.push(ord as u32);
            record.commit_coords.push(coord);
            record.commit_lanes.push(lane);
        }
        let commit_count = if joint {
            member_count
        } else {
            u32::from(member_count != 0)
        };
        if next_roots != expected_roots || record.commit_coords.len() as u32 != commit_count {
            return Err(fail(
                "P5 CUDA committed roots did not match the transition record",
            ));
        }
        let scores = coords
            .iter()
            .zip(scores.chunks_exact(P5_CANDIDATES))
            .map(|(&coord, losses)| {
                Ok(P5CoordScore {
                    coord,
                    status: 0,
                    losses: losses.try_into()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let count_improved = scores
            .iter()
            .filter(|score| score.losses[1..].iter().any(|&loss| loss < l_before))
            .count() as u32;
        let minimum = scores.iter().flat_map(|score| score.losses).min();
        if scores.iter().any(|score| score.losses[0] != l_before)
            || count_improved != member_count
            || minimum != Some(l_best)
        {
            return Err(fail(
                "P5 CUDA lane losses disagree with the transition record",
            ));
        }
        state.roots = next_roots;
        state.epoch = epoch_after;
        let readback_validate_micros = readback_start.elapsed().as_micros() as u64;
        Ok(StepResult {
            record,
            scores,
            affected_rows: plan.affected_counts,
            evaluated_destinations: plan.representative_counts,
            device_micros,
            score_micros,
            plan_micros,
            upload_allocate_micros,
            readback_validate_micros,
        })
    }

    /// Differential replay of all integer losses, exact Pick address and distance,
    /// decision branches, exact root bytes, and epoch on a finite fixture.
    pub fn check_finite_parity(&self) -> Result<()> {
        let (moves, codes) = p5_orbit_tables();
        let (base, perturb) = (0usize..16)
            .flat_map(|a| (a + 1..16).map(move |b| (a, b)))
            .find(|&(a, b)| {
                codes[a] != codes[b]
                    && moves[a][1..].contains(&(b as u8))
                    && moves[b][1..].contains(&(a as u8))
            })
            .ok_or_else(|| fail("P5 fixture needs a mutually reachable pair of roots"))?;
        let mut initial = P5S1State::from_roots(8, 4, 7, [42; 32], vec![base as u8; 32])?;
        initial.roots[4] = perturb as u8;
        initial.roots[8] = perturb as u8;

        let row = |window: Vec<u16>, active: bool| {
            let p = window.len().saturating_sub(2);
            let mut losses = vec![300; p * 4];
            if active {
                losses[..p].fill(100);
            }
            P5RowBrier {
                window,
                target: 1,
                no_pick_brier_q32: 325,
                pick_brier_q32: losses,
            }
        };
        let controls = || {
            let mut rows = vec![row(vec![0, 1], false)];
            for i in 0..81 {
                rows.push(row(
                    (0..(3 + i))
                        .map(|j| [0, 1, 3, 2, 7][(i * i + j * (i + 1) + j * j) % 5])
                        .collect(),
                    false,
                ));
            }
            rows.push(row(vec![3; 64], false));
            rows
        };
        let mut independent = controls();
        let latest_tie_row = independent.len() - 1;
        independent.push(row(vec![1, 3, 3], true));
        independent.push(row(vec![2, 3, 3], true));
        let mut conflict = controls();
        conflict.push(row(vec![1, 3, 3], true));
        let mut varied = controls();
        for (row_index, row) in varied[..latest_tie_row].iter_mut().enumerate() {
            for (pick_index, loss) in row.pick_brier_q32.iter_mut().enumerate() {
                *loss = (173 * row_index as u64 + 223 * pick_index as u64) % 1024;
            }
        }
        let cases = [
            ("no-member", controls(), vec![4, 8, 24], Some((0, false))),
            ("single", independent.clone(), vec![4, 24], Some((1, false))),
            ("joint", independent, vec![4, 8, 24], Some((2, true))),
            ("fallback", conflict, vec![4, 12, 24], Some((2, false))),
            ("varied-loss", varied, vec![0, 4, 5, 8, 12, 24, 28], None),
        ];

        for (case, rows, coords, required_branch) in cases {
            let batch = P5BatchBrier::new(rows)?;
            let mut expected = initial.clone();
            let scores = p5_score_block_cpu(&expected, &batch, &coords)?;
            let record = p5_bvgk_step_cpu(&mut expected, &batch, &coords)?;
            if required_branch.is_some_and(|branch| branch != (record.s_len, record.joint)) {
                return Err(fail(format!("P5 finite fixture missed its {case} branch")));
            }
            let expected_bytes = expected.encode_bytes()?;
            let mut initial_eval = self.evaluate(&initial, &batch)?;
            check_eval(&initial_eval, &initial, &batch, &moves, &codes)?;
            if initial_eval.loss_q32 != record.l_before {
                return Err(fail(format!(
                    "P5 finite {case} initial loss differs from optimizer"
                )));
            }
            // The final Pick row in the controls ties across its entire window.
            // Force an earlier valid address with the same successor and loss.
            initial_eval.rows[latest_tie_row].position -= 1;
            if check_eval(&initial_eval, &initial, &batch, &moves, &codes).is_ok() {
                return Err(fail(
                    "P5 finite oracle accepted a deliberately wrong address",
                ));
            }
            for mode in [MoveMode::AllGenerators, MoveMode::DistinctDestinations] {
                let mut observed = initial.clone();
                let step = self.step(&mut observed, &batch, &coords, mode)?;
                let degree: Vec<u32> = coords
                    .iter()
                    .map(|&coord| {
                        batch
                            .rows
                            .iter()
                            .filter(|row| row.window.contains(&((coord / 4) as u16)))
                            .count() as u32
                    })
                    .collect();
                if step.scores != scores
                    || step.record != record
                    || step.affected_rows != degree
                    || observed.encode_bytes()? != expected_bytes
                {
                    return Err(fail(format!(
                        "P5 finite {case} failed {mode:?} integer replay"
                    )));
                }
                let eval = self.evaluate(&observed, &batch)?;
                check_eval(&eval, &expected, &batch, &moves, &codes)?;
                if eval.loss_q32 != step.record.committed_loss {
                    return Err(fail(format!(
                        "P5 finite {case} committed loss failed rescan"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn check_eval(
    eval: &EvalResult,
    state: &P5S1State,
    batch: &P5BatchBrier,
    moves: &[[u8; P5_CANDIDATES]; 16],
    codes: &[u16; 16],
) -> Result<()> {
    if eval.rows.len() != batch.rows.len() {
        return Err(fail("P5 finite replay changed the row count"));
    }
    for (index, (actual, row)) in eval.rows.iter().zip(&batch.rows).enumerate() {
        let pick = pick_with_virtual_move(&state.roots, moves, codes, &row.window, None);
        let (position, distance, successor) = pick
            .map(|(pos, distance)| (pos as u32, distance, row.window[pos + 1]))
            .unwrap_or((0, u32::MAX, 0));
        let expected = PickResult {
            position,
            distance,
            successor,
            loss_q32: row.brier_for_pick(pick),
        };
        if *actual != expected {
            return Err(fail(format!(
                "P5 finite replay row {index} changed the Pick or loss"
            )));
        }
    }
    Ok(())
}
