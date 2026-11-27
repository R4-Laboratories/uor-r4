use crate::p4_cuda::{P4GpuEngine, ParityStepOutHost};
use crate::{fail, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, Ptx, PushKernelArg,
    ValidAsZeroBits,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};

pub const WORDS: usize = 4095;
pub const CELLS: usize = 20;
const LANES: usize = 33;
const SLOT_BYTES: usize = 655_360;
const NO_POS: u32 = u32::MAX;
const Q48_ONE: u64 = 1u64 << 48;
const CUBIN: &[u8] = include_bytes!(env!("S4_CUBIN"));

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum Mode {
    Active,
    Disabled,
    WrongAddress,
    RoleBlind,
    ShuffledRole,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Input {
    votes: [i64; 3],
    offset: u32,
    len: u32,
    pick_pos: u32,
    pick_successor: u32,
    pick_class: u32,
    target: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Choice {
    pub resolved_role: u32,
    pub effective_role: u32,
    pub gate_index: u32,
    pub source_pos: u32,
    pub source: u32,
    pub rival: u32,
    pub weight: i32,
    pub status: u32,
    pub transfer: u64,
    pub source_before: u64,
    pub rival_before: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Evaluation {
    pub gate: Choice,
    pub token: u32,
    pub ties: u32,
    pub top_mass: u64,
    pub target_mass: u64,
    pub prll_mass: u64,
    pub prll_bits: u64,
    pub brier: u64,
}

// SAFETY: all three repr(C) ABIs are integer-only, checked against the CUDA layout.
unsafe impl DeviceRepr for Input {}
unsafe impl DeviceRepr for Evaluation {}
// SAFETY: zero is valid for every integer field, including the nested Choice.
unsafe impl ValidAsZeroBits for Evaluation {}
const _: [(); 48] = [(); size_of::<Input>()];
const _: [(); 56] = [(); size_of::<Choice>()];
const _: [(); 104] = [(); size_of::<Evaluation>()];

struct DeviceBatch {
    inputs: CudaSlice<Input>,
    flat: CudaSlice<u16>,
    scores: CudaSlice<u64>,
    rows: usize,
}

pub struct Batch {
    pub route: Vec<ParityStepOutHost>,
    inputs: DeviceBatch,
}

pub struct Fit {
    loss: CudaSlice<u64>,
    support: CudaSlice<u64>,
}

#[derive(Serialize)]
pub struct FitResult {
    pub weights: [i8; CELLS],
    pub support: [u64; CELLS],
    pub deltas_by_cell: Vec<Vec<i64>>,
}

pub struct Output {
    pub rows: Vec<Evaluation>,
    pub scores: Vec<u64>,
}

pub struct Gpu {
    stream: Arc<CudaStream>,
    accumulate: CudaFunction,
    commit: CudaFunction,
    forward: CudaFunction,
}

fn launch_config(rows: usize, threads: u32) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (u32::try_from(rows)?, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        Ok(Self {
            stream,
            accumulate: module.load_function("s4_fit_accumulate")?,
            commit: module.load_function("s4_fit_commit")?,
            forward: module.load_function("s4_forward")?,
        })
    }

    pub fn digest(&self) -> String {
        hex::encode(Sha256::digest(CUBIN))
    }

    fn upload(
        &self,
        prefixes: &[Vec<u16>],
        targets: &[u16],
        traces: &[ParityStepOutHost],
        scores: &[u64],
    ) -> Result<DeviceBatch> {
        if prefixes.is_empty()
            || prefixes.len() != targets.len()
            || prefixes.len() != traces.len()
            || scores.len() != prefixes.len() * WORDS
        {
            return Err(fail(
                "frozen P4 output dimensions do not match causal inputs",
            ));
        }
        let mut inputs = Vec::with_capacity(prefixes.len());
        let mut flat = Vec::new();
        for ((prefix, &target), trace) in prefixes.iter().zip(targets).zip(traces) {
            if trace.status != 0
                || trace.context_tokens as usize != prefix.len()
                || prefix.is_empty()
                || prefix.len() > 2048
                || prefix
                    .iter()
                    .enumerate()
                    .any(|(i, &t)| t > WORDS as u16 || (i > 0 && t == 0))
                || target == 0
                || target > WORDS as u16
            {
                return Err(fail(format!(
                    "invalid P4 route or causal row {}",
                    inputs.len()
                )));
            }
            inputs.push(Input {
                votes: trace.s4_role_scores,
                offset: u32::try_from(flat.len())?,
                len: u32::try_from(prefix.len())?,
                pick_pos: if trace.pick_has == 0 {
                    NO_POS
                } else {
                    trace.pick_pos
                },
                pick_successor: trace.pick_successor,
                pick_class: trace.pick_class,
                target: u32::from(target),
            });
            flat.extend_from_slice(prefix);
        }
        Ok(DeviceBatch {
            inputs: self.stream.clone_htod(&inputs)?,
            flat: self.stream.clone_htod(&flat)?,
            scores: self.stream.clone_htod(scores)?,
            rows: prefixes.len(),
        })
    }

    pub fn prepare(
        &self,
        p4: &P4GpuEngine,
        prefixes: &[Vec<u16>],
        targets: &[u16],
    ) -> Result<Batch> {
        // The old kernel is called for exactly one step. Its own argmax is never
        // fed back by the treatment. Generation observes only our device output.
        let (route, scores) = p4.run_parity_batch(
            prefixes,
            &vec![1; prefixes.len()],
            &vec![true; prefixes.len()],
        )?;
        let inputs = self.upload(prefixes, targets, &route, &scores)?;
        Ok(Batch { route, inputs })
    }

    pub fn start_fit(&self) -> Result<Fit> {
        Ok(Fit {
            loss: self.stream.alloc_zeros(CELLS * LANES)?,
            support: self.stream.alloc_zeros(CELLS)?,
        })
    }

    pub fn observe_fit(&self, fit: &mut Fit, batch: &Batch) -> Result<()> {
        let batch = &batch.inputs;
        let mut errors = self.stream.alloc_zeros::<u32>(batch.rows)?;
        let mut launch = self.stream.launch_builder(&self.accumulate);
        launch
            .arg(&batch.inputs)
            .arg(&batch.flat)
            .arg(&batch.scores)
            .arg(&mut fit.loss)
            .arg(&mut fit.support)
            .arg(&mut errors);
        // SAFETY: validated dimensions and matching repr(C) buffers.
        unsafe {
            launch.launch(launch_config(batch.rows, 64)?)?;
        }
        if self
            .stream
            .clone_dtoh(&errors)?
            .iter()
            .any(|&error| error != 0)
        {
            return Err(fail("s4 fit rejected the frozen raw address"));
        }
        Ok(())
    }

    pub fn finish_fit(&self, fit: &Fit) -> Result<FitResult> {
        let mut weights = self.stream.alloc_zeros::<i8>(CELLS)?;
        let mut launch = self.stream.launch_builder(&self.commit);
        launch.arg(&fit.loss).arg(&mut weights);
        // SAFETY: exact number and type of cells in both device buffers.
        unsafe {
            launch.launch(launch_config(1, 32)?)?;
        }
        let weights = self
            .stream
            .clone_dtoh(&weights)?
            .try_into()
            .map_err(|_| fail("incorrect fitted weight count"))?;
        let support = self
            .stream
            .clone_dtoh(&fit.support)?
            .try_into()
            .map_err(|_| fail("incorrect fit support count"))?;
        let deltas_by_cell: Vec<Vec<i64>> = self
            .stream
            .clone_dtoh(&fit.loss)?
            .chunks_exact(LANES)
            .map(|cell| cell.iter().map(|&word| word as i64).collect())
            .collect();
        Ok(FitResult {
            weights,
            support,
            deltas_by_cell,
        })
    }

    pub fn evaluate(
        &self,
        batch: &Batch,
        weights: &[i8; CELLS],
        mode: Mode,
        capture: bool,
        measure_prll: bool,
    ) -> Result<Output> {
        let batch = &batch.inputs;
        if weights.iter().any(|w| !(-64..=64).contains(w)) {
            return Err(fail("invalid signed gate weight"));
        }
        let mut slots = self.stream.alloc_zeros::<u8>(batch.rows * SLOT_BYTES)?;
        let mut out = self.stream.alloc_zeros::<Evaluation>(batch.rows)?;
        let mut words =
            self.stream
                .alloc_zeros::<u64>(if capture { batch.rows * WORDS } else { 1 })?;
        let mode = mode as u32;
        let should_capture = u32::from(capture);
        let should_measure = u32::from(measure_prll);
        let weights = self.stream.clone_htod(weights)?;
        let mut launch = self.stream.launch_builder(&self.forward);
        launch
            .arg(&batch.inputs)
            .arg(&batch.flat)
            .arg(&batch.scores)
            .arg(&weights)
            .arg(&mode)
            .arg(&should_capture)
            .arg(&should_measure)
            .arg(&mut slots)
            .arg(&mut out)
            .arg(&mut words);
        // SAFETY: bounded weights, checked device ABI, one non-aliasing slot per row.
        unsafe {
            launch.launch(launch_config(batch.rows, 256)?)?;
        }
        let rows = self.stream.clone_dtoh(&out)?;
        if let Some((idx, _)) = rows.iter().enumerate().find(|(_, r)| r.gate.status != 0) {
            return Err(fail(format!(
                "s4 evaluation rejected raw address at row {idx}"
            )));
        }
        let scores = if capture {
            self.stream.clone_dtoh(&words)?
        } else {
            Vec::new()
        };
        Ok(Output { rows, scores })
    }

    pub fn finite(&self) -> Result<()> {
        let prefix = vec![0, 7, 11, 2, 5];
        let mut scores = vec![1u64; WORDS];
        scores[10] = 9 * (Q48_ONE / 32);
        scores[16] = 10 * (Q48_ONE / 32);
        let route = ParityStepOutHost {
            context_tokens: 5,
            pick_has: 1,
            pick_pos: 1,
            pick_class: 1,
            pick_successor: 11,
            s4_role_scores: [9, 0, -2],
            ..Default::default()
        };
        let mut tied = route;
        tied.s4_role_scores = [9, 9, 0];
        let mut role_two = route;
        role_two.s4_role_scores = [0, 9, -2];
        let routes = vec![route, tied, role_two];
        let prefixes = vec![prefix.clone(); routes.len()];
        let frozen = scores.repeat(routes.len());
        let batch = Batch {
            inputs: self.upload(&prefixes, &[11, 11, 17], &routes, &frozen)?,
            route: routes,
        };
        let mut fit = self.start_fit()?;
        self.observe_fit(&mut fit, &batch)?;
        let fit = self.finish_fit(&fit)?;
        if fit.weights[3] != 64
            || fit.weights[4] != -64
            || fit.weights[16] != 12
            || fit.support[3] != 1
            || fit.support[4] != 1
            || fit.support[16] != 2
            || fit.support.iter().sum::<u64>() != 4
            || fit.deltas_by_cell[3][32] != -639_547_391
            || fit.deltas_by_cell[4][31] != -542_557_183
            || fit.deltas_by_cell[16][6] != -4_124_414
        {
            return Err(fail(
                "finite CUDA optimizer failed to learn the true signed correction",
            ));
        }
        let active = self.evaluate(&batch, &fit.weights, Mode::Active, true, true)?;
        let mut reversed = fit.weights;
        reversed[3] = -64;
        let reverse = self.evaluate(&batch, &reversed, Mode::Active, true, true)?;
        let zero = self.evaluate(&batch, &[0; CELLS], Mode::Active, true, false)?;
        let blind = self.evaluate(&batch, &fit.weights, Mode::RoleBlind, false, false)?;
        let off = self.evaluate(&batch, &fit.weights, Mode::Disabled, true, true)?;
        let wrong = self.evaluate(&batch, &fit.weights, Mode::WrongAddress, false, false)?;
        let shuffle = self.evaluate(&batch, &fit.weights, Mode::ShuffledRole, false, false)?;
        let expected_transfer = (scores[16] - 1) >> 2;
        if active.rows[0].token != 11
            || active.rows[0].gate.source_pos != 2
            || active.rows[0].gate.transfer != expected_transfer
            || active.scores[10] != scores[10] + expected_transfer
            || active.scores[16] != scores[16] - expected_transfer
            || active.rows[0].brier >= off.rows[0].brier
            || active.rows[0].prll_bits >= off.rows[0].prll_bits
            || reverse.rows[0].brier <= off.rows[0].brier
            || reverse.rows[0].prll_bits <= off.rows[0].prll_bits
            || reverse.scores[10] + reverse.rows[0].gate.transfer != scores[10]
            || reverse.scores[16] != scores[16] + reverse.rows[0].gate.transfer
            || zero.scores != frozen
            || active.rows[1].token != 17
            || active.rows[1].gate.source != 0
            || active.rows[2].token != 17
            || active.rows[2].gate.transfer != (scores[10] - 1) >> 2
            || active.scores[2 * WORDS + 10] + active.rows[2].gate.transfer != scores[10]
            || active.scores[2 * WORDS + 16] != scores[16] + active.rows[2].gate.transfer
            || blind.rows[0].token != 17
            || blind.rows[1].token != 17
            || wrong.rows[0].gate.source_pos != 3
            || wrong.rows[0].gate.source != 2
            || shuffle.rows[0].gate.effective_role != 2
            || shuffle.rows[0].token != 17
            || off.scores != frozen
        {
            return Err(fail(
                "finite CUDA source, ablation or mass-transfer check failed",
            ));
        }
        let mut saturated = scores.clone();
        saturated[10] = Q48_ONE;
        let boundary = Batch {
            inputs: self.upload(&[prefix.clone()], &[11], &[route], &saturated)?,
            route: vec![route],
        };
        let capped = self.evaluate(&boundary, &fit.weights, Mode::Active, true, false)?;
        if capped.scores != saturated || capped.rows[0].gate.transfer != 0 {
            return Err(fail("CUDA transfer overran the receiving Q48 ceiling"));
        }
        let mut at_floor = scores.clone();
        at_floor[10] = 1;
        let boundary = Batch {
            inputs: self.upload(&[prefix.clone()], &[11], &[route], &at_floor)?,
            route: vec![route],
        };
        let floor_result = self.evaluate(&boundary, &reversed, Mode::Active, true, false)?;
        if floor_result.scores != at_floor || floor_result.rows[0].gate.transfer != 0 {
            return Err(fail("CUDA transfer crossed the departing one-unit floor"));
        }
        let mut no_alternate = route;
        no_alternate.context_tokens = 3;
        let short = Batch {
            inputs: self.upload(&[vec![0, 11, 11]], &[11], &[no_alternate], &scores)?,
            route: vec![no_alternate],
        };
        let no_wrong = self.evaluate(&short, &fit.weights, Mode::WrongAddress, true, false)?;
        if no_wrong.scores != scores
            || no_wrong.rows[0].gate.source != 0
            || no_wrong.rows[0].gate.gate_index != 3
            || no_wrong.rows[0].gate.weight != 64
        {
            return Err(fail("CUDA fabricated an alternative source"));
        }
        let mut bad = route;
        bad.pick_pos = 2;
        let batch = Batch {
            inputs: self.upload(&[prefix], &[11], &[bad], &scores)?,
            route: vec![bad],
        };
        if self
            .evaluate(&batch, &fit.weights, Mode::Active, false, false)
            .is_ok()
        {
            return Err(fail("planted off-by-one raw address was accepted"));
        }
        Ok(())
    }
}
