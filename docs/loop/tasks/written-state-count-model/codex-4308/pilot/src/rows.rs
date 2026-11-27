use crate::{fail, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, Ptx, PushKernelArg,
    ValidAsZeroBits,
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use uor_r4_count_runtime::{
    kn_score::KN_BASE_MASS,
    p4_serving::{
        skip_keys_for_prefix, CausalSessionCache, P4Session, P4State, P4_BASE_BUCKETS, P4_EXPERTS,
        P4_MAX_CONTEXT, P4_Q48_ONE, P4_TOTAL_BUCKETS, P4_VOCAB,
    },
};
use uor_r4_discrete::p5_brier_core::{
    full_vocab_brier_q32, p5_orbit_tables, P5BatchBrier, P5RowBrier, P5_MAX_BATCH_ROWS,
    P5_PICK_CLASSES,
};
use uor_r4_integer::discrete_d11::count_deployment::{
    load_client_artifact, ClientArtifactFile, DiscreteCountModel,
};

#[allow(dead_code)]
#[path = "../../../../../../../crates/uor-r4-integer/src/lawful_likelihood.rs"]
mod lawful_likelihood;
#[allow(dead_code, unused_imports)]
#[path = "../../../antigravity-17/p4-qual/src/cuda.rs"]
mod p4_cuda;

const CUBIN: &[u8] = include_bytes!(env!("P5_ROWS_CUBIN"));
// Same conservative bound used by P4GpuEngine, checked by p4_kernels.cu.
const SLOT_BYTES: usize = 655_360;
const ROWS_PER_LAUNCH: usize = 128;
const MAX_PARITY_ROWS: usize = 32;
const COUNTERFACTUAL_ROWS: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Absolute token index of the record BOS in the supplied split.
    pub record: usize,
    /// Positive offset of the true target within that record.
    pub target_at: usize,
}

#[derive(Clone, Debug)]
pub struct NativeRow {
    pub target: u16,
    pub top1: u16,
    pub top1_mass_q48: u64,
    pub target_mass_q48: u64,
    /// Computed by `evaluate`; zero in `Prepared::native`.
    pub prll_mass_q48: u64,
    pub prll_bits_q12: u64,
    pub brier_q32: u64,
    pub pick_pos: Option<usize>,
    pub pick_distance: Option<u32>,
    pub pick_successor: Option<u16>,
    pub pick_class: u8,
    pub pick_ties: u32,
    pub bucket: usize,
}

pub struct Prepared {
    pub batch: P5BatchBrier,
    pub native: Vec<NativeRow>,
    /// Sum of the optimistic per-row feasible-superset minima.
    pub oracle_brier_q32: u64,
    /// Keep the native distance class and optimize the source position.
    pub source_only_oracle_brier_q32: u64,
    /// Keep the native source position and optimize the distance class.
    pub class_only_oracle_brier_q32: u64,
    pub oracle_by_row_q32: Vec<u64>,
    pub oracle_rows: usize,
    pub prepare_time: Duration,
    pub gpu_time: Duration,
    pub materialize_time: Duration,
    pub oracle_time: Duration,
}

#[derive(serde::Serialize)]
pub struct ParityChecks {
    pub native_rows: usize,
    pub native_score_words: usize,
    pub counterfactual_rows: usize,
    pub counterfactuals: usize,
    pub counterfactual_score_words: usize,
    pub class_sensitive_rows: usize,
    pub position_sensitive_rows: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RowInput {
    prefix_offset: u32,
    prefix_len: u32,
    target: u32,
    brier_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeOutput {
    target_mass: u64,
    top_mass: u64,
    prll_mass: u64,
    prll_bits: u64,
    brier: u64,
    skip_keys: [u64; 3],
    top: u32,
    top_ties: u32,
    pick_has: u32,
    pick_pos: u32,
    pick_dist: u32,
    pick_ties: u32,
    pick_succ: u32,
    pick_class: u32,
    bucket: u32,
    highest_order: u32,
    s3_pkey: u32,
    s3_backoff: u32,
    s3_mode: u32,
    s2_hit: u32,
    cache_hit: u32,
    prefix_len: u32,
}

// SAFETY: These primitive-only repr(C) layouts match the checked CUDA ABIs.
unsafe impl DeviceRepr for RowInput {}
unsafe impl DeviceRepr for NativeOutput {}
// SAFETY: NativeOutput consists entirely of integer fields.
unsafe impl ValidAsZeroBits for NativeOutput {}
const _: [(); 16] = [(); std::mem::size_of::<RowInput>()];
const _: [(); 128] = [(); std::mem::size_of::<NativeOutput>()];

impl NativeOutput {
    fn public(self, target: u16) -> NativeRow {
        let picked = self.pick_has != 0;
        NativeRow {
            target,
            top1: self.top as u16,
            top1_mass_q48: self.top_mass,
            target_mass_q48: self.target_mass,
            prll_mass_q48: self.prll_mass,
            prll_bits_q12: self.prll_bits,
            brier_q32: self.brier,
            pick_pos: picked.then_some(self.pick_pos as usize),
            pick_distance: picked.then_some(self.pick_dist),
            pick_successor: picked.then_some(self.pick_succ as u16),
            pick_class: self.pick_class as u8,
            pick_ties: self.pick_ties,
            bucket: self.bucket as usize,
        }
    }
}

fn active_prefix(tokens: &[u16], sample: Sample) -> Result<(&[u16], u16)> {
    let target_idx = sample
        .record
        .checked_add(sample.target_at)
        .ok_or_else(|| fail("sample target offset overflow"))?;
    let record = tokens
        .get(sample.record..=target_idx)
        .ok_or_else(|| fail(format!("out-of-bounds sample {sample:?}")))?;
    let (&target, prefix) = record
        .split_last()
        .ok_or_else(|| fail(format!("empty sample {sample:?}")))?;
    if prefix.first() != Some(&0)
        || target == 0
        || usize::from(target) >= P4_VOCAB
        || prefix[1..]
            .iter()
            .any(|&id| id < 2 || usize::from(id) >= P4_VOCAB)
    {
        return Err(fail(format!(
            "sample does not identify one open causal record: {sample:?}"
        )));
    }
    // P4Session::observe_generated rebuilds its cache from the exact active
    // window once history slides. Reconstructing that window is equivalent.
    let start = prefix.len().saturating_sub(P4_MAX_CONTEXT);
    Ok((&prefix[start..], target))
}

fn rotated_weights(weights: &[u64]) -> Vec<u64> {
    let plain_bucket_count = P4_BASE_BUCKETS / P5_PICK_CLASSES;
    let class_words = plain_bucket_count * P4_EXPERTS;
    let mut out = Vec::with_capacity(weights.len() * P5_PICK_CLASSES);
    for rotation in 0..P5_PICK_CLASSES {
        for cache in weights.chunks_exact(P4_BASE_BUCKETS * P4_EXPERTS) {
            for class in 0..P5_PICK_CLASSES {
                let from = (class + rotation) % P5_PICK_CLASSES * class_words;
                out.extend_from_slice(&cache[from..from + class_words]);
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
struct Oracle {
    single: u8,
    doubled: u8,
    independent: u8,
}

#[derive(Default)]
struct OracleLosses {
    joint: u64,
    source_only: u64,
    class_only: u64,
}

impl Oracle {
    fn orbit() -> Self {
        // A distinct pair of plain IDs can alias or differ. Enumerate all
        // capped (0,1,2,>=3) distances from the actual P5 root codes.
        let (_, codes) = p5_orbit_tables();
        let mut one_block = 0u8;
        for &a in &codes {
            for &b in &codes {
                one_block |= 1 << (a ^ b).count_ones().min(3);
            }
        }
        let single = (0..4).fold(1, |mask, _| Self::sums(mask, one_block));
        let doubled = (0..4)
            .filter(|&d| single & (1 << d) != 0)
            .fold(0, |mask, d| mask | (1 << (d * 2).min(3)));
        Self {
            single,
            doubled,
            independent: Self::sums(single, single),
        }
    }

    fn sums(left: u8, right: u8) -> u8 {
        let mut result = 0;
        for a in 0..4 {
            for b in 0..4 {
                if left & (1 << a) != 0 && right & (1 << b) != 0 {
                    result |= 1 << (a + b).min(3);
                }
            }
        }
        result
    }

    fn minima(
        self,
        row: &P5RowBrier,
        native: &NativeRow,
        seen: &mut HashSet<(u16, u16)>,
    ) -> Result<OracleLosses> {
        let positions = row.window.len().saturating_sub(2);
        if positions == 0 {
            let loss = row.no_pick_brier_q32;
            if native.pick_pos.is_some() || native.pick_class != 0 || native.brier_q32 != loss {
                return Err(fail("native score differs from the sole no-source outcome"));
            }
            return Ok(OracleLosses {
                joint: loss,
                source_only: loss,
                class_only: loss,
            });
        }
        let query = (row.window[positions], row.window[positions + 1]);
        let last_exact = (1..=positions)
            .rev()
            .find(|&p| (row.window[p - 1], row.window[p]) == query);
        let mut best = OracleLosses {
            joint: u64::MAX,
            source_only: u64::MAX,
            class_only: u64::MAX,
        };
        let mut includes_native = false;
        seen.clear();
        for p in (last_exact.unwrap_or(1)..=positions).rev() {
            let pair = (row.window[p - 1], row.window[p]);
            // Identical plain source pairs always tie. Only the latest can win.
            if !seen.insert(pair) {
                continue;
            }
            let mask = if last_exact.is_some() {
                // An exact query pair forces the minimum distance to zero.
                1
            } else if pair.0 == query.0 || pair.1 == query.1 {
                self.single
            } else if pair.0 == query.1 && pair.1 == query.0 {
                self.doubled
            } else {
                self.independent
            };
            for distance in 0..4 {
                if mask & (1 << distance) != 0 {
                    let loss = row.pick_brier_q32[distance * positions + p - 1];
                    let native_class = usize::from(native.pick_class) == distance + 1;
                    let native_position = native.pick_pos == Some(p);
                    best.joint = best.joint.min(loss);
                    if native_class {
                        best.source_only = best.source_only.min(loss);
                    }
                    if native_position {
                        best.class_only = best.class_only.min(loss);
                    }
                    includes_native |= native_position && native_class;
                }
            }
        }
        if !includes_native {
            return Err(fail("oracle feasible superset excluded the native route"));
        }
        for restricted in [best.source_only, best.class_only] {
            if best.joint > restricted || restricted > native.brier_q32 {
                return Err(fail("oracle losses violate joint <= restricted <= native"));
            }
        }
        Ok(best)
    }
}

pub struct RowsGpu {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    pub cubin_sha256: String,
    pub artifact_sha256: String,
    state_sha256: String,
    order_count: u32,
    knc: CudaSlice<u8>,
    orders: CudaSlice<u64>,
    codes: CudaSlice<u16>,
    token_h4: CudaSlice<u8>,
    h4_shell: CudaSlice<u8>,
    s3_index: [CudaSlice<u8>; 2],
    s3_blob: [CudaSlice<u8>; 2],
    s3_len: [u32; 2],
    s2_index: [CudaSlice<u8>; 3],
    s2_blob: [CudaSlice<u8>; 3],
    s2_len: [u32; 3],
    weights: CudaSlice<u64>,
}

struct Chunk {
    native: Vec<NativeOutput>,
    no_pick: Vec<u64>,
    pick_brier: Vec<u64>,
    scores: Vec<u64>,
    inputs: Vec<RowInput>,
    prepare_time: Duration,
    gpu_time: Duration,
    materialize_time: Duration,
}

impl RowsGpu {
    pub fn new(p4_path: &Path) -> Result<Self> {
        let ClientArtifactFile::Embedded {
            model,
            state_bytes,
            expected_state_sha256,
            identity,
        } = load_client_artifact(p4_path, false)?
        else {
            return Err(fail("expected a bound embedded P4 deployment"));
        };
        let artifact_sha256 = identity.artifact_sha256.clone();
        let deployment = DiscreteCountModel::bind_embedded(
            model,
            &state_bytes,
            &expected_state_sha256,
            identity,
        )?;
        let p4 = &deployment
            .p4
            .as_ref()
            .ok_or_else(|| fail("deployment has no P4 state"))?
            .inner;
        let raw = p4_cuda::split_uorp4a01_sections(&state_bytes)?;
        let (order_count, orders) = p4_cuda::parse_uorknc03_descriptors(raw.counts_v3)?;
        let s3 = p4_cuda::parse_uors3c01_slices(raw.s3_compact)?;
        let s2 = p4_cuda::parse_uors2c01_slices(raw.s2_skip_compact)?;
        if p4.s6.weights_q32.len() != P4_TOTAL_BUCKETS * P4_EXPERTS {
            return Err(fail("unexpected P4 S6 weight dimensions"));
        }
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        let kernel = module.load_function("p5_prepare_rows")?;
        let upload_bytes = |bytes: &[u8]| {
            let bytes = if bytes.is_empty() { &[0] } else { bytes };
            stream.clone_htod(bytes)
        };
        let out = Self {
            stream: Arc::clone(&stream),
            kernel,
            cubin_sha256: hex::encode(Sha256::digest(CUBIN)),
            artifact_sha256,
            state_sha256: expected_state_sha256,
            order_count,
            knc: upload_bytes(raw.counts_v3)?,
            orders: stream.clone_htod(&orders)?,
            codes: stream.clone_htod(&p4.codes.iter().flatten().copied().collect::<Vec<_>>())?,
            token_h4: upload_bytes(s3.token_h4)?,
            h4_shell: upload_bytes(&p4_cuda::flat_h4_shell())?,
            s3_index: [upload_bytes(s3.p_idx)?, upload_bytes(s3.b_idx)?],
            s3_blob: [upload_bytes(s3.p_blob)?, upload_bytes(s3.b_blob)?],
            s3_len: [s3.p_ctx_len, s3.b_ctx_len],
            s2_index: [
                upload_bytes(s2.idx[0])?,
                upload_bytes(s2.idx[1])?,
                upload_bytes(s2.idx[2])?,
            ],
            s2_blob: [
                upload_bytes(s2.blob[0])?,
                upload_bytes(s2.blob[1])?,
                upload_bytes(s2.blob[2])?,
            ],
            s2_len: s2.ctx_len,
            weights: stream.clone_htod(&rotated_weights(&p4.s6.weights_q32))?,
        };
        out.stream.synchronize()?;
        Ok(out)
    }

    fn launch(
        &self,
        prefixes: &[(&[u16], u16)],
        codes: &CudaSlice<u16>,
        full_tables: bool,
        prll: bool,
        capture_scores: bool,
    ) -> Result<Chunk> {
        let start = Instant::now();
        let mut flat = Vec::new();
        let mut inputs = Vec::with_capacity(prefixes.len());
        let mut brier_len = 0usize;
        for &(prefix, target) in prefixes {
            inputs.push(RowInput {
                prefix_offset: u32::try_from(flat.len())?,
                prefix_len: u32::try_from(prefix.len())?,
                target: target.into(),
                brier_offset: u32::try_from(brier_len)?,
            });
            flat.extend_from_slice(prefix);
            if full_tables {
                brier_len += (P5_PICK_CLASSES - 1) * prefix.len().saturating_sub(2);
            }
        }
        let d_flat = self.stream.clone_htod(&flat)?;
        let d_inputs = self.stream.clone_htod(&inputs)?;
        let rotations = if full_tables { P5_PICK_CLASSES } else { 1 };
        let slots = prefixes.len() * rotations;
        let mut d_slots = self.stream.alloc_zeros::<u8>(slots * SLOT_BYTES)?;
        let mut d_status = self.stream.alloc_zeros::<u32>(slots)?;
        let mut d_native = self.stream.alloc_zeros::<NativeOutput>(prefixes.len())?;
        let mut d_no_pick =
            self.stream
                .alloc_zeros::<u64>(if full_tables { prefixes.len() } else { 1 })?;
        let mut d_pick = self.stream.alloc_zeros::<u64>(brier_len.max(1))?;
        let score_len = if capture_scores {
            prefixes.len() * P4_VOCAB
        } else {
            1
        };
        let mut d_scores = self.stream.alloc_zeros::<u64>(score_len)?;
        let full = u32::from(full_tables);
        let prll = u32::from(prll);
        let capture = u32::from(capture_scores);
        self.stream.synchronize()?;
        let prepare_time = start.elapsed();
        let started_kernel = Instant::now();
        let mut args = self.stream.launch_builder(&self.kernel);
        args.arg(&d_flat)
            .arg(&d_inputs)
            .arg(&self.knc)
            .arg(&self.orders)
            .arg(&self.order_count)
            .arg(codes)
            .arg(&self.token_h4)
            .arg(&self.h4_shell)
            .arg(&self.s3_index[0])
            .arg(&self.s3_blob[0])
            .arg(&self.s3_index[1])
            .arg(&self.s3_blob[1])
            .arg(&self.s3_len[0])
            .arg(&self.s3_len[1])
            .arg(&self.s2_index[0])
            .arg(&self.s2_blob[0])
            .arg(&self.s2_len[0])
            .arg(&self.s2_index[1])
            .arg(&self.s2_blob[1])
            .arg(&self.s2_len[1])
            .arg(&self.s2_index[2])
            .arg(&self.s2_blob[2])
            .arg(&self.s2_len[2])
            .arg(&self.weights)
            .arg(&full)
            .arg(&prll)
            .arg(&capture)
            .arg(&mut d_slots)
            .arg(&mut d_status)
            .arg(&mut d_native)
            .arg(&mut d_no_pick)
            .arg(&mut d_pick)
            .arg(&mut d_scores);
        // SAFETY: Arguments follow the C kernel ABI. Each (row,rotation) owns
        // one bounded scratch slot and the permuted classes never overlap.
        unsafe {
            args.launch(LaunchConfig {
                grid_dim: (u32::try_from(prefixes.len())?, u32::try_from(rotations)?, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        self.stream.synchronize()?;
        let gpu_time = started_kernel.elapsed();
        let materialized = Instant::now();
        let status = self.stream.clone_dtoh(&d_status)?;
        if let Some((slot, code)) = status.into_iter().enumerate().find(|&(_, code)| code != 0) {
            return Err(fail(format!(
                "P4 CUDA row {}, rotation {} returned status {code}",
                slot / rotations,
                slot % rotations
            )));
        }
        Ok(Chunk {
            native: self.stream.clone_dtoh(&d_native)?,
            no_pick: if full_tables {
                self.stream.clone_dtoh(&d_no_pick)?
            } else {
                Vec::new()
            },
            pick_brier: if full_tables {
                self.stream.clone_dtoh(&d_pick)?
            } else {
                Vec::new()
            },
            scores: if capture_scores {
                self.stream.clone_dtoh(&d_scores)?
            } else {
                Vec::new()
            },
            inputs,
            prepare_time,
            gpu_time,
            materialize_time: materialized.elapsed(),
        })
    }

    pub fn prepare(&self, tokens: &[u16], samples: &[Sample]) -> Result<Prepared> {
        self.prepare_impl(tokens, samples, true)
    }

    pub fn prepare_for_training(&self, tokens: &[u16], samples: &[Sample]) -> Result<Prepared> {
        self.prepare_impl(tokens, samples, false)
    }

    fn prepare_impl(
        &self,
        tokens: &[u16],
        samples: &[Sample],
        with_oracle: bool,
    ) -> Result<Prepared> {
        if samples.is_empty() || samples.len() > P5_MAX_BATCH_ROWS {
            return Err(fail("P5 preparation requires 1..=4096 rows"));
        }
        let mut rows = Vec::with_capacity(samples.len());
        let mut native = Vec::with_capacity(samples.len());
        let mut oracle_totals = OracleLosses::default();
        let mut oracle_by_row_q32 = Vec::new();
        let mut oracle = None;
        let mut oracle_time = Duration::ZERO;
        if with_oracle {
            let start = Instant::now();
            oracle = Some((Oracle::orbit(), HashSet::new()));
            oracle_by_row_q32.reserve_exact(samples.len());
            oracle_time = start.elapsed();
        }
        let mut prepare_time = Duration::ZERO;
        let mut gpu_time = Duration::ZERO;
        let mut materialize_time = Duration::ZERO;
        for samples in samples.chunks(ROWS_PER_LAUNCH) {
            let start = Instant::now();
            let prefixes = gather(tokens, samples)?;
            prepare_time += start.elapsed();
            let result = self.launch(&prefixes, &self.codes, true, false, false)?;
            prepare_time += result.prepare_time;
            gpu_time += result.gpu_time;
            materialize_time += result.materialize_time;
            let start = Instant::now();
            let prior_oracle_time = oracle_time;
            for (i, (&(prefix, target), raw)) in prefixes.iter().zip(result.native).enumerate() {
                let start = result.inputs[i].brier_offset as usize;
                let end = start + (P5_PICK_CLASSES - 1) * prefix.len().saturating_sub(2);
                let row = P5RowBrier {
                    window: prefix.to_vec(),
                    target,
                    no_pick_brier_q32: result.no_pick[i],
                    pick_brier_q32: result.pick_brier[start..end].to_vec(),
                };
                check_native(raw, prefix, target, false)?;
                let observed = raw.public(target);
                if row.brier_for_pick(observed.pick_pos.zip(observed.pick_distance))
                    != observed.brier_q32
                {
                    return Err(fail(
                        "GPU table entry differs from full native vocabulary Brier",
                    ));
                }
                if let Some((oracle, seen)) = &mut oracle {
                    let start = Instant::now();
                    let minima = oracle.minima(&row, &observed, seen)?;
                    oracle_totals.joint += minima.joint;
                    oracle_totals.source_only += minima.source_only;
                    oracle_totals.class_only += minima.class_only;
                    oracle_by_row_q32.push(minima.joint);
                    oracle_time += start.elapsed();
                }
                rows.push(row);
                native.push(observed);
            }
            materialize_time += start.elapsed() - (oracle_time - prior_oracle_time);
        }
        let batch = P5BatchBrier::new(rows)?;
        batch.validate_for_vocab(P4_VOCAB)?;
        Ok(Prepared {
            batch,
            native,
            oracle_brier_q32: oracle_totals.joint,
            source_only_oracle_brier_q32: oracle_totals.source_only,
            class_only_oracle_brier_q32: oracle_totals.class_only,
            oracle_rows: oracle_by_row_q32.len(),
            oracle_by_row_q32,
            prepare_time,
            gpu_time,
            materialize_time,
            oracle_time,
        })
    }

    /// Evaluate the unchanged P4 kernel with its frozen experts/cache and the
    /// requested S1 codes. A bound P4 artifact with the updated trunk can also
    /// be opened with `new` and passed `None`.
    pub fn evaluate(
        &self,
        tokens: &[u16],
        samples: &[Sample],
        replacement_codes: Option<&[[u16; 4]]>,
    ) -> Result<Vec<NativeRow>> {
        let replacement = self.upload_codes(replacement_codes)?;
        let codes = replacement.as_ref().unwrap_or(&self.codes);
        let mut rows = Vec::with_capacity(samples.len());
        for samples in samples.chunks(ROWS_PER_LAUNCH) {
            let prefixes = gather(tokens, samples)?;
            let result = self.launch(&prefixes, codes, false, true, false)?;
            for (&(prefix, target), raw) in prefixes.iter().zip(result.native) {
                check_native(raw, prefix, target, true)?;
                rows.push(raw.public(target));
            }
        }
        Ok(rows)
    }

    /// Hash the scored legal Q48 words in ordinary P4 trace order.
    pub fn mass_hashes(
        &self,
        tokens: &[u16],
        samples: &[Sample],
        replacement_codes: Option<&[[u16; 4]]>,
    ) -> Result<Vec<String>> {
        if samples.is_empty() || samples.len() > 5 {
            return Err(fail("P4 mass hashes accept 1..=5 diagnostic samples"));
        }
        let prefixes = gather(tokens, samples)?;
        let replacement = self.upload_codes(replacement_codes)?;
        let codes = replacement.as_ref().unwrap_or(&self.codes);
        let result = self.launch(&prefixes, codes, false, false, true)?;
        let mut hashes = Vec::with_capacity(samples.len());
        for ((&(prefix, target), raw), scores) in prefixes
            .iter()
            .zip(result.native)
            .zip(result.scores.chunks_exact(P4_VOCAB))
        {
            check_native(raw, prefix, target, false)?;
            let mut digest = Sha256::new();
            for &score in &scores[1..] {
                digest.update(score.to_le_bytes());
            }
            hashes.push(hex::encode(digest.finalize()));
        }
        Ok(hashes)
    }

    fn upload_codes(&self, replacement: Option<&[[u16; 4]]>) -> Result<Option<CudaSlice<u16>>> {
        let Some(codes) = replacement else {
            return Ok(None);
        };
        if codes.len() != P4_VOCAB {
            return Err(fail(
                "replacement S1 Pick codes must contain all 4096 plain IDs",
            ));
        }
        Ok(Some(self.stream.clone_htod(
            &codes.iter().flatten().copied().collect::<Vec<_>>(),
        )?))
    }

    /// Diagnostic only. Compare all vocabulary Q48 words, the native source
    /// trace, exact Q32 Brier and integer PRLL against the production session.
    /// On up to three rows with an alternate legal successor, also compare
    /// each class lookup against independent full-vocabulary recomputation.
    pub fn check_parity(
        &self,
        p4_path: &Path,
        tokens: &[u16],
        samples: &[Sample],
        replacement_codes: Option<&[[u16; 4]]>,
    ) -> Result<ParityChecks> {
        if samples.is_empty() || samples.len() > MAX_PARITY_ROWS {
            return Err(fail("P4 parity accepts 1..=32 diagnostic samples"));
        }
        let ClientArtifactFile::Embedded {
            model,
            state_bytes,
            expected_state_sha256,
            identity,
        } = load_client_artifact(p4_path, false)?
        else {
            return Err(fail("expected an embedded P4 deployment for parity"));
        };
        if expected_state_sha256 != self.state_sha256 {
            return Err(fail(
                "parity P4 fixed-state digest disagrees with the CUDA model",
            ));
        }
        if replacement_codes.is_none() && identity.artifact_sha256 != self.artifact_sha256 {
            return Err(fail(
                "parity P4 deployment digest disagrees with the CUDA model",
            ));
        }
        let mut bound = DiscreteCountModel::bind_embedded(
            model,
            &state_bytes,
            &expected_state_sha256,
            identity,
        )?;
        let p4 = &mut bound
            .p4
            .as_mut()
            .ok_or_else(|| fail("deployment has no P4 state"))?
            .inner;
        let replacement = self.upload_codes(replacement_codes)?;
        if let Some(codes) = replacement_codes {
            p4.codes.copy_from_slice(codes);
        }
        let prefixes = gather(tokens, samples)?;
        let result = self.launch(
            &prefixes,
            replacement.as_ref().unwrap_or(&self.codes),
            true,
            true,
            true,
        )?;
        let mut session = P4Session::new();
        let mut counterfactual_rows = 0;
        let mut counterfactuals = 0;
        let mut class_sensitive_rows = 0;
        let mut position_sensitive_rows = 0;
        for (i, (&(prefix, target), raw)) in prefixes.iter().zip(result.native).enumerate() {
            check_native(raw, prefix, target, true)?;
            // Cold sync matches the exact cache rebuild after an active window
            // slides. Also replay shorter records token by token for parity.
            if prefix.first() == Some(&0) {
                session.sync_prompt(&prefix[..1])?;
                for &token in &prefix[1..] {
                    session.observe_generated(token)?;
                }
            } else {
                session.sync_prompt(prefix)?;
            }
            let (prediction, trace) = session.step(p4, true, true)?;
            let trace = trace.ok_or_else(|| fail("P4Session did not return its native trace"))?;
            let scores = session.last_scores_q48();
            let cuda_scores = &result.scores[i * P4_VOCAB..(i + 1) * P4_VOCAB];
            if let Some(token) = scores.iter().zip(cuda_scores).position(|(a, b)| a != b) {
                return Err(fail(format!(
                    "P4 Q48 parity failed at sample {:?}, token {token}: CPU {} CUDA {}",
                    samples[i], scores[token], cuda_scores[token]
                )));
            }
            let likelihood_scores: Vec<i64> = scores.iter().map(|&mass| mass as i64).collect();
            let prll =
                lawful_likelihood::masses_from_scores(&likelihood_scores)?[usize::from(target)];
            let native = raw.public(target);
            let s3_mode =
                ["primary", "backoff", "starling", "uniform_miss"].get(raw.s3_mode as usize);
            if prediction != native.top1
                || trace.mass_q48 != native.top1_mass_q48
                || trace.top_ties != raw.top_ties
                || scores[usize::from(target)] != native.target_mass_q48
                || trace.pick_pos.map(|pos| pos as usize) != native.pick_pos
                || trace.pick_distance != native.pick_distance
                || trace.pick_successor != native.pick_successor
                || trace.pick_ties != native.pick_ties
                || trace.pick_class != native.pick_class
                || trace.bucket != native.bucket
                || trace.highest_order != raw.highest_order as usize
                || trace.s3_pkey != raw.s3_pkey
                || u32::from(trace.s3_backoff_key) != raw.s3_backoff
                || s3_mode.copied() != Some(trace.s3_mode.as_str())
                || trace.s2_skip_keys != raw.skip_keys
                || u32::from(trace.s2_skip_hit_mask) != raw.s2_hit
                || u32::from(trace.cache_hit_state) != raw.cache_hit
                || full_vocab_brier_q32(scores, target)? != native.brier_q32
                || prll != native.prll_mass_q48
                || lawful_likelihood::bits_q12(prll)? != native.prll_bits_q12
            {
                return Err(fail(format!(
                    "native P4 trace/metric parity failed at sample {:?}",
                    samples[i]
                )));
            }
            let positions = prefix.len().saturating_sub(2);
            if positions == 0 && result.no_pick[i] != native.brier_q32 {
                return Err(fail("CUDA class-0 row disagrees with no-source P4 Brier"));
            }
            if counterfactual_rows < COUNTERFACTUAL_ROWS {
                let alternate = (1..=positions)
                    .rev()
                    .filter(|&p| Some(prefix[p + 1]) != native.pick_successor)
                    .max_by_key(|&p| (prefix[p + 1] == target, p));
                if let Some(pos) = alternate {
                    let mut cache = CausalSessionCache::new();
                    cache.reset_from_tokens(prefix);
                    if forced_p4_scores(
                        p4,
                        prefix,
                        &cache,
                        native.pick_class,
                        native.pick_successor,
                    )? != scores
                    {
                        return Err(fail(format!(
                            "forced full-vocabulary reference disagrees with native P4 at {:?}",
                            samples[i]
                        )));
                    }
                    let start = result.inputs[i].brier_offset as usize;
                    let mut losses = [0; P5_PICK_CLASSES];
                    for (class, loss) in losses.iter_mut().enumerate() {
                        let (successor, observed) = if class == 0 {
                            (None, result.no_pick[i])
                        } else {
                            let cell = start + (class - 1) * positions + pos - 1;
                            (Some(prefix[pos + 1]), result.pick_brier[cell])
                        };
                        let full = forced_p4_scores(p4, prefix, &cache, class as u8, successor)?;
                        let expected = full_vocab_brier_q32(&full, target)?;
                        if observed != expected {
                            return Err(fail(format!(
                                "P5 counterfactual mismatch at {:?}, class {class}, position {pos}: CPU {expected} CUDA {observed}",
                                samples[i]
                            )));
                        }
                        *loss = expected;
                        counterfactuals += 1;
                    }
                    class_sensitive_rows +=
                        usize::from(losses[1..].windows(2).any(|pair| pair[0] != pair[1]));
                    position_sensitive_rows +=
                        usize::from(losses[usize::from(native.pick_class)] != native.brier_q32);
                    counterfactual_rows += 1;
                }
            }
        }
        Ok(ParityChecks {
            native_rows: samples.len(),
            native_score_words: samples.len() * P4_VOCAB,
            counterfactual_rows,
            counterfactuals,
            counterfactual_score_words: counterfactuals * (P4_VOCAB - 1),
            class_sensitive_rows,
            position_sensitive_rows,
        })
    }
}

/// Diagnostic CPU reference with a specified P4 bucket class and Pick
/// successor. Use the production expert views to rebuild all mixed Q80
/// accumulators, add the supplied point mass there, and clamp only at the
/// final Q48 conversion. This path works without GPU class permutation or
/// reuse of any Brier delta.
fn forced_p4_scores(
    state: &P4State<'_>,
    prefix: &[u16],
    cache: &CausalSessionCache,
    class: u8,
    successor: Option<u16>,
) -> Result<Vec<u64>> {
    let mut s3 = Vec::new();
    let (_, s3_default) = state.s3.fill_sparse_masses_q48(prefix, &mut s3)?;
    let mut s2: [Vec<(u16, u64)>; 3] = std::array::from_fn(|_| Vec::new());
    let mut s2_defaults = [0; 3];
    for (family, key) in skip_keys_for_prefix(prefix).into_iter().enumerate() {
        let (_, default) = state
            .s2_skip
            .fill_sparse_family_q48(family, key, &mut s2[family])?;
        s2_defaults[family] = default;
    }
    let mut cache_masses: [Vec<(u16, u64)>; 4] = std::array::from_fn(|_| Vec::new());
    let (cache_bucket, cache_defaults) = cache.eval_sparse_all(&mut cache_masses);
    let mut hits = Vec::new();
    let (count_bucket, _, totals, default_kn) =
        state
            .counts
            .visit_p4_counts(prefix, class, |token, kn, counts| {
                hits.push((token, kn, *counts))
            })?;
    let weights = state
        .s6
        .bucket_weights(usize::from(cache_bucket) * P4_BASE_BUCKETS + usize::from(count_bucket))?;
    let weighted = |expert: usize, mass: u64| u128::from(weights[expert]) * u128::from(mass);
    let mut base = weighted(0, default_kn) + weighted(12, KN_BASE_MASS);
    let sparse_experts = std::iter::once((s3_default, s3.as_slice()))
        .chain(s2_defaults.into_iter().zip(s2.iter().map(Vec::as_slice)))
        .chain(
            cache_defaults
                .into_iter()
                .zip(cache_masses.iter().map(Vec::as_slice)),
        );
    for (lane, (default, _)) in sparse_experts.clone().enumerate() {
        base += weighted(lane + 14, default);
    }
    let mut accum = vec![base; P4_VOCAB];
    for (id, kn, counts) in hits {
        let token = usize::from(id);
        let mut mass = accum[token] - weighted(0, default_kn) + weighted(0, kn);
        for (order, (count, total)) in counts.into_iter().zip(totals).enumerate() {
            if count != 0 && total != 0 {
                let mle = ((u128::from(count) << 48) / u128::from(total)) as u64;
                mass += weighted(order + 1, mle);
            }
        }
        accum[token] = mass;
    }
    for (lane, (default, masses)) in sparse_experts.enumerate() {
        let default = weighted(lane + 14, default);
        for &(token, mass) in masses {
            let accum = &mut accum[usize::from(token)];
            *accum = *accum - default + weighted(lane + 14, mass);
        }
    }
    if let Some(token) = successor {
        accum[usize::from(token)] += weighted(13, P4_Q48_ONE);
    }
    let mut scores: Vec<u64> = accum
        .into_iter()
        .map(|mass| (mass >> 32).clamp(1, u128::from(P4_Q48_ONE)) as u64)
        .collect();
    scores[0] = 0;
    Ok(scores)
}

fn gather<'a>(tokens: &'a [u16], samples: &[Sample]) -> Result<Vec<(&'a [u16], u16)>> {
    samples
        .iter()
        .map(|&sample| active_prefix(tokens, sample))
        .collect()
}

fn check_native(raw: NativeOutput, prefix: &[u16], target: u16, prll: bool) -> Result<()> {
    let n = raw.public(target);
    let picking = prefix.len() >= 3;
    let legal_pick = match (n.pick_pos, n.pick_successor, n.pick_distance) {
        (Some(pos), Some(id), Some(distance)) => {
            picking
                && pos > 0
                && pos < prefix.len() - 1
                && prefix[pos + 1] == id
                && u32::from(n.pick_class) == 1 + distance.min(3)
        }
        (None, None, None) => !picking && n.pick_class == 0,
        _ => false,
    };
    if !legal_pick
        || raw.prefix_len as usize != prefix.len()
        || raw.top == 0
        || raw.top as usize >= P4_VOCAB
        || raw.target_mass == 0
        || raw.top_mass < raw.target_mass
        || (prll && raw.prll_mass == 0)
    {
        return Err(fail(
            "P4 CUDA returned incomplete or illegal native metadata",
        ));
    }
    Ok(())
}
