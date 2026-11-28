use crate::{fail, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, Ptx, PushKernelArg,
    ValidAsZeroBits,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};
use uor_r4_integer::discrete_d11::{
    native_occurrence::lexical::{Carry, OrderedPrefix, SCORE_UNIT_Q48, WEIGHT_WORDS},
    p4_lexical_bfs,
};

pub const PRIOR_WORDS: usize = 4095;
const VOCAB: usize = PRIOR_WORDS + 1;
const ROOT_OFFSET: usize = 70 + VOCAB;
const CONTEXT: usize = 2048;
const THREADS: u32 = 128;
const ONE_Q48: u64 = 1 << 48;
const CUBIN: &[u8] = include_bytes!(env!("LEXICAL_CUBIN"));
const GENERATED_BFS: &[u8; 256] = include_bytes!(concat!(env!("OUT_DIR"), "/lexical-bfs.bin"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Mode {
    Geometry,
    Ordered,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Row {
    offset: u32,
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Feature {
    pub ordered_states: [u32; 4],
    pub carry_code: u32,
    pub status: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct EpochStats {
    pub rows: u32,
    pub already_strict: u32,
    pub blocked_alias: u32,
    pub updates: u32,
    pub clipped_words: u32,
    pub failed_margin: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Evaluation {
    pub score_q48: i64,
    pub prior_q48: u64,
    pub gold_score_q48: i64,
    pub gold_prior_q48: u64,
    pub rival_score_q48: i64,
    pub rival_prior_q48: u64,
    pub space_score_q48: i64,
    pub space_prior_q48: u64,
    pub token: u32,
    pub top_ties: u32,
    pub gold_rank: u32,
    pub gold_prior_rank: u32,
    pub rival: u32,
    pub frozen_token: u32,
    pub frozen_rival: u32,
    pub carry_code: u32,
    pub alias_blocked: u32,
    pub strongest_alias: u32,
    pub winner_addresses: [u32; 4],
    pub gold_addresses: [u32; 4],
    pub rival_addresses: [u32; 4],
    pub ordered_states: [u32; 4],
}

// SAFETY: all values are repr(C) structs of fixed-width integers and match the
// CUDA size assertions. Zero has valid representation for every field.
unsafe impl DeviceRepr for Row {}
unsafe impl DeviceRepr for Feature {}
unsafe impl ValidAsZeroBits for Feature {}
unsafe impl DeviceRepr for EpochStats {}
unsafe impl ValidAsZeroBits for EpochStats {}
unsafe impl DeviceRepr for Evaluation {}
unsafe impl ValidAsZeroBits for Evaluation {}
const _: [(); 8] = [(); size_of::<Row>()];
const _: [(); 24] = [(); size_of::<Feature>()];
const _: [(); 24] = [(); size_of::<EpochStats>()];
const _: [(); 168] = [(); size_of::<Evaluation>()];

pub struct Capture {
    features: Vec<Feature>,
    device_features: CudaSlice<Feature>,
    priors: CudaSlice<u64>,
}

impl Capture {
    pub fn features(&self) -> &[Feature] {
        &self.features
    }
}

pub struct FitResult {
    pub weights: [i16; WEIGHT_WORDS],
    pub epochs: Vec<EpochStats>,
}

pub struct Gpu {
    stream: Arc<CudaStream>,
    trunk: CudaSlice<u32>,
    capture: CudaFunction,
    fit: CudaFunction,
    evaluate: CudaFunction,
}

fn grid(blocks: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Gpu {
    pub fn new(trunk_words: &[u32], bfs: &[u8]) -> Result<Self> {
        if trunk_words.len() != ROOT_OFFSET + VOCAB
            || trunk_words[0] != 1
            || trunk_words[1..69].iter().any(|&word| word >= 64)
            || trunk_words[69..ROOT_OFFSET].iter().any(|&word| word != 0)
            || trunk_words[ROOT_OFFSET..].iter().any(|&word| word > 0xffff)
            || bfs != GENERATED_BFS
            || bfs.iter().any(|&word| word >= 64)
        {
            return Err(fail("invalid P4 lexical recurrence ABI or shared BFS"));
        }
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        Ok(Self {
            trunk: stream.clone_htod(trunk_words)?,
            stream,
            capture: module.load_function("lexical_capture")?,
            fit: module.load_function("lexical_fit")?,
            evaluate: module.load_function("lexical_evaluate")?,
        })
    }

    pub fn digest(&self) -> String {
        hex::encode(Sha256::digest(CUBIN))
    }

    /// The caller supplies P4 priors from the same active history and original
    /// candidate order 1..4095. No gold input reaches feature capture.
    pub fn capture(&self, prefixes: &[Vec<u16>], priors: &[u64]) -> Result<Capture> {
        if prefixes.is_empty()
            || prefixes.len().checked_mul(PRIOR_WORDS) != Some(priors.len())
            || priors.iter().any(|&mass| !(1..=ONE_Q48).contains(&mass))
        {
            return Err(fail("invalid P4 lexical prior shape or Q48 mass"));
        }
        let n = u32::try_from(prefixes.len())?;
        let mut offsets = Vec::with_capacity(prefixes.len());
        let mut flat = Vec::new();
        for prefix in prefixes {
            if prefix.is_empty()
                || prefix.len() > CONTEXT
                || prefix.last() == Some(&1)
                || prefix.iter().any(|&id| usize::from(id) >= VOCAB)
            {
                return Err(fail("invalid open P4 lexical prefix"));
            }
            offsets.push(Row {
                offset: u32::try_from(flat.len())?,
                len: u32::try_from(prefix.len())?,
            });
            flat.extend(prefix.iter().map(|&id| u32::from(id)));
        }
        let offsets = self.stream.clone_htod(&offsets)?;
        let flat = self.stream.clone_htod(&flat)?;
        let mut device_features = self.stream.alloc_zeros::<Feature>(prefixes.len())?;
        let mut launch = self.stream.launch_builder(&self.capture);
        launch
            .arg(&n)
            .arg(&offsets)
            .arg(&flat)
            .arg(&self.trunk)
            .arg(&mut device_features);
        // SAFETY: all prefix offsets cover the uploaded flattened IDs. The
        // bounded thread set writes exactly one feature per valid input row.
        unsafe { launch.launch(grid(n.div_ceil(THREADS)))? };
        let features = self.stream.clone_dtoh(&device_features)?;
        if features
            .iter()
            .any(|row| row.status != 0 || row.carry_code > 0xffff)
        {
            return Err(fail("P4 lexical recurrence refused a causal row"));
        }
        Ok(Capture {
            features,
            device_features,
            priors: self.stream.clone_htod(priors)?,
        })
    }

    /// Perform sequential updates from zero in the precommitted epoch orders.
    /// One codebook scan per row finds the strongest current nongold candidate.
    pub fn fit(
        &self,
        capture: &Capture,
        targets: &[u16],
        orders: &[Vec<u32>],
        mode: Mode,
    ) -> Result<FitResult> {
        if targets.len() != capture.features.len()
            || targets.is_empty()
            || orders.is_empty()
            || targets
                .iter()
                .any(|&id| !(1..VOCAB).contains(&usize::from(id)))
        {
            return Err(fail("invalid lexical fit rows or target IDs"));
        }
        for order in orders {
            let mut seen = vec![false; targets.len()];
            if order.len() != targets.len()
                || order.iter().any(|&i| {
                    !seen
                        .get_mut(i as usize)
                        .is_some_and(|slot| !std::mem::replace(slot, true))
                })
            {
                return Err(fail("lexical epoch must visit each row once"));
            }
        }
        let n = u32::try_from(targets.len())?;
        let targets = self.stream.clone_htod(targets)?;
        let mut weights = self.stream.alloc_zeros::<i16>(WEIGHT_WORDS)?;
        let mut stats = self.stream.alloc_zeros::<EpochStats>(1)?;
        let mut epochs = Vec::with_capacity(orders.len());
        let mode = mode as u32;
        for order in orders {
            let order = self.stream.clone_htod(order)?;
            let mut launch = self.stream.launch_builder(&self.fit);
            launch
                .arg(&n)
                .arg(&order)
                .arg(&targets)
                .arg(&capture.device_features)
                .arg(&capture.priors)
                .arg(&self.trunk)
                .arg(&mode)
                .arg(&mut weights)
                .arg(&mut stats);
            // SAFETY: order is a permutation of the uploaded row range and
            // one block exclusively owns the epoch and signed weight state.
            unsafe { launch.launch(grid(1))? };
            let words = self.stream.clone_dtoh(&stats)?;
            let epoch = *words
                .first()
                .ok_or_else(|| fail("missing GPU epoch stats"))?;
            if epoch.rows != n || epoch.already_strict + epoch.blocked_alias + epoch.updates != n {
                return Err(fail("device lexical epoch counters do not cover every row"));
            }
            epochs.push(epoch);
        }
        let weights = self
            .stream
            .clone_dtoh(&weights)?
            .try_into()
            .map_err(|_| fail("device returned the wrong lexical weight count"))?;
        Ok(FitResult { weights, epochs })
    }

    pub fn evaluate(
        &self,
        capture: &Capture,
        targets: Option<&[u16]>,
        mode: Mode,
        weights: &[i16; WEIGHT_WORDS],
    ) -> Result<Vec<Evaluation>> {
        let n = capture.features.len();
        let absent;
        let targets = match targets {
            Some(targets)
                if targets.len() == n
                    && targets
                        .iter()
                        .all(|&id| (1..VOCAB).contains(&usize::from(id))) =>
            {
                targets
            }
            Some(_) => return Err(fail("invalid labelled lexical replay targets")),
            None => {
                absent = vec![0; n];
                &absent
            }
        };
        let targets = self.stream.clone_htod(targets)?;
        let weights = self.stream.clone_htod(weights)?;
        let mode = mode as u32;
        let mut out = self.stream.alloc_zeros::<Evaluation>(n)?;
        let mut launch = self.stream.launch_builder(&self.evaluate);
        launch
            .arg(&targets)
            .arg(&capture.device_features)
            .arg(&capture.priors)
            .arg(&self.trunk)
            .arg(&mode)
            .arg(&weights)
            .arg(&mut out);
        // SAFETY: one block writes one result and the capture owns validated
        // feature and mass ranges. Optional target zero denotes blind replay.
        unsafe { launch.launch(grid(u32::try_from(n)?))? };
        let out = self.stream.clone_dtoh(&out)?;
        if out
            .iter()
            .any(|row| !(1..VOCAB as u32).contains(&row.token))
        {
            return Err(fail("device lexical replay returned an illegal winner"));
        }
        Ok(out)
    }
}

pub fn finite() -> Result<()> {
    let mut trunk = vec![0; ROOT_OFFSET + VOCAB];
    trunk[0] = 1;
    trunk[ROOT_OFFSET + 11] = 1;
    let gpu = Gpu::new(&trunk, p4_lexical_bfs())?;
    let prefixes = vec![vec![0, 7], vec![99, 0, 7]];
    let priors: Vec<u64> = (0..2)
        .flat_map(|_| {
            let mut prior = vec![1; PRIOR_WORDS];
            prior[9] = 18 * SCORE_UNIT_Q48 as u64;
            prior[10] = 17 * SCORE_UNIT_Q48 as u64;
            prior
        })
        .collect();
    let capture = gpu.capture(&prefixes, &priors)?;
    if capture.features()[0] != capture.features()[1] {
        return Err(fail(
            "finite P4 lexical capture did not reset at the last BOS",
        ));
    }
    let zero = [0; WEIGHT_WORDS];
    let base = gpu.evaluate(&capture, Some(&[11, 11]), Mode::Geometry, &zero)?;
    let blind = gpu.evaluate(&capture, None, Mode::Geometry, &zero)?;
    let fit = gpu.fit(&capture, &[11, 11], &[vec![0, 1]], Mode::Geometry)?;
    let learned = gpu.evaluate(&capture, Some(&[11, 11]), Mode::Geometry, &fit.weights)?;
    let carry = Carry::from_packed(capture.features()[0].carry_code as u16);
    let gold = carry.addresses([1, 0, 0, 0])?.map(u32::from);
    let rival = carry.addresses([0; 4])?.map(u32::from);
    if base[0].token != 10
        || base[0].gold_rank != 2
        || blind[0].token != 10
        || blind[0].gold_rank != 0
        || learned[0].token != 11
        || learned[0].gold_rank != 1
        || learned[0].gold_addresses != gold
        || learned[0].rival_addresses != rival
        || fit.weights[gold[0] as usize] != 1
        || fit.weights[rival[0] as usize] != -1
        || fit.epochs[0].updates != 1
        || fit.epochs[0].already_strict != 1
        || fit.epochs[0].failed_margin != 0
    {
        return Err(fail("finite lexical signed reversal failed"));
    }
    let inverted = fit.weights.map(|word| -word);
    let reversed = gpu.evaluate(&capture, Some(&[11, 11]), Mode::Geometry, &inverted)?;
    if reversed[0].token != 10 || reversed[0].gold_score_q48 >= base[0].gold_score_q48 {
        return Err(fail("finite opposite signed reversal failed"));
    }

    let mut swapped_priors = priors[..PRIOR_WORDS].to_vec();
    swapped_priors.swap(9, 10);
    let swapped = gpu.capture(&prefixes[..1], &swapped_priors)?;
    let swapped_fit = gpu.fit(&swapped, &[10], &[vec![0]], Mode::Geometry)?;
    let swapped_base = gpu.evaluate(&swapped, Some(&[10]), Mode::Geometry, &zero)?;
    let swapped_learned =
        gpu.evaluate(&swapped, Some(&[10]), Mode::Geometry, &swapped_fit.weights)?;
    if swapped_base[0].token != 11
        || swapped_learned[0].token != 10
        || swapped_fit.weights != inverted
    {
        return Err(fail(
            "finite reciprocal fit did not learn the opposite signed words",
        ));
    }

    let ordered = gpu.evaluate(&capture, Some(&[11, 11]), Mode::Ordered, &zero)?;
    let oracle = OrderedPrefix::new(&[0, 7])?;
    if ordered[0].token != base[0].token
        || ordered[0].prior_q48 != base[0].prior_q48
        || ordered[0].gold_addresses != oracle.addresses(11)?.map(u32::from)
        || ordered[0].rival_addresses != oracle.addresses(10)?.map(u32::from)
        || ordered[0].ordered_states != oracle.states()
    {
        return Err(fail("finite ordered suffix or zero-prior parity failed"));
    }
    let ordered_fit = gpu.fit(&capture, &[11, 11], &[vec![0, 1]], Mode::Ordered)?;
    let ordered_learned = gpu.evaluate(
        &capture,
        Some(&[11, 11]),
        Mode::Ordered,
        &ordered_fit.weights,
    )?;
    if ordered_learned[0].token != 11 || ordered_fit.epochs[0].failed_margin != 0 {
        return Err(fail(
            "finite ordered control did not learn the planted reversal",
        ));
    }
    let tied = gpu.capture(&[vec![0, 7]], &vec![1; PRIOR_WORDS])?;
    let tie = gpu.evaluate(&tied, Some(&[2]), Mode::Geometry, &zero)?;
    let blocked = gpu.fit(&tied, &[2], &[vec![0]], Mode::Geometry)?;
    if tie[0].token != 1
        || tie[0].top_ties != PRIOR_WORDS as u32
        || tie[0].gold_rank != 2
        || blocked.epochs[0].blocked_alias != 1
        || blocked.weights != zero
    {
        return Err(fail(
            "finite original-ID ties or exact alias were not preserved",
        ));
    }

    let mut saturated = vec![1; PRIOR_WORDS];
    saturated[9] = ONE_Q48;
    let saturation = gpu.capture(&[vec![0, 7]], &saturated)?;
    let result = gpu.fit(&saturation, &[11], &[vec![0]], Mode::Geometry)?;
    if result.epochs[0].clipped_words != 1
        || result.epochs[0].failed_margin != 1
        || result.weights[gold[0] as usize] != i16::MAX
        || result.weights[rival[0] as usize] != i16::MIN
    {
        return Err(fail("finite P4 lexical signed saturation was not observed"));
    }

    let long = (0..CONTEXT)
        .map(|i| if i == 0 { 0 } else { 6 + (i % 23) as u16 })
        .collect::<Vec<_>>();
    let long_capture = gpu.capture(std::slice::from_ref(&long), &vec![1; PRIOR_WORDS])?;
    let long_score = gpu.evaluate(&long_capture, Some(&[2]), Mode::Ordered, &zero)?;
    let oracle = OrderedPrefix::new(&long)?;
    if long_score[0].gold_addresses != oracle.addresses(2)?.map(u32::from)
        || long_score[0].ordered_states != oracle.states()
    {
        return Err(fail("finite 2,048-ID ordered prefix parity failed"));
    }
    if gpu.fit(&tied, &[2], &[vec![1]], Mode::Geometry).is_ok() {
        return Err(fail("finite invalid row permutation was admitted"));
    }
    Ok(())
}
