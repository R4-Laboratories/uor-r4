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
const CUBIN: &[u8] = include_bytes!(env!("RETENTION_CUBIN"));
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
pub struct Decision {
    pub margin_q48: i64,
    pub rival_prior_q48: u64,
    pub rival: u32,
    pub rank: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Visit {
    pub before: Decision,
    pub old_rival_margin_q48: i64,
    pub after: Decision,
    pub clipped_words: u32,
    #[serde(skip)]
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Feasibility {
    pub ceiling: Decision,
    pub prior_rank: u32,
    pub strongest_alias: u32,
    pub alias_blocked: u32,
    pub alias_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Witness {
    pub decision: Decision,
    pub winner_addresses: [u32; 4],
    pub winner_prior_q48: u64,
    pub winner_score_q48: i64,
    pub top_ties: u32,
    pub winner: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AddressMap {
    counts: [u32; 4],
    reciprocals: [u32; 4],
    slots: [u32; WEIGHT_WORDS],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Support {
    pub geometry: [u64; 4],
    pub ordered: [u64; 4],
}

impl AddressMap {
    fn new(masks: [u64; 4]) -> Result<Self> {
        let mut map = Self {
            counts: [0; 4],
            reciprocals: [0; 4],
            slots: [0; WEIGHT_WORDS],
        };
        for (bank, mask) in masks.into_iter().enumerate() {
            let count = mask.count_ones();
            if count == 0 {
                return Err(fail("empty geometry support bank"));
            }
            map.counts[bank] = count;
            if count > 1 {
                map.reciprocals[bank] = ((1u64 << 32) / u64::from(count)) as u32;
            }
            for (i, slot) in (0..64)
                .filter(|slot| mask & (1u64 << slot) != 0)
                .enumerate()
            {
                map.slots[64 * bank + i] = slot;
            }
        }
        Ok(map)
    }
}

// SAFETY: these repr(C) structs have matching CUDA size assertions and all
// fields admit zero. Visit explicitly reserves its trailing alignment word.
unsafe impl DeviceRepr for Row {}
unsafe impl DeviceRepr for Feature {}
unsafe impl ValidAsZeroBits for Feature {}
unsafe impl DeviceRepr for EpochStats {}
unsafe impl ValidAsZeroBits for EpochStats {}
unsafe impl DeviceRepr for Decision {}
unsafe impl ValidAsZeroBits for Decision {}
unsafe impl DeviceRepr for Visit {}
unsafe impl ValidAsZeroBits for Visit {}
unsafe impl DeviceRepr for Feasibility {}
unsafe impl ValidAsZeroBits for Feasibility {}
unsafe impl DeviceRepr for Witness {}
unsafe impl ValidAsZeroBits for Witness {}
unsafe impl DeviceRepr for AddressMap {}
const _: [(); 8] = [(); size_of::<Row>()];
const _: [(); 24] = [(); size_of::<Feature>()];
const _: [(); 24] = [(); size_of::<EpochStats>()];
const _: [(); 24] = [(); size_of::<Decision>()];
const _: [(); 64] = [(); size_of::<Visit>()];
const _: [(); 40] = [(); size_of::<Feasibility>()];
const _: [(); 64] = [(); size_of::<Witness>()];
const _: [(); 1056] = [(); size_of::<AddressMap>()];

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

#[derive(Serialize)]
pub struct Epoch {
    pub stats: EpochStats,
    #[serde(serialize_with = "serialize_weights")]
    pub weights: [i16; WEIGHT_WORDS],
    pub visits: Vec<Visit>,
    pub checkpoint: Vec<Decision>,
}

#[derive(Serialize)]
pub struct FitResult {
    #[serde(serialize_with = "serialize_weights")]
    pub weights: [i16; WEIGHT_WORDS],
    pub epochs: Vec<Epoch>,
}

fn serialize_weights<S: serde::Serializer>(
    weights: &[i16; WEIGHT_WORDS],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    weights.as_slice().serialize(serializer)
}

pub struct Gpu {
    stream: Arc<CudaStream>,
    trunk: CudaSlice<u32>,
    capture: CudaFunction,
    support: CudaFunction,
    reference_fit: CudaFunction,
    fit: CudaFunction,
    checkpoint: CudaFunction,
    feasibility: CudaFunction,
    address_probe: CudaFunction,
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
            support: module.load_function("retention_support")?,
            reference_fit: module.load_function("lexical_fit")?,
            fit: module.load_function("retention_fit")?,
            checkpoint: module.load_function("retention_checkpoint")?,
            feasibility: module.load_function("retention_feasibility")?,
            address_probe: module.load_function("retention_address_probe")?,
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

    /// Fit label-free support using only causal states and original IDs.
    pub fn support(&self, capture: &Capture) -> Result<Support> {
        // The geometry variant never reads the map. A valid dummy keeps the
        // shared support kernel's ABI the same in both variants.
        let dummy = self.stream.clone_htod(&[AddressMap::new([1; 4])?])?;
        let geometry = self.support_masks(capture, Mode::Geometry, &dummy)?;
        let map = self.stream.clone_htod(&[AddressMap::new(geometry)?])?;
        let ordered = self.support_masks(capture, Mode::Ordered, &map)?;
        Ok(Support { geometry, ordered })
    }

    fn support_masks(
        &self,
        capture: &Capture,
        mode: Mode,
        map: &CudaSlice<AddressMap>,
    ) -> Result<[u64; 4]> {
        let n = u32::try_from(capture.features.len())?;
        let mode = mode as u32;
        let mut masks = self.stream.alloc_zeros::<u64>(4)?;
        let mut launch = self.stream.launch_builder(&self.support);
        launch
            .arg(&capture.device_features)
            .arg(&self.trunk)
            .arg(&mode)
            .arg(map)
            .arg(&mut masks);
        // SAFETY: one block per feature and four initialized atomic masks.
        unsafe { launch.launch(grid(n))? };
        self.stream
            .clone_dtoh(&masks)?
            .try_into()
            .map_err(|_| fail("device returned the wrong support mask count"))
    }

    fn map(&self, support: &Support) -> Result<CudaSlice<AddressMap>> {
        if support.geometry != support.ordered {
            return Err(fail(
                "ordered addresses did not fill matched geometry support",
            ));
        }
        Ok(self
            .stream
            .clone_htod(&[AddressMap::new(support.geometry)?])?)
    }

    fn validated_targets(&self, capture: &Capture, targets: &[u16]) -> Result<CudaSlice<u16>> {
        if targets.len() != capture.features.len()
            || targets
                .iter()
                .any(|&id| !(1..VOCAB).contains(&usize::from(id)))
        {
            return Err(fail("invalid lexical capture targets"));
        }
        Ok(self.stream.clone_htod(targets)?)
    }

    pub fn fit(
        &self,
        capture: &Capture,
        targets: &[u16],
        orders: &[Vec<u32>],
        mode: Mode,
        support: &Support,
    ) -> Result<FitResult> {
        if orders.is_empty() {
            return Err(fail("empty lexical fit orders"));
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
        let device_targets = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let mut weights = self.stream.alloc_zeros::<i16>(WEIGHT_WORDS)?;
        let mut stats = self.stream.alloc_zeros::<EpochStats>(1)?;
        let mut visits = self.stream.alloc_zeros::<Visit>(targets.len())?;
        let mut epochs = Vec::with_capacity(orders.len());
        let mode = mode as u32;
        for order in orders {
            let order = self.stream.clone_htod(order)?;
            let mut launch = self.stream.launch_builder(&self.fit);
            launch
                .arg(&n)
                .arg(&order)
                .arg(&device_targets)
                .arg(&capture.device_features)
                .arg(&capture.priors)
                .arg(&self.trunk)
                .arg(&mode)
                .arg(&map)
                .arg(&mut weights)
                .arg(&mut stats)
                .arg(&mut visits);
            // SAFETY: order is a checked permutation of the capture and one
            // block exclusively owns the sequential epoch and weight state.
            unsafe { launch.launch(grid(1))? };
            let epoch_words = self.read_weights(&weights)?;
            if epoch_words.iter().enumerate().any(|(word, &value)| {
                value != 0 && support.geometry[word / 64] & (1u64 << (word % 64)) == 0
            }) {
                return Err(fail("lexical fit wrote outside the matched support"));
            }
            let stats = self.read_stats(&stats, n)?;
            let visits = self.stream.clone_dtoh(&visits)?;
            let mut observed_clips = 0u64;
            let mut observed_strict = 0u32;
            for (visit, &gold) in visits.iter().zip(targets) {
                validate_decision(visit.before, gold)?;
                validate_decision(visit.after, gold)?;
                observed_clips += u64::from(visit.clipped_words);
                observed_strict += u32::from(visit.before.margin_q48 > 0);
                if visit.reserved != 0
                    || visit.clipped_words > 8
                    || visit.old_rival_margin_q48 < visit.after.margin_q48
                    || (visit.before.rival == visit.after.rival
                        && visit.old_rival_margin_q48 != visit.after.margin_q48)
                    || (visit.before.margin_q48 > 0
                        && (visit.clipped_words != 0 || visit.before != visit.after))
                {
                    return Err(fail("inconsistent device lexical visit"));
                }
            }
            if observed_clips != u64::from(stats.clipped_words)
                || observed_strict != stats.already_strict
            {
                return Err(fail("device lexical traces disagree with epoch counters"));
            }
            let (checkpoint, _) = self.scan_checkpoint(
                capture,
                targets,
                &device_targets,
                mode,
                &map,
                &weights,
                false,
            )?;
            epochs.push(Epoch {
                stats,
                weights: epoch_words,
                visits,
                checkpoint,
            });
        }
        let weights = epochs
            .last()
            .ok_or_else(|| fail("missing device fit epochs"))?
            .weights;
        Ok(FitResult { weights, epochs })
    }

    fn read_stats(&self, stats: &CudaSlice<EpochStats>, n: u32) -> Result<EpochStats> {
        let words = self.stream.clone_dtoh(stats)?;
        let epoch = *words
            .first()
            .ok_or_else(|| fail("missing GPU epoch stats"))?;
        if epoch.rows != n
            || u64::from(epoch.already_strict)
                + u64::from(epoch.blocked_alias)
                + u64::from(epoch.updates)
                != u64::from(n)
            || epoch.failed_margin > epoch.updates
        {
            return Err(fail("device lexical epoch counters do not cover every row"));
        }
        Ok(epoch)
    }

    fn read_weights(&self, weights: &CudaSlice<i16>) -> Result<[i16; WEIGHT_WORDS]> {
        self.stream
            .clone_dtoh(weights)?
            .try_into()
            .map_err(|_| fail("device returned the wrong lexical weight count"))
    }

    pub fn evaluate(
        &self,
        capture: &Capture,
        targets: &[u16],
        mode: Mode,
        support: &Support,
        weights: &[i16; WEIGHT_WORDS],
    ) -> Result<Vec<Decision>> {
        let device_targets = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let weights = self.stream.clone_htod(weights)?;
        let (decisions, _) = self.scan_checkpoint(
            capture,
            targets,
            &device_targets,
            mode as u32,
            &map,
            &weights,
            false,
        )?;
        Ok(decisions)
    }

    pub fn witness(
        &self,
        capture: &Capture,
        targets: &[u16],
        mode: Mode,
        support: &Support,
        weights: &[i16; WEIGHT_WORDS],
    ) -> Result<Vec<Witness>> {
        let device_targets = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let weights = self.stream.clone_htod(weights)?;
        let (_, witnesses) = self.scan_checkpoint(
            capture,
            targets,
            &device_targets,
            mode as u32,
            &map,
            &weights,
            true,
        )?;
        Ok(witnesses)
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_checkpoint(
        &self,
        capture: &Capture,
        targets: &[u16],
        device_targets: &CudaSlice<u16>,
        mode: u32,
        map: &CudaSlice<AddressMap>,
        weights: &CudaSlice<i16>,
        witness: bool,
    ) -> Result<(Vec<Decision>, Vec<Witness>)> {
        let n = targets.len();
        let write_witness = u32::from(witness);
        let mut decisions = self
            .stream
            .alloc_zeros::<Decision>(if witness { 1 } else { n })?;
        let mut witnesses = self
            .stream
            .alloc_zeros::<Witness>(if witness { n } else { 1 })?;
        let mut launch = self.stream.launch_builder(&self.checkpoint);
        launch
            .arg(device_targets)
            .arg(&capture.device_features)
            .arg(&capture.priors)
            .arg(&self.trunk)
            .arg(&mode)
            .arg(map)
            .arg(weights)
            .arg(&write_witness)
            .arg(&mut decisions)
            .arg(&mut witnesses);
        // SAFETY: disjoint row outputs with the selected output buffer sized
        // for the full launch. The unused buffer is never read or written.
        unsafe { launch.launch(grid(u32::try_from(n)?))? };
        if witness {
            let witnesses = self.stream.clone_dtoh(&witnesses)?;
            for (item, &gold) in witnesses.iter().zip(targets) {
                validate_decision(item.decision, gold)?;
                let winner = if item.decision.rank == 1 {
                    u32::from(gold)
                } else {
                    item.decision.rival
                };
                if item.winner != winner
                    || !(1..=PRIOR_WORDS as u32).contains(&item.top_ties)
                    || !(1..=ONE_Q48).contains(&item.winner_prior_q48)
                    || item
                        .winner_addresses
                        .iter()
                        .enumerate()
                        .any(|(bank, &address)| {
                            !(bank * 64..(bank + 1) * 64).contains(&(address as usize))
                        })
                {
                    return Err(fail("invalid device lexical witness"));
                }
            }
            Ok((Vec::new(), witnesses))
        } else {
            let decisions = self.stream.clone_dtoh(&decisions)?;
            for (&item, &gold) in decisions.iter().zip(targets) {
                validate_decision(item, gold)?;
            }
            Ok((decisions, Vec::new()))
        }
    }

    pub fn feasibility(
        &self,
        capture: &Capture,
        targets: &[u16],
        mode: Mode,
        support: &Support,
    ) -> Result<Vec<Feasibility>> {
        let n = targets.len();
        let device_targets = self.validated_targets(capture, targets)?;
        let map = self.map(support)?;
        let mode = mode as u32;
        let mut output = self.stream.alloc_zeros::<Feasibility>(n)?;
        let mut launch = self.stream.launch_builder(&self.feasibility);
        launch
            .arg(&device_targets)
            .arg(&capture.device_features)
            .arg(&capture.priors)
            .arg(&self.trunk)
            .arg(&mode)
            .arg(&map)
            .arg(&mut output);
        // SAFETY: the checked capture and target arrays each contain n rows.
        unsafe { launch.launch(grid(u32::try_from(n)?))? };
        let output = self.stream.clone_dtoh(&output)?;
        for (item, &gold) in output.iter().zip(targets) {
            validate_decision(item.ceiling, gold)?;
            if !(1..=PRIOR_WORDS as u32).contains(&item.prior_rank)
                || item.alias_blocked > 1
                || item.alias_count >= PRIOR_WORDS as u32
                || (item.strongest_alias == 0) != (item.alias_count == 0)
                || item.strongest_alias == u32::from(gold)
                || item.strongest_alias >= VOCAB as u32
                || item.alias_blocked == 1 && (item.alias_count == 0 || item.ceiling.rank == 1)
            {
                return Err(fail("invalid device lexical feasibility"));
            }
        }
        Ok(output)
    }
}

fn validate_decision(item: Decision, gold: u16) -> Result<()> {
    let gold = u32::from(gold);
    let elects_gold = item.margin_q48 > 0 || (item.margin_q48 == 0 && gold < item.rival);
    if !(1..=PRIOR_WORDS as u32).contains(&item.rival)
        || !(1..=PRIOR_WORDS as u32).contains(&item.rank)
        || item.rival == gold
        || !(1..=ONE_Q48).contains(&item.rival_prior_q48)
        || elects_gold != (item.rank == 1)
    {
        return Err(fail("invalid device lexical election"));
    }
    Ok(())
}

fn fixture_priors(overrides: &[(u16, u64)]) -> Vec<u64> {
    let mut priors = vec![1; PRIOR_WORDS];
    for &(id, mass) in overrides {
        priors[usize::from(id - 1)] = mass;
    }
    priors
}

fn fixture_trunk(roots: &[(u16, u32)]) -> Vec<u32> {
    let mut trunk = vec![0; ROOT_OFFSET + VOCAB];
    trunk[0] = 1;
    for &(id, root) in roots {
        trunk[ROOT_OFFSET + usize::from(id)] = root;
    }
    trunk
}

fn finite_reference(
    gpu: &Gpu,
    capture: &Capture,
    targets: &[u16],
    orders: &[Vec<u32>],
    fitted: &FitResult,
) -> Result<()> {
    let n = u32::try_from(targets.len())?;
    let targets = gpu.validated_targets(capture, targets)?;
    let geometry = Mode::Geometry as u32;
    let mut weights = gpu.stream.alloc_zeros::<i16>(WEIGHT_WORDS)?;
    let mut stats = gpu.stream.alloc_zeros::<EpochStats>(1)?;
    for (order, recorded) in orders.iter().zip(&fitted.epochs) {
        let order = gpu.stream.clone_htod(order)?;
        let mut launch = gpu.stream.launch_builder(&gpu.reference_fit);
        launch
            .arg(&n)
            .arg(&order)
            .arg(&targets)
            .arg(&capture.device_features)
            .arg(&capture.priors)
            .arg(&gpu.trunk)
            .arg(&geometry)
            .arg(&mut weights)
            .arg(&mut stats);
        // SAFETY: the instrumented fit already validated the identical
        // row permutations and targets. The original kernel owns the state.
        unsafe { launch.launch(grid(1))? };
        if gpu.read_weights(&weights)? != recorded.weights
            || gpu.read_stats(&stats, n)? != recorded.stats
        {
            return Err(fail(
                "finite instrumented fit diverged from frozen lexical fit",
            ));
        }
    }
    Ok(())
}

fn finite_ordered(gpu: &Gpu) -> Result<()> {
    let canonical_mask = p4_lexical_bfs()
        .iter()
        .fold(0u64, |mask, &slot| mask | (1u64 << slot));
    let map = gpu
        .stream
        .clone_htod(&[AddressMap::new([canonical_mask; 4])?])?;
    let canonical = Support {
        geometry: [canonical_mask; 4],
        ordered: [canonical_mask; 4],
    };
    let weights = std::array::from_fn(|i| (i as i16 % 19) - 9);
    let gold = 11;
    let long = (0..CONTEXT)
        .map(|i| if i == 0 { 0 } else { 6 + (i % 23) as u16 })
        .collect::<Vec<_>>();
    for prefix in [&[99, 0, 7][..], &long[..]] {
        let capture = gpu.capture(&[prefix.to_vec()], &vec![1; PRIOR_WORDS])?;
        let active = &prefix[prefix.iter().rposition(|&id| id == 0).unwrap_or(0)..];
        let oracle = OrderedPrefix::new(active)?;
        let mut output = gpu.stream.alloc_zeros::<u32>(4 * PRIOR_WORDS)?;
        let mut launch = gpu.stream.launch_builder(&gpu.address_probe);
        launch
            .arg(&capture.device_features)
            .arg(&gpu.trunk)
            .arg(&map)
            .arg(&mut output);
        // SAFETY: the probe uses the first captured feature and writes all
        // 4,095 four-word addresses to the initialized output range.
        unsafe { launch.launch(grid(1))? };
        let addresses = gpu.stream.clone_dtoh(&output)?;
        if capture.features()[0].ordered_states != oracle.states() {
            return Err(fail(
                "finite ordered capture changed the original suffix states",
            ));
        }
        for (index, actual) in addresses.chunks_exact(4).enumerate() {
            let expected = oracle
                .matched_addresses(u16::try_from(index + 1)?)?
                .map(u32::from);
            if actual != expected {
                return Err(fail(
                    "finite matched ordered device addresses diverged from production",
                ));
            }
        }
        let mut scores = Vec::with_capacity(PRIOR_WORDS);
        for id in 1..VOCAB as u16 {
            let sum: i64 = oracle
                .matched_addresses(id)?
                .into_iter()
                .map(|index| i64::from(weights[usize::from(index)]))
                .sum();
            scores.push((1 + sum * SCORE_UNIT_Q48, std::cmp::Reverse(id)));
        }
        let gold_score = scores[usize::from(gold - 1)].0;
        scores.sort_unstable_by(|a, b| b.cmp(a));
        let gold_rank = scores
            .iter()
            .position(|&(_, id)| id.0 == gold)
            .ok_or_else(|| fail("finite ordered oracle lost the target"))?
            + 1;
        let (rival_score, rival) = scores[usize::from(gold_rank == 1)];
        let (winning_score, winner) = scores[0];
        let ties = scores
            .iter()
            .take_while(|score| score.0 == winning_score)
            .count();
        let observed = gpu.witness(&capture, &[gold], Mode::Ordered, &canonical, &weights)?[0];
        let expected = Witness {
            decision: Decision {
                margin_q48: gold_score - rival_score,
                rival_prior_q48: 1,
                rival: u32::from(rival.0),
                rank: u32::try_from(gold_rank)?,
            },
            winner_addresses: oracle.matched_addresses(winner.0)?.map(u32::from),
            winner_prior_q48: 1,
            winner_score_q48: winning_score,
            top_ties: u32::try_from(ties)?,
            winner: u32::from(winner.0),
        };
        if observed != expected {
            return Err(fail(
                "finite ordered full election disagreed with the production addresses",
            ));
        }
    }
    Ok(())
}

pub fn finite() -> Result<()> {
    let unit = SCORE_UNIT_Q48 as u64;
    let base = 100 * unit;
    let prefixes = [vec![0, 7], vec![99, 0, 7]];
    let zero = [0; WEIGHT_WORDS];

    let trunk = fixture_trunk(&[(7, 0x10), (8, 0x11), (10, 0x11), (12, 0x01)]);
    let gpu = Gpu::new(&trunk, p4_lexical_bfs())?;
    let geometry = Mode::Geometry;
    let priors = [
        fixture_priors(&[(9, base), (8, base + unit), (7, base)]),
        fixture_priors(&[(10, base), (11, base + unit), (12, base)]),
    ]
    .concat();
    let capture = gpu.capture(&prefixes, &priors)?;
    if capture.features()[0] != capture.features()[1] {
        return Err(fail(
            "finite P4 lexical capture did not reset at the last BOS",
        ));
    }
    let support = gpu.support(&capture)?;
    if support.geometry != support.ordered || support.geometry.map(u64::count_ones) != [2, 2, 1, 1]
    {
        return Err(fail("finite label-free matched support was not filled"));
    }
    let targets = [9, 10];
    let orders = [vec![0, 1], vec![1, 0]];
    let fit = gpu.fit(&capture, &targets, &orders, geometry, &support)?;
    finite_reference(&gpu, &capture, &targets, &orders, &fit)?;
    let first = fit.epochs[0].visits[0];
    let erased = fit.epochs[0].checkpoint[0];
    let unit_i = SCORE_UNIT_Q48;
    if first.before.rival != 8
        || first.before.rank != 3
        || first.after.rival != 7
        || first.after.rank != 1
        || first.after.margin_q48 != 2 * unit_i
        || erased.rival != 8
        || erased.rank != 3
        || erased.margin_q48 != -5 * unit_i
    {
        return Err(fail(
            "finite subsequent update did not erase the earlier all-ID win",
        ));
    }
    let witness = gpu.witness(
        &capture,
        &targets,
        geometry,
        &support,
        &fit.epochs[0].weights,
    )?;
    let carry = Carry::from_packed(capture.features()[0].carry_code as u16);
    let rival_addresses = carry.addresses([1, 1, 0, 0])?.map(u32::from);
    if witness[0].decision != erased
        || witness[0].winner != 8
        || witness[0].winner_addresses != rival_addresses
        || witness[0].winner_prior_q48 != base + unit
        || witness[0].top_ties != 1
    {
        return Err(fail("finite witness disagreed with the frozen checkpoint"));
    }

    let replacement = fixture_priors(&[(9, base), (8, base + 6 * unit), (7, base + 5 * unit)]);
    let replacement = gpu.capture(&prefixes[..1], &replacement)?;
    let fit = gpu.fit(&replacement, &[9], &[vec![0]], geometry, &support)?;
    let visit = fit.epochs[0].visits[0];
    if visit.before.rival != 8
        || visit.before.margin_q48 != -6 * unit_i
        || visit.old_rival_margin_q48 != 2 * unit_i
        || visit.after.rival != 7
        || visit.after.margin_q48 != -unit_i
        || visit.after.rank != 2
        || visit.clipped_words != 0
    {
        return Err(fail("finite new full-election rival was not exposed"));
    }

    let tied = gpu.capture(&prefixes[..1], &vec![1; PRIOR_WORDS])?;
    let feasibility = gpu.feasibility(&tied, &[2], geometry, &support)?;
    let ties = gpu.witness(&tied, &[2], geometry, &support, &zero)?;
    let blocked = gpu.fit(&tied, &[2], &[vec![0]], geometry, &support)?;
    let later_id = gpu.feasibility(&tied, &[1], geometry, &support)?;
    if feasibility[0].ceiling.margin_q48 != 0
        || feasibility[0].ceiling.rival != 1
        || feasibility[0].ceiling.rank != 2
        || feasibility[0].prior_rank != 2
        || feasibility[0].alias_count != 4090
        || feasibility[0].alias_blocked != 1
        || later_id[0].ceiling.margin_q48 != 0
        || later_id[0].ceiling.rank != 1
        || later_id[0].alias_blocked != 0
        || ties[0].winner != 1
        || ties[0].top_ties != PRIOR_WORDS as u32
        || blocked.epochs[0].stats.blocked_alias != 1
        || blocked.weights != zero
    {
        return Err(fail(
            "finite full-address aliases or original-ID ties changed",
        ));
    }

    let saturated = fixture_priors(&[(9, 1), (12, ONE_Q48)]);
    let saturated = gpu.capture(&prefixes[..1], &saturated)?;
    let fit = gpu.fit(&saturated, &[9], &[vec![0]], geometry, &support)?;
    let cap = gpu.feasibility(&saturated, &[9], geometry, &support)?[0];
    let at_limit = fit.epochs[0].visits[0];
    let gold_addresses = carry.addresses([0; 4])?.map(usize::from);
    let low_addresses = carry.addresses([1, 0, 0, 0])?.map(usize::from);
    if fit.epochs[0].stats.failed_margin != 1
        || at_limit.clipped_words != 1
        || at_limit.old_rival_margin_q48 != 1 - unit_i
        // Token 7 still ties the gold after the one-bank update. Moving the
        // second bank to its row-local extremum breaks that tie. Token 12
        // sets the same worst margin in both elections.
        || at_limit.after.margin_q48 != cap.ceiling.margin_q48
        || cap.ceiling.margin_q48 != 1 - unit_i
        || at_limit.after.rival != 12
        || cap.ceiling.rival != 12
        || at_limit.after.rank != 9
        || cap.ceiling.rank != 8
        || fit.weights[gold_addresses[0]] != i16::MAX
        || fit.weights[low_addresses[0]] != i16::MIN
    {
        return Err(fail(format!(
            "finite saturation disagreed with the row oracle: visit={at_limit:?}, ceiling={cap:?}"
        )));
    }

    if gpu.evaluate(&capture, &targets, geometry, &support, &zero)?
        != gpu.evaluate(&capture, &targets, Mode::Ordered, &support, &zero)?
        || gpu.fit(&tied, &[2], &[vec![1]], geometry, &support).is_ok()
    {
        return Err(fail(
            "finite zero-arm parity or row permutation validation failed",
        ));
    }
    finite_ordered(&gpu)
}
