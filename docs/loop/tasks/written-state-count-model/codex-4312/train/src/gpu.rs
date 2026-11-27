use crate::{plan, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
};
use m1_d11_artifact_scorer_instrument::{artifact::ArtifactTables, kernel::host_record_sizes};
use std::{sync::Arc, time::Instant};
use uor_r4_discrete::{
    corpus::{sha256, PreparedCorpus},
    d11_export::export_d11,
    normalization_mechanism::{NormFrameTable, NormalizationMode},
    position_mechanism::PositionStepTable,
    snapshot::Snapshot,
    trunk::{
        compute_recurrence_state, decode_codebook_table, RecurrenceMode, SparseCountTable,
        TrunkTables,
    },
    w2_capacity::decode_read_vector,
};
use uor_r4_integer::discrete_d11::IntegerDiscreteModel;

const CHUNK: usize = 8192;
const THREADS: u32 = 128;
const READBACK: usize = 1 << 20;
const TRACES: usize = 64;
const FAMILIES: usize = 12;
const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/visible_fit.cubin"));
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/visible_fit.ptx"));
const TARGET: &str = env!("VISIBLE_FIT_CUBIN_TARGET");

pub fn kernel_digests() -> serde_json::Value {
    serde_json::json!({ "target": TARGET, "cubin_sha256": sha256(CUBIN), "ptx_sha256": sha256(PTX.as_bytes()) })
}

struct Rows {
    positions: Vec<u32>,
    lengths: Vec<u8>,
}

impl Rows {
    fn from_corpus(corpus: &PreparedCorpus, source_event_order: Option<&[u8]>) -> Result<Self> {
        let mut positions = Vec::with_capacity(corpus.eligible_len());
        let mut lengths = Vec::with_capacity(corpus.eligible_len());
        let mut start = 0usize;
        let mut scanned = 0usize;
        for &at in corpus.eligible_positions() {
            while scanned < at {
                if corpus.tokens()[scanned] == 0 {
                    start = scanned;
                }
                scanned += 1;
            }
            let length = (at - start).min(plan::CONTEXT as usize);
            if length == 0 || corpus.tokens()[at] == 0 {
                return Err("selected output row has no causal prefix or has a BOS target".into());
            }
            positions.push(u32::try_from(at)?);
            lengths.push(u8::try_from(length)?);
        }
        if let Some(order) = source_event_order {
            if order.len()
                != positions
                    .len()
                    .checked_mul(4)
                    .ok_or("order length overflow")?
            {
                return Err("full-count source order length differs".into());
            }
            let mut lengths_by_position = vec![0u8; corpus.tokens().len()];
            for (position, length) in positions.into_iter().zip(lengths) {
                lengths_by_position[position as usize] = length;
            }
            positions = Vec::with_capacity(corpus.eligible_len());
            lengths = Vec::with_capacity(corpus.eligible_len());
            for bytes in order.chunks_exact(4) {
                let position = u32::from_le_bytes(bytes.try_into()?);
                let length = lengths_by_position
                    .get_mut(position as usize)
                    .ok_or("full-count target position outside stream")?;
                let length = std::mem::take(length);
                if length == 0 {
                    return Err(
                        "full-count event order repeats or adds an unselected target".into(),
                    );
                }
                positions.push(position);
                lengths.push(length);
            }
        }
        Ok(Self { positions, lengths })
    }

    fn prefix<'a>(&self, tokens: &'a [u32], ordinal: usize) -> &'a [u32] {
        let at = self.positions[ordinal] as usize;
        &tokens[at - usize::from(self.lengths[ordinal])..at]
    }
}

struct FitDevice {
    stream: Arc<CudaStream>,
    capture: CudaFunction,
    winner: CudaFunction,
    tokens: CudaSlice<u32>,
    positions: CudaSlice<u32>,
    lengths: CudaSlice<u8>,
    table: CudaSlice<i32>,
    gates: CudaSlice<i32>,
    plans: CudaSlice<i32>,
    roots: CudaSlice<u32>,
    steps: CudaSlice<u32>,
    norms: CudaSlice<u32>,
    stages: u32,
    scratch: CudaSlice<u8>,
    errors: CudaSlice<u32>,
    unigram: CudaSlice<u32>,
    keys: CudaSlice<u64>,
    counts: CudaSlice<u32>,
    mask: u32,
}

impl FitDevice {
    fn new(
        base: &ArtifactTables,
        tokens: &[u32],
        rows: &Rows,
        roots: &[[u8; 4]],
        steps: &PositionStepTable,
        norms: &NormFrameTable,
    ) -> Result<Self> {
        let context = CudaContext::new(0)?;
        let (major, minor) = context.compute_capability()?;
        if TARGET != format!("sm_{major}{minor}") {
            return Err(
                format!("count cubin {TARGET} differs from device sm_{major}{minor}").into(),
            );
        }
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(CUBIN.to_vec()))?;
        let capacity = rows
            .positions
            .len()
            .checked_mul(FAMILIES * 2)
            .and_then(usize::checked_next_power_of_two)
            .ok_or("count table capacity overflow")?;
        let mask = u32::try_from(capacity - 1)?;
        let packed_roots = roots
            .iter()
            .map(|blocks| {
                blocks
                    .iter()
                    .enumerate()
                    .fold(0u32, |word, (block, &root)| {
                        word | (u32::from(root) << (4 * block))
                    })
            })
            .collect::<Vec<_>>();
        let steps = steps
            .export_words()
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let norms = norms
            .export_words()
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        let (scratch_bytes, _, plan_bytes) = host_record_sizes();
        if plan_bytes != i64::from(m1_d11_artifact_scorer_instrument::kernel::PLAN_WORDS as u32) * 4
        {
            return Err("D11 scorer row-plan ABI differs".into());
        }
        let scratch_bytes = usize::try_from(scratch_bytes)?
            .checked_mul(CHUNK)
            .ok_or("D11 scratch size overflow")?;
        Ok(Self {
            capture: module.load_function("visible_count_rows")?,
            winner: module.load_function("visible_unigram_best")?,
            tokens: stream.clone_htod(tokens)?,
            positions: stream.clone_htod(&rows.positions)?,
            lengths: stream.clone_htod(&rows.lengths)?,
            table: stream.clone_htod(&base.table)?,
            gates: stream.clone_htod(&base.gates)?,
            plans: stream.clone_htod(&base.plans)?,
            roots: stream.clone_htod(&packed_roots)?,
            steps: stream.clone_htod(&steps)?,
            norms: stream.clone_htod(&norms)?,
            stages: u32::try_from(base.shape[4])?,
            scratch: stream.alloc_zeros(scratch_bytes)?,
            errors: stream.alloc_zeros(1)?,
            unigram: stream.alloc_zeros(4097)?,
            keys: stream.alloc_zeros(capacity)?,
            counts: stream.alloc_zeros(capacity)?,
            mask,
            stream,
        })
    }

    fn capture(&mut self, first: usize, count: usize, replay: Option<&[u32]>) -> Result<Vec<u32>> {
        let ordinals = self.stream.clone_htod(replay.unwrap_or(&[0]))?;
        let mut traces = self
            .stream
            .alloc_zeros::<u32>(replay.map_or(1, |rows| rows.len() * 4))?;
        let first = u32::try_from(first)?;
        let count = u32::try_from(count)?;
        let trace_only = u32::from(replay.is_some());
        let mut launch = self.stream.launch_builder(&self.capture);
        launch
            .arg(&self.tokens)
            .arg(&self.positions)
            .arg(&self.lengths)
            .arg(&ordinals)
            .arg(&first)
            .arg(&count)
            .arg(&self.table)
            .arg(&self.gates)
            .arg(&self.plans)
            .arg(&self.roots)
            .arg(&self.steps)
            .arg(&self.norms)
            .arg(&self.stages)
            .arg(&mut self.scratch)
            .arg(&mut self.keys)
            .arg(&mut self.counts)
            .arg(&self.mask)
            .arg(&mut self.unigram)
            .arg(&mut self.errors)
            .arg(&mut traces)
            .arg(&trace_only);
        // The inputs and output ABI are the typed D11 scorer arrays. Rows and
        // scratch are bounded by CHUNK. The probe buffer is sized only for
        // replay, and the kernel does not access it on the count path.
        // The default helper requests 1,024 threads even for tiny replay
        // launches. The SM89 count kernel uses 80 registers per thread.
        let grid = LaunchConfig {
            grid_dim: (count.div_ceil(THREADS), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            launch.launch(grid)?;
        }
        if replay.is_some() {
            return Ok(self.stream.clone_dtoh(&traces)?);
        }
        Ok(Vec::new())
    }

    fn check_errors(&self) -> Result<()> {
        if self.stream.clone_dtoh(&self.errors)? != [0] {
            return Err("GPU rejected at least one read, recurrence or count insertion".into());
        }
        Ok(())
    }

    fn readback(&mut self, expected: u64) -> Result<TrunkTables> {
        self.check_errors()?;
        // One device thread resolves the exact unigram count then original-ID tie.
        unsafe {
            self.stream
                .launch_builder(&self.winner)
                .arg(&mut self.unigram)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let mut unigram_counts = self.stream.clone_dtoh(&self.unigram)?;
        let unigram_best_global =
            u16::try_from(unigram_counts.pop().ok_or("missing unigram winner")?)?;
        if unigram_counts
            .iter()
            .map(|&count| u64::from(count))
            .sum::<u64>()
            != expected
            || unigram_counts[0] != 0
            || unigram_best_global == 0
        {
            return Err("GPU unigram target sum or winner is invalid".into());
        }
        let mut families: [Vec<(u64, u32)>; FAMILIES] = std::array::from_fn(|_| Vec::new());
        let mut sums = [0u64; FAMILIES];
        let capacity = self.mask as usize + 1;
        for start in (0..capacity).step_by(READBACK) {
            let end = (start + READBACK).min(capacity);
            let keys = self.stream.clone_dtoh(&self.keys.slice(start..end))?;
            let counts = self.stream.clone_dtoh(&self.counts.slice(start..end))?;
            for (key, count) in keys.into_iter().zip(counts) {
                if key == 0 && count == 0 {
                    continue;
                }
                let family = usize::try_from((key >> 44) & 15)?;
                let local = key & ((1 << 44) - 1);
                if key >> 48 != 1 || count == 0 || family >= FAMILIES || local & 4095 == 0 {
                    return Err("GPU emitted an invalid sparse count row".into());
                }
                families[family].push((local, count));
                sums[family] += u64::from(count);
            }
        }
        if sums != [expected; FAMILIES] {
            return Err(format!("GPU count sums differ from target dose: {sums:?}").into());
        }
        let mut sparse_tables = Vec::with_capacity(FAMILIES);
        for mut family in families {
            family.sort_unstable_by_key(|&(key, _)| key);
            if family.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err("GPU emitted duplicate addresses in one sparse family".into());
            }
            let mut words = Vec::with_capacity(3 * family.len());
            for (key, count) in family {
                words.extend([(key >> 12) as u32, (key & 4095) as u32, count]);
            }
            sparse_tables.push(SparseCountTable::from_triples(&words)?);
        }
        let [read_table, rec_table, read_rec_table, rec_shuf_table, read_rec_shuf_table, rec_unit_table, read_rec_unit_table, query_table, query_read_table, query_rec_table, query_rec_shuf_table, query_rec_unit_table] =
            <[SparseCountTable; FAMILIES]>::try_from(sparse_tables)
                .map_err(|_| "sparse count-family inventory differs")?;
        Ok(TrunkTables {
            unigram_counts,
            unigram_best_global,
            read_table,
            rec_table,
            read_rec_table,
            rec_shuf_table,
            read_rec_shuf_table,
            rec_unit_table,
            read_rec_unit_table,
            query_table,
            query_read_table,
            query_rec_table,
            query_rec_shuf_table,
            query_rec_unit_table,
        })
    }
}

pub fn fit(
    snapshot: &Snapshot,
    corpus: &PreparedCorpus,
    steps: &PositionStepTable,
    norms: &NormFrameTable,
    source_event_order: Option<&[u8]>,
) -> Result<(TrunkTables, serde_json::Value)> {
    let rows = Rows::from_corpus(corpus, source_event_order)?;
    let (reference, cleanroom) = ArtifactTables::from_bytes(&export_d11(snapshot)?)?;
    let (roots, _) = decode_codebook_table(&snapshot.table)?;
    let mut device = FitDevice::new(&reference, corpus.tokens(), &rows, &roots, steps, norms)?;
    let replay = (0..TRACES.min(rows.positions.len()))
        .map(|i| (i * rows.positions.len() / TRACES.min(rows.positions.len())) as u32)
        .collect::<Vec<_>>();
    let words = device.capture(0, replay.len(), Some(&replay))?;
    device.check_errors()?;
    let mut replay_rows = Vec::with_capacity(replay.len());
    for (&ordinal, device_codes) in replay.iter().zip(words.chunks_exact(4)) {
        let prefix = rows.prefix(corpus.tokens(), ordinal as usize);
        let words = compare_reference(
            &cleanroom,
            &roots,
            steps,
            norms,
            prefix,
            ordinal,
            device_codes,
        )?;
        let target = corpus.tokens()[rows.positions[ordinal as usize] as usize] as u16;
        replay_rows.push(uor_r4_discrete::trunk::TrunkTrainRow {
            read_code: words[0],
            rec_code: words[1],
            rec_shuf_code: words[2],
            rec_unit_code: words[3],
            query_token: *prefix.last().ok_or("empty replay prefix")? as u16,
            target_token: target,
        });
    }
    let start = Instant::now();
    for first in (0..rows.positions.len()).step_by(CHUNK) {
        let count = (rows.positions.len() - first).min(CHUNK);
        device.capture(first, count, None)?;
        if first % (128 * CHUNK) == 0 {
            device.check_errors()?;
            eprintln!(
                "count fit reached {}/{} targets",
                first + count,
                rows.positions.len()
            );
        }
    }
    device.check_errors()?;
    let fit_ms = start.elapsed().as_millis();
    let start = Instant::now();
    let tables = device.readback(u64::try_from(rows.positions.len())?)?;
    if rows.positions.len() <= TRACES && tables != TrunkTables::build(&replay_rows)? {
        return Err("finite all-table GPU fit differs from the independent integer oracle".into());
    }
    Ok((
        tables,
        serde_json::json!({
            "rows": rows.positions.len(),
            "probe_rows": replay_rows.len(),
            "probe_words": replay_rows.len()*4,
            "gpu_hash_slots": u64::from(device.mask)+1,
            "gpu_fit_ms": fit_ms,
            "readback_and_pack_rows_ms": start.elapsed().as_millis(),
            "kernels": kernel_digests(),
            "finite_all_tables_exact": rows.positions.len() <= TRACES,
        }),
    ))
}

fn compare_reference(
    reference: &IntegerDiscreteModel,
    roots: &[[u8; 4]],
    steps: &PositionStepTable,
    norms: &NormFrameTable,
    prefix: &[u32],
    ordinal: u32,
    device_codes: &[u32],
) -> Result<[u16; 4]> {
    let integer = reference.evaluate_prefix(prefix)?;
    let vector: &[i32; 32] = integer.read_vector.as_slice().try_into()?;
    let read = decode_read_vector(vector)?;
    let prefix: Vec<u16> = prefix.iter().map(|&token| token as u16).collect();
    let code = |recurrence, normalization| -> Result<u16> {
        Ok(compute_recurrence_state(
            &prefix,
            roots,
            steps,
            norms,
            recurrence,
            normalization,
            u64::from(ordinal) + 1,
        )?
        .1)
    };
    let codes = [
        read,
        code(
            RecurrenceMode::ActiveHolonomyV1,
            NormalizationMode::ScaleMatchedSnapV1,
        )?,
        code(
            RecurrenceMode::ShuffledOrderControlV1,
            NormalizationMode::ScaleMatchedSnapV1,
        )?,
        code(
            RecurrenceMode::ActiveHolonomyV1,
            NormalizationMode::UnitSnapControlV1,
        )?,
    ];
    if device_codes != codes.map(u32::from) {
        return Err(format!("native read/recurrence words differ at target row {ordinal}: GPU {device_codes:?}, Rust {codes:?}").into());
    }
    Ok(codes)
}
