use super::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, sync::Arc};

const OCCURRENCE_SHA: &str = "4aa74abc5a79fa83c96e946e6de52fd3229184efb1d39a7e8a958fed0a171385";
const BASELINE_SHA: &str = "5ce90f0e0a8404a09f2515662b3e2b760993cdbdf8a1f99ea12eafef2e730a27";
const CLASSES_SHA: &str = "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23";
const WEIGHTS_SHA: &str = "c4c2cb4139f1af761da5c5d6dcc656dab81ad9a813646c3efef7be672ca5a08a";
const MARKER_SHA: &str = "359e7b659d7c6934d5d51070f3cb1d43bcba6bca9e6d349bfbe71c9edd6911b0";
const BASE: [(&str, &str); 8] = [
    (
        "shape",
        "709148a2224121bbf8ea00bee2d6af3f27baca11ff0e3ace8ba533848c71fafd",
    ),
    (
        "table",
        "9bcb0265b52fa7fa86336e41403fe76a532f10bd270be351051fd8626f95d53d",
    ),
    (
        "cells",
        "fae89184d30bacc4ef16af6339cb458e92891dd91a43cf237cdbc473deb60383",
    ),
    (
        "routes",
        "5b6fb58e61fa475939767d68a446f97f1bff02c0e5935a3ea8bb51e6515783d8",
    ),
    (
        "gates",
        "4a3fa113f73a35ffdc477efd023a82ba8239cd2a9e045ef90899102c97c66bb7",
    ),
    (
        "plans",
        "9a0235f2be6a21592553571f7e0b68e8f94b83d98c737d462f19c32d34f850cf",
    ),
    (
        "trunk",
        "c69c748209abad27c3be2bff7ed438fa12df54bdd83b546d9dda9331112ba577",
    ),
    (
        "roots",
        "875a0857dfef840d88a64c90a5777a6154a1674cde08a71429e2cca93fb7bb5a",
    ),
];
const BASELINE_STRIDE: usize = CONTEXT + 3;
const BASELINE_WORDS: usize = 21;
const MARKER_CFG: [u32; 16] = [
    201, 35, 560, 652, 714, COLON, 0, 0xf, 0xff, 0xfff, 0xffff, 4, 0x10000, 20, COLON, 223,
];
const MARKER_MAP_LEN: usize = (4_usize << 20) + 0x20000;
const STRIDE: usize = CONTEXT + 8;
const OCC_WORDS: usize = 29 + CONTEXT * 20;
const WRITER_WORDS: usize = 47;
const SCRATCH_WORDS: usize = 22_528 / 4;
const CHUNK: usize = 64;

pub struct Files<'a> {
    pub copy: &'a str,
    pub occurrence: &'a str,
    pub baseline: &'a str,
    pub base: &'a str,
    pub classes: &'a str,
    pub weights: &'a str,
    pub marker: &'a str,
}

struct Marker {
    cfg: CudaSlice<u32>,
    map: CudaSlice<u32>,
    info: CudaSlice<u32>,
    scores: CudaSlice<u32>,
}

struct Device {
    stream: Arc<CudaStream>,
    marker_kernel: CudaFunction,
    occurrence_kernel: CudaFunction,
    copy_kernel: CudaFunction,
    shape: CudaSlice<u32>,
    table: CudaSlice<u32>,
    cells: CudaSlice<u32>,
    routes: CudaSlice<u32>,
    gates: CudaSlice<u32>,
    plans: CudaSlice<u32>,
    trunk: CudaSlice<u32>,
    roots: CudaSlice<u32>,
    classes: CudaSlice<u8>,
    weights: CudaSlice<i16>,
    marker: Marker,
}

fn grid(count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((count as u32).div_ceil(32), 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn digest(path: &str) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut chunk = [0u8; 65_536];
    loop {
        let len = file.read(&mut chunk)?;
        if len == 0 {
            return Ok(format!("{:x}", hash.finalize()));
        }
        hash.update(&chunk[..len]);
    }
}

fn checked(path: impl AsRef<Path>, sha: &str) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != sha {
        return Err(format!("frozen input digest mismatch at {}", path.display()).into());
    }
    Ok(bytes)
}

fn image(
    ctx: &Arc<CudaContext>,
    path: &str,
    sha: Option<&str>,
    symbol: &str,
) -> Result<CudaFunction> {
    let bytes = if let Some(sha) = sha {
        checked(path, sha)?
    } else {
        fs::read(path)?
    };
    if !bytes.starts_with(b"\x7fELF") {
        return Err(format!("expected AOT ELF module at {path}").into());
    }
    Ok(ctx
        .load_module(Ptx::from_binary(bytes))?
        .load_function(symbol)?)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

impl Marker {
    fn open(stream: &Arc<CudaStream>, path: &str) -> Result<Self> {
        let bytes = checked(path, MARKER_SHA)?;
        let mut rows: BTreeMap<u32, (u32, BTreeMap<u32, i32>)> = BTreeMap::new();
        for line in std::str::from_utf8(&bytes)?.lines() {
            let cols: Vec<_> = line.split('\t').collect();
            let [key, support, token, score] = cols.as_slice() else {
                return Err("invalid marker TSV shape".into());
            };
            let (key, support, token, score) = (
                key.parse()?,
                support.parse()?,
                token.parse()?,
                score.parse()?,
            );
            let (old_support, entries) = rows
                .entry(key)
                .or_insert_with(|| (support, BTreeMap::new()));
            if *old_support != support || token >= VOCAB || entries.insert(token, score).is_some() {
                return Err("inconsistent frozen marker state".into());
            }
        }
        let mut map = vec![0u32; MARKER_MAP_LEN];
        let mut info = Vec::new();
        let mut scores = Vec::new();
        for (index, (key, (support, entries))) in rows.into_iter().enumerate() {
            *map.get_mut(key as usize).ok_or("marker key outside map")? = u32::try_from(index + 1)?;
            info.extend([
                support,
                u32::try_from(scores.len() / 2)?,
                u32::try_from(entries.len())?,
            ]);
            for (token, score) in entries {
                scores.extend([token, score as u32]);
            }
        }
        Ok(Self {
            cfg: stream.clone_htod(&MARKER_CFG)?,
            map: stream.clone_htod(&map)?,
            info: stream.clone_htod(&info)?,
            scores: stream.clone_htod(&scores)?,
        })
    }
}

impl Device {
    fn open(files: &Files<'_>) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        if ctx.compute_capability()? != (8, 9) {
            return Err("registered code images require sm_89".into());
        }
        let stream = ctx.default_stream();
        let marker_kernel = image(&ctx, files.baseline, Some(BASELINE_SHA), "score_openings")?;
        let occurrence_kernel = image(
            &ctx,
            files.occurrence,
            Some(OCCURRENCE_SHA),
            "score_occurrences",
        )?;
        let copy_kernel = image(&ctx, files.copy, None, "copy_writer")?;
        let upload = |index: usize| -> Result<CudaSlice<u32>> {
            let (name, sha) = BASE[index];
            let payload = checked(Path::new(files.base).join(name), sha)?;
            if payload.len() % 4 != 0 || payload.is_empty() {
                return Err(format!("malformed base buffer {name}").into());
            }
            Ok(stream.clone_htod(&words(&payload))?)
        };
        let bytes = checked(files.weights, WEIGHTS_SHA)?;
        let weights: Vec<_> = bytes
            .chunks_exact(2)
            .map(|x| i16::from_le_bytes([x[0], x[1]]))
            .collect();
        if weights.len() != 513
            || weights
                .iter()
                .any(|weight| !(-4096..=4096).contains(weight))
        {
            return Err("invalid exact reader words".into());
        }
        let device_weights = stream.clone_htod(&weights)?;
        if stream.clone_dtoh(&device_weights)? != weights {
            return Err("reader words changed on device".into());
        }
        Ok(Self {
            marker_kernel,
            occurrence_kernel,
            copy_kernel,
            shape: upload(0)?,
            table: upload(1)?,
            cells: upload(2)?,
            routes: upload(3)?,
            gates: upload(4)?,
            plans: upload(5)?,
            trunk: upload(6)?,
            roots: upload(7)?,
            classes: stream.clone_htod(&checked(files.classes, CLASSES_SHA)?)?,
            weights: device_weights,
            marker: Marker::open(&stream, files.marker)?,
            stream,
        })
    }

    fn marker_scores(&self, rows: &[&Row]) -> Result<Vec<u32>> {
        let mut gates = vec![NONE; rows.len()];
        let colons: Vec<_> = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.prefix.last() == Some(&COLON))
            .collect();
        if colons.is_empty() {
            return Ok(gates);
        }
        let n = colons.len();
        let mut records = vec![0; n * BASELINE_STRIDE];
        for ((_, row), record) in colons.iter().zip(records.chunks_exact_mut(BASELINE_STRIDE)) {
            record[0] = row.prefix.len() as u32;
            record[1] = row.target;
            record[2] = row.rival;
            record[3..3 + row.prefix.len()].copy_from_slice(&row.prefix);
        }
        let packed = self.stream.clone_htod(&records)?;
        let mut scratch = self.stream.alloc_zeros::<u32>(n * SCRATCH_WORDS)?;
        let mut output = self.stream.alloc_zeros::<u32>(n * BASELINE_WORDS)?;
        let stride = BASELINE_STRIDE as u32;
        let count = n as u32;
        let marker_arm = 1u32;
        let mut call = self.stream.launch_builder(&self.marker_kernel);
        call.arg(&self.marker.cfg)
            .arg(&self.shape)
            .arg(&self.table)
            .arg(&self.cells)
            .arg(&self.routes)
            .arg(&self.gates)
            .arg(&self.plans)
            .arg(&self.trunk)
            .arg(&packed)
            .arg(&stride)
            .arg(&count)
            .arg(&marker_arm)
            .arg(&self.marker.map)
            .arg(&self.marker.info)
            .arg(&self.marker.scores)
            .arg(&mut scratch)
            .arg(&mut output);
        // Frozen baseline ABI. Each thread receives one validated colon prefix and unique scratch/output.
        unsafe { call.launch(grid(n))? };
        for ((index, row), record) in colons.into_iter().zip(
            self.stream
                .clone_dtoh(&output)?
                .chunks_exact(BASELINE_WORDS),
        ) {
            if record[0] != 0 || record[4] >= VOCAB {
                return Err(format!("marker scorer rejected {}", row.pair_id).into());
            }
            gates[index] = record[4];
        }
        Ok(gates)
    }

    fn events(&self, rows: &[&Row], arms: usize) -> Result<Vec<WriterEvent>> {
        let mut events = Vec::with_capacity(rows.len() * arms);
        for chunk in rows.chunks(CHUNK) {
            events.extend(self.chunk(chunk, arms)?);
        }
        Ok(events)
    }

    fn chunk(&self, rows: &[&Row], arms: usize) -> Result<Vec<WriterEvent>> {
        let n = rows.len();
        let marker = self.marker_scores(rows)?;
        let mut records = vec![0; n * STRIDE];
        for ((row, record), &gate) in rows
            .iter()
            .zip(records.chunks_exact_mut(STRIDE))
            .zip(&marker)
        {
            record[0] = row.prefix.len() as u32;
            record[1] = row.target;
            record[2] = row.rival;
            record[3] = row.gold_pos.unwrap_or(NONE);
            record[4] = row.rival_pos.unwrap_or(NONE);
            record[5] = gate;
            // The frozen scorer computes its own selected read and native
            // reference. This suppresses its experimental supplied writer arms.
            record[6] = 1;
            record[7] = row.wrong_same_id_pos.unwrap_or(NONE);
            record[8..8 + row.prefix.len()].copy_from_slice(&row.prefix);
        }
        let packed = self.stream.clone_htod(&records)?;
        let marker = self.stream.clone_htod(&marker)?;
        let mut scratch = self.stream.alloc_zeros::<u32>(n * arms * SCRATCH_WORDS)?;
        let mut occurrence = self.stream.alloc_zeros::<u32>(n * OCC_WORDS)?;
        let count = n as u32;
        let mut call = self.stream.launch_builder(&self.occurrence_kernel);
        call.arg(&self.shape)
            .arg(&self.table)
            .arg(&self.cells)
            .arg(&self.routes)
            .arg(&self.gates)
            .arg(&self.plans)
            .arg(&self.trunk)
            .arg(&self.roots)
            .arg(&self.classes)
            .arg(&self.weights)
            .arg(&packed)
            .arg(&count)
            .arg(&mut scratch)
            .arg(&mut occurrence);
        // Frozen occurrence ABI. Scratch is large enough per row and per subsequent writer arm.
        unsafe { call.launch(grid(n))? };

        let arm_count = arms as u32;
        let mut output = self.stream.alloc_zeros::<u32>(n * arms * WRITER_WORDS)?;
        let mut call = self.stream.launch_builder(&self.copy_kernel);
        call.arg(&self.shape)
            .arg(&self.table)
            .arg(&self.cells)
            .arg(&self.routes)
            .arg(&self.gates)
            .arg(&self.plans)
            .arg(&self.trunk)
            .arg(&packed)
            .arg(&occurrence)
            .arg(&marker)
            .arg(&count)
            .arg(&arm_count)
            .arg(&mut scratch)
            .arg(&mut output);
        // Same CUDA stream serializes both launches. No address is dereferenced before device bounds checks.
        unsafe { call.launch(grid(n * arms))? };

        let reference = self.stream.clone_dtoh(&occurrence)?;
        let output = self.stream.clone_dtoh(&output)?;
        let mut events = Vec::with_capacity(n * arms);
        for ((row, native), result) in rows
            .iter()
            .zip(reference.chunks_exact(OCC_WORDS))
            .zip(output.chunks_exact(arms * WRITER_WORDS))
        {
            let first = events.len();
            for words in result.chunks_exact(WRITER_WORDS) {
                events.push(WriterEvent::decode(row, words)?);
            }
            // The unmodified CUDA artifact independently fixes reference gate/prediction.
            let ordinary = &events[first];
            if native[0] != 0
                || native[9] != 0
                || native[3] != ordinary.read_pos.unwrap_or(NONE)
                || native[10] != ordinary.gate_id
                || native[11] != ordinary.native_proposal
                || native[12] != ordinary.first_admitted_cell.unwrap_or(NONE)
            {
                return Err(format!("frozen ordinary writer disagrees at {}", row.pair_id).into());
            }
        }
        Ok(events)
    }
}

fn optional(word: u32) -> Option<u32> {
    (word != NONE).then_some(word)
}

impl WriterEvent {
    fn decode(row: &Row, out: &[u32]) -> Result<Self> {
        if out[0] != 0
            || out[3] >= VOCAB
            || out[8] >= VOCAB
            || out[9] >= VOCAB
            || out[10] > 1
            || out[46] > 1
        {
            return Err(format!("writer rejected {} with code {}", row.pair_id, out[0]).into());
        }
        let causal_successor = if let Some(pos) = optional(out[13]) {
            let id = *row
                .prefix
                .get(pos as usize)
                .ok_or("causal pointer outside prefix")?;
            Some((pos, id))
        } else {
            None
        };
        let mut read_vector = [0; 32];
        for (value, &word) in read_vector.iter_mut().zip(&out[14..46]) {
            *value = word as i32;
        }
        let event = Self {
            read_pos: optional(out[1]),
            claim_pos: optional(out[2]),
            gate_id: out[3],
            source_id: optional(out[4]),
            source_distance: optional(out[5]),
            source_admitted: out[46] == 1,
            minimum_distance: out[6],
            first_admitted_cell: optional(out[7]),
            native_proposal: out[8],
            committed_id: out[9],
            copied: out[10] == 1,
            read_code: out[11],
            recurrence_code: out[12],
            causal_successor,
            read_vector,
        };
        if event.copied {
            if event.claim_pos.is_none()
                || event.claim_pos != event.read_pos
                || event.source_id != Some(event.committed_id)
                || !event.source_admitted
                || event.source_distance != Some(event.minimum_distance)
            {
                return Err("device violated exact-copy prerequisites".into());
            }
        } else if event.committed_id != event.native_proposal {
            return Err("device changed a native proposal without authenticated Copy".into());
        }
        Ok(event)
    }
}

pub fn replay<'a>(
    files: &Files<'_>,
    rows: impl Iterator<Item = &'a Row>,
) -> Result<Vec<Observation<'a>>> {
    let device = Device::open(files)?;
    let rows: Vec<_> = rows.collect();
    let first = device.events(&rows, ARMS)?;
    let mut successor_rows = Vec::new();
    for (&row, events) in rows.iter().zip(first.chunks_exact(ARMS)) {
        for event in events {
            if event.committed_id != EOS {
                successor_rows.push(row.self_feed(event.committed_id));
            }
        }
    }
    let successor_refs: Vec<_> = successor_rows.iter().collect();
    let mut successors = device.events(&successor_refs, 1)?.into_iter();
    let mut first = first.into_iter();
    let mut observations = Vec::with_capacity(rows.len());
    for row in rows {
        let mut arms = BTreeMap::new();
        for name in ARM_NAMES {
            let first = first.next().ok_or("missing first transition")?;
            let successor = if first.committed_id == EOS {
                None
            } else {
                let successor = successors.next().ok_or("missing self-fed transition")?;
                if successor.copied || successor.claim_pos.is_some() {
                    return Err("copy state persisted into native successor".into());
                }
                Some(successor)
            };
            let terminal = first.committed_id == EOS
                || successor
                    .as_ref()
                    .is_some_and(|event| event.committed_id == EOS);
            arms.insert(
                name,
                Arm {
                    first,
                    successor,
                    terminal,
                },
            );
        }
        observations.push(Observation {
            pair_id: &row.pair_id,
            prefix: &row.prefix,
            arms,
        });
    }
    if first.next().is_some() || successors.next().is_some() {
        return Err("unclaimed device transition".into());
    }
    Ok(observations)
}
