use super::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};
const NONE: u32 = u32::MAX;
const MAP_LEN: usize = (4usize << contract::LEVEL_SHIFT) + contract::MARKER_BUCKET as usize + 65536;
const ARM_NAMES: [&str; 2] = ["colon", "marker"];

struct Device {
    stream: Arc<CudaStream>,
    fit: CudaFunction,
    score: CudaFunction,
    cfg: CudaSlice<u32>,
    shape: CudaSlice<u32>,
    table: CudaSlice<u32>,
    cells: CudaSlice<u32>,
    routes: CudaSlice<u32>,
    gates: CudaSlice<u32>,
    plans: CudaSlice<u32>,
    trunk: CudaSlice<u32>,
}
type Weights = BTreeMap<u32, (u32, BTreeMap<u32, i32>)>;

fn grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(64), 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn packed(rows: &[Row]) -> Vec<u32> {
    let mut data = vec![0u32; rows.len() * STRIDE];
    for (row, chunk) in rows.iter().zip(data.chunks_exact_mut(STRIDE)) {
        chunk[0] = row.prefix.len() as u32;
        chunk[1] = row.target;
        chunk[2] = row.rival.unwrap_or(NONE);
        chunk[3..3 + row.prefix.len()].copy_from_slice(&row.prefix);
    }
    data
}

impl Device {
    fn open(image: &str, base: &str) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        if ctx.compute_capability()? != (8, 9) {
            return Err("the registered device is sm_89".into());
        }
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_binary(fs::read(image)?))?;
        let upload = |name: &str| -> Result<CudaSlice<u32>> {
            let bytes = fs::read(Path::new(base).join(name))?;
            if bytes.len() % 4 != 0 {
                return Err("partial word in base payload".into());
            }
            let words = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect::<Vec<_>>();
            Ok(stream.clone_htod(&words)?)
        };
        Ok(Self {
            fit: module.load_function("fit_openings")?,
            score: module.load_function("score_openings")?,
            cfg: stream.clone_htod(&config())?,
            shape: upload("shape")?,
            table: upload("table")?,
            cells: upload("cells")?,
            routes: upload("routes")?,
            gates: upload("gates")?,
            plans: upload("plans")?,
            trunk: upload("trunk")?,
            stream,
        })
    }

    fn fit_arm(&self, rows: &[Row], arm: u32) -> Result<Weights> {
        let contrast = rows.iter().filter(|r| r.rival.is_some()).count();
        let touched = 5usize
            .checked_mul(rows.len() + 2 * contrast)
            .ok_or("update budget overflow")?;
        if touched >= i32::MAX as usize / 2 {
            return Err("signed count overflow".into());
        }
        let capacity = (2 * touched)
            .max(16)
            .checked_next_power_of_two()
            .ok_or("hash capacity overflow")?;
        let mask = u32::try_from(capacity - 1)?;
        let mut slots = self.stream.alloc_zeros::<u64>(capacity)?;
        let mut scores = self.stream.alloc_zeros::<i32>(capacity)?;
        let mut supports = self.stream.alloc_zeros::<u32>(MAP_LEN)?;
        let mut errors = self.stream.alloc_zeros::<u32>(1)?;
        let stride = STRIDE as u32;
        for chunk in rows.chunks(8192) {
            let data = self.stream.clone_htod(&packed(chunk))?;
            let n = chunk.len() as u32;
            let mut call = self.stream.launch_builder(&self.fit);
            call.arg(&self.trunk)
                .arg(&self.cfg)
                .arg(&data)
                .arg(&stride)
                .arg(&n)
                .arg(&arm)
                .arg(&mut slots)
                .arg(&mut scores)
                .arg(&mask)
                .arg(&mut supports)
                .arg(&mut errors);
            // Input rows include a validated prefix and fit buffers have two slots per possible touched entry.
            unsafe {
                call.launch(grid(chunk.len()))?;
            }
        }
        if self.stream.clone_dtoh(&errors)? != [0] {
            return Err("device rejected a key or update".into());
        }
        let supports = self.stream.clone_dtoh(&supports)?;
        let slots = self.stream.clone_dtoh(&slots)?;
        let scores = self.stream.clone_dtoh(&scores)?;
        let mut weights = Weights::new();
        for (word, score) in slots.into_iter().zip(scores) {
            if word == 0 {
                continue;
            }
            let word = word - 1;
            let key = u32::try_from(word >> 12)?;
            let token = (word & 4095) as u32;
            let seen = *supports
                .get(key as usize)
                .ok_or("device key outside registered space")?;
            if seen == 0
                || weights
                    .entry(key)
                    .or_insert_with(|| (seen, BTreeMap::new()))
                    .1
                    .insert(token, score)
                    .is_some()
            {
                return Err("duplicate state key or zero support".into());
            }
        }
        for level in 0..5 {
            let rows_at_level = weights
                .iter()
                .filter(|(k, _)| (*k >> contract::LEVEL_SHIFT) == level);
            let mut total_support = 0u64;
            let mut total_signed = 0i64;
            for (_, (seen, values)) in rows_at_level {
                total_support += u64::from(*seen);
                total_signed += values.values().map(|&x| i64::from(x)).sum::<i64>();
            }
            if total_support != rows.len() as u64
                || total_signed != 2 * (rows.len() - contrast) as i64
            {
                return Err(format!("wrong update totals at level {level}").into());
            }
        }
        Ok(weights)
    }

    fn score_arm(&self, rows: &[Row], weights: &Weights, arm: u32) -> Result<Vec<Value>> {
        let mut map = vec![0u32; MAP_LEN];
        let mut info = Vec::new();
        let mut pairs = Vec::new();
        for (i, (&key, (seen, values))) in weights.iter().enumerate() {
            *map.get_mut(key as usize)
                .ok_or("native row key out of domain")? = u32::try_from(i + 1)?;
            info.extend([
                *seen,
                u32::try_from(pairs.len() / 2)?,
                u32::try_from(values.len())?,
            ]);
            for (&token, &score) in values {
                pairs.extend([token, score as u32]);
            }
        }
        if info.is_empty() {
            info.push(0);
            pairs.push(0);
        }
        let map = self.stream.clone_htod(&map)?;
        let info = self.stream.clone_htod(&info)?;
        let pairs = self.stream.clone_htod(&pairs)?;
        // Hash each selected integer slice after DMA readback, in the product's packed LE layout.
        let physical_map = self.stream.clone_dtoh(&map)?;
        let physical_info = self.stream.clone_dtoh(&info)?;
        let physical_scores = self.stream.clone_dtoh(&pairs)?;
        let empty_digest = format!("{:x}", Sha256::digest([]));
        let mut digests = BTreeMap::new();
        for &key in weights.keys() {
            let index = physical_map[key as usize]
                .checked_sub(1)
                .ok_or("lost native row")? as usize;
            let meta = &physical_info[3 * index..3 * index + 3];
            let start = 2 * meta[1] as usize;
            let end = start + 2 * meta[2] as usize;
            let mut sha = Sha256::new();
            for pair in physical_scores[start..end].chunks_exact(2) {
                sha.update(u16::try_from(pair[0])?.to_le_bytes());
                sha.update(pair[1].to_le_bytes());
            }
            digests.insert(key, format!("{:x}", sha.finalize()));
        }
        let scratch_bytes = usize::try_from(d11::kernel::host_record_sizes().0)?;
        let mut scratch = self
            .stream
            .alloc_zeros::<u8>(scratch_bytes * rows.len().min(128))?;
        let mut observations = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(128) {
            let n = chunk.len() as u32;
            let stride = STRIDE as u32;
            let data = self.stream.clone_htod(&packed(chunk))?;
            let mut out = self.stream.alloc_zeros::<u32>(21 * chunk.len())?;
            let mut call = self.stream.launch_builder(&self.score);
            call.arg(&self.cfg)
                .arg(&self.shape)
                .arg(&self.table)
                .arg(&self.cells)
                .arg(&self.routes)
                .arg(&self.gates)
                .arg(&self.plans)
                .arg(&self.trunk)
                .arg(&data)
                .arg(&stride)
                .arg(&n)
                .arg(&arm)
                .arg(&map)
                .arg(&info)
                .arg(&pairs)
                .arg(&mut scratch)
                .arg(&mut out);
            // Each device row writes a unique 21-word output and one full D11 scratch record.
            unsafe {
                call.launch(grid(chunk.len()))?;
            }
            for (r, v) in chunk
                .iter()
                .zip(self.stream.clone_dtoh(&out)?.chunks_exact(21))
            {
                if v[0] != 0
                    || v[4] >= VOCAB
                    || v[5] >= VOCAB
                    || v[9..12].iter().any(|x| !(1..=VOCAB).contains(x))
                {
                    return Err(format!("invalid CUDA readout {:?}", v).into());
                }
                observations.push(json!({"group_id":r.group_id,"assignment":r.assignment,"side":r.side,
                    "pair_id":r.pair_id,"cohort":r.cohort,"target":r.target,"rival":r.rival,
                    "raw_target":r.raw_target.unwrap_or(r.target),"question_state":(v[1]!=NONE).then_some(v[1]),
                    "key":(v[2]!=NONE).then_some(v[2]),"support":v[3],"selected":v[4],"emitted":v[5],
                    "scores":[v[6] as i32,v[7] as i32,v[8] as i32],"ranks":[v[9],v[10],v[11]],
                    "winning_score":v[12] as i32,"first_admitted_cell":v[13],"writer_runner_up":v[14],
                    "trained_candidates":v[15],"keys":&v[16..21],"sparse_scores_sha256":digests.get(&v[2]).unwrap_or(&empty_digest)}));
            }
        }
        Ok(observations)
    }
}

fn save(weights: &Weights, path: &Path) -> Result<()> {
    let mut file = new_file(path)?;
    for (key, (seen, values)) in weights {
        for (token, score) in values {
            writeln!(file, "{key}\t{seen}\t{token}\t{score}")?;
        }
    }
    file.flush()?;
    Ok(())
}

fn load(path: &Path) -> Result<Weights> {
    let mut weights = Weights::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let cols = line.split('\t').collect::<Vec<_>>();
        if cols.len() != 4 {
            return Err("bad score TSV".into());
        }
        let key: u32 = cols[0].parse()?;
        let seen: u32 = cols[1].parse()?;
        let id: u32 = cols[2].parse()?;
        let score: i32 = cols[3].parse()?;
        let (prev, row) = weights
            .entry(key)
            .or_insert_with(|| (seen, BTreeMap::new()));
        if id >= VOCAB || *prev != seen || seen == 0 || row.insert(id, score).is_some() {
            return Err("bad fitted address".into());
        }
    }
    if weights.is_empty() {
        return Err("empty fitted state".into());
    }
    Ok(weights)
}

fn save_lines(path: &Path, vals: &[Value]) -> Result<()> {
    let mut out = new_file(path)?;
    for val in vals {
        serde_json::to_writer(&mut out, val)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

pub fn fit(image: &str, base: &str, donor: &str, contrast: &str, out: &str) -> Result<()> {
    let mut rows = read_rows(donor, false)?;
    let donor_count = rows.len();
    let other = read_rows(contrast, true)?;
    if donor_count != 118915
        || other.len() != 800
        || other.iter().any(|r| {
            r.target == contract::SPACE
                || r.rival == Some(r.target)
                || r.rival == Some(contract::SPACE)
        })
    {
        return Err(
            "frozen training dose differs from the corrected donor census or paired contrasts"
                .into(),
        );
    }
    rows.extend(other);
    fs::create_dir(out)?;
    let device = Device::open(image, base)?;
    let mut receipts = Vec::new();
    for (arm, name) in ARM_NAMES.iter().enumerate() {
        let begin = std::time::Instant::now();
        let weights = device.fit_arm(&rows, arm as u32)?;
        let dur = begin.elapsed().as_millis();
        let path = Path::new(out).join(format!("{name}.tsv"));
        save(&weights, &path)?;
        if weights != load(&path)? {
            return Err("integer score TSV reload differs".into());
        }
        let contrast = rows.len() - donor_count;
        receipts.push(json!({"arm":name,"donor":donor_count,"contrast":contrast,"elapsed_ms":dur,
            "positive_updates":5*rows.len(),"negative_updates":10*contrast,"level_visits":5*rows.len(),
            "learned_addresses":weights.len(),"learned_token_words":weights.values().map(|x|x.1.len()).sum::<usize>()}));
    }
    save_lines(&Path::new(out).join("receipts.jsonl"), &receipts)
}

pub fn score(image: &str, base: &str, panel: &str, dir: &str, out: &str) -> Result<()> {
    let rows = read_rows(panel, true)?;
    let reversed: Vec<_> = rows.iter().rev().cloned().collect();
    fs::create_dir(out)?;
    let device = Device::open(image, base)?;
    for (arm, name) in ARM_NAMES.iter().enumerate() {
        let weights = load(&Path::new(dir).join(format!("{name}.tsv")))?;
        let held = device.score_arm(&rows, &weights, arm as u32)?;
        let mut replay = device.score_arm(&reversed, &weights, arm as u32)?;
        replay.reverse();
        if held != replay {
            return Err("device order-reversed replay differs".into());
        }
        save_lines(&Path::new(out).join(format!("{name}.jsonl")), &held)?;
    }
    Ok(())
}

mod planted;
pub use planted::fixture;
