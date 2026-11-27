use super::*;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

mod fixture;
pub use fixture::fixture;

const STATE_FILE: &str = "weights.i16le";
const CLASS_SHA: &str = "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23";
const WEIGHT_BOUND: i32 = 4096;
const DONOR_CAP: i32 = 128;
const CONTRAST_BOUND: i32 = 7 * TRAIN_ROWS as i32 + 8 * (TRAIN_ROWS / 4) as i32;
const CENSUS_LABELS: [&str; 11] = [
    "donor_ordinary",
    "donor_unparsed",
    "donor_nonlexical",
    "donor_absent",
    "donor_multiple",
    "donor_unique",
    "contrast",
    "hard_nonrival",
    "train_absent",
    "train_absent_empty",
    "bad_row",
];
const ARM_NAMES: [&str; 5] = ["learned", "correct", "swapped", "absent", "marker"];
const ARM_WORDS: usize = 4;
const HEADER_WORDS: usize = 9 + ARM_NAMES.len() * ARM_WORDS;
const CANDIDATE_WORDS: usize = 4 + FEATURES;
const OUT_WORDS: usize = HEADER_WORDS + CONTEXT * CANDIDATE_WORDS;

struct Device {
    stream: Arc<CudaStream>,
    fit: CudaFunction,
    publish: CudaFunction,
    score: CudaFunction,
    roots: CudaSlice<u32>,
    classes: CudaSlice<u8>,
}

struct WriterState {
    shape: CudaSlice<u32>,
    table: CudaSlice<u32>,
    cells: CudaSlice<u32>,
    routes: CudaSlice<u32>,
    gates: CudaSlice<u32>,
    plans: CudaSlice<u32>,
    trunk: CudaSlice<u32>,
}

#[derive(Deserialize)]
struct Baseline {
    pair_id: String,
    group_id: String,
    cohort: String,
    side: u32,
    assignment: u32,
    target: u32,
    rival: u32,
    selected: u32,
    emitted: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Candidate {
    pos: u32,
    id: u32,
    eligible: bool,
    score: Option<i32>,
    rank: Option<u32>,
    features: Option<[u32; FEATURES]>,
    geometry: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Emission {
    source_pos: Option<u32>,
    gate_id: u32,
    emitted_id: u32,
    first_admitted_cell: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Arm {
    #[serde(flatten)]
    immediate: Emission,
    lexical: Option<Emission>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Arms {
    learned: Arm,
    correct: Arm,
    swapped: Arm,
    absent: Arm,
    marker: Arm,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Decision {
    parsed_start: Option<u32>,
    parsed_end: Option<u32>,
    no_read_score: i32,
    winner_pos: Option<u32>,
    best_score: Option<i32>,
    best_ties: u32,
    candidates: Vec<Candidate>,
    gold_rank: Option<u32>,
    rival_rank: Option<u32>,
    arms: Arms,
}

struct FitState {
    donor: CudaSlice<i32>,
    contrast: CudaSlice<i32>,
    weights: CudaSlice<i16>,
    census: CudaSlice<u32>,
}

struct FitReadback {
    weights: Vec<i16>,
    donor: Vec<i32>,
    contrast: Vec<i32>,
    census: Vec<u32>,
    snapshots: Vec<String>,
}

#[derive(Clone, Copy)]
enum StreamKind {
    Donor = 0,
    Contrast = 1,
    Absent = 2,
}

fn grid(n: usize, width: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(width), 1, 1),
        block_dim: (width, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn upload_words(stream: &Arc<CudaStream>, base: &Path, name: &str) -> Result<CudaSlice<u32>> {
    let bytes = fs::read(base.join(name))?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(format!("invalid u32 payload {name}").into());
    }
    let words = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<_>>();
    Ok(stream.clone_htod(&words)?)
}

fn classes(path: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if bytes.len() != VOCAB as usize
        || bytes.iter().any(|&c| c > 9)
        || format!("{:x}", Sha256::digest(&bytes)) != CLASS_SHA
    {
        return Err("invalid frozen canonical class table".into());
    }
    Ok(bytes)
}

fn base_identity(base: &Path) -> Result<()> {
    if fs::read_to_string(base.join("base.sha256"))?.trim() != BASE_SHA {
        return Err("wrong staged D11 model".into());
    }
    Ok(())
}

impl Device {
    fn open(image: &str, base: &str, token_classes: &str) -> Result<Self> {
        let base = Path::new(base);
        base_identity(base)?;
        let classes = classes(token_classes)?;
        let ctx = CudaContext::new(0)?;
        if ctx.compute_capability()? != (8, 9) {
            return Err("device must be sm_89".into());
        }
        let stream = ctx.default_stream();
        let image = fs::read(image)?;
        if !image.starts_with(b"\x7fELF") {
            return Err("expected the registered AOT cubin image".into());
        }
        let module = ctx.load_module(Ptx::from_binary(image))?;
        let roots = upload_words(&stream, base, "roots")?;
        if roots.len() != VOCAB as usize {
            return Err("wrong packed-root vocabulary".into());
        }
        Ok(Self {
            fit: module.load_function("fit_occurrences")?,
            publish: module.load_function("publish_weights")?,
            score: module.load_function("score_occurrences")?,
            roots,
            classes: stream.clone_htod(&classes)?,
            stream,
        })
    }

    fn writer_state(&self, base: &str) -> Result<WriterState> {
        let upload = |name| upload_words(&self.stream, Path::new(base), name);
        let result = WriterState {
            shape: upload("shape")?,
            table: upload("table")?,
            cells: upload("cells")?,
            routes: upload("routes")?,
            gates: upload("gates")?,
            plans: upload("plans")?,
            trunk: upload("trunk")?,
        };
        if result.shape.len() != 8
            || result.table.len() != VOCAB as usize * 4 * 8
            || result.cells.len() != VOCAB as usize
            || result.routes.len() != 4 * 5
            || result.trunk.len() <= 70 + 2 * VOCAB as usize
        {
            return Err("incomplete writer input buffers".into());
        }
        Ok(result)
    }

    fn fit_state(&self) -> Result<FitState> {
        Ok(FitState {
            donor: self.stream.alloc_zeros(STATE_WORDS)?,
            contrast: self.stream.alloc_zeros(STATE_WORDS)?,
            weights: self.stream.alloc_zeros(STATE_WORDS)?,
            census: self.stream.alloc_zeros(CENSUS_LABELS.len())?,
        })
    }

    fn accumulate(&self, words: Vec<u32>, kind: StreamKind, state: &mut FitState) -> Result<()> {
        if words.is_empty() {
            return Ok(());
        }
        let rows = u32::try_from(words.len() / STRIDE)?;
        let kind = kind as u32;
        let input = self.stream.clone_htod(&words)?;
        let mut call = self.stream.launch_builder(&self.fit);
        call.arg(&self.roots)
            .arg(&self.classes)
            .arg(&input)
            .arg(&rows)
            .arg(&kind)
            .arg(&state.weights)
            .arg(&mut state.donor)
            .arg(&mut state.contrast)
            .arg(&mut state.census);
        // Host validates the fixed-width rows and source positions before launch.
        // The device atomically updates 513-word bounded lanes and the fixed census.
        unsafe { call.launch(grid(rows as usize, 64))? };
        Ok(())
    }

    fn publish(&self, state: &mut FitState) -> Result<Vec<i16>> {
        let mut call = self.stream.launch_builder(&self.publish);
        call.arg(&state.donor)
            .arg(&state.contrast)
            .arg(&mut state.weights);
        // Threads are guarded against the fixed 513-coordinate extent.
        unsafe { call.launch(grid(STATE_WORDS, 64))? };
        Ok(self.stream.clone_dtoh(&state.weights)?)
    }

    fn fit_batches(
        &self,
        donor: &[Donor],
        positive: &[Row],
        absent: &[Row],
        batches: usize,
    ) -> Result<FitReadback> {
        let mut state = self.fit_state()?;
        let mut snapshots = Vec::with_capacity(batches);
        let mut weights = Vec::new();
        for batch in 0..batches {
            let donor_range = batch * donor.len() / batches..(batch + 1) * donor.len() / batches;
            let groups = positive.len() / 4;
            let group_range = batch * groups / batches..(batch + 1) * groups / batches;
            self.accumulate(
                donor[donor_range]
                    .iter()
                    .flat_map(Donor::row_words)
                    .collect(),
                StreamKind::Donor,
                &mut state,
            )?;
            self.accumulate(
                positive[group_range.start * 4..group_range.end * 4]
                    .iter()
                    .flat_map(|row| row.row_words(NONE))
                    .collect(),
                StreamKind::Contrast,
                &mut state,
            )?;
            self.accumulate(
                absent[group_range]
                    .iter()
                    .flat_map(|row| row.row_words(NONE))
                    .collect(),
                StreamKind::Absent,
                &mut state,
            )?;
            weights = self.publish(&mut state)?;
            snapshots.push(format!("{:x}", Sha256::digest(weight_bytes(&weights))));
        }
        Ok(FitReadback {
            weights,
            donor: self.stream.clone_dtoh(&state.donor)?,
            contrast: self.stream.clone_dtoh(&state.contrast)?,
            census: self.stream.clone_dtoh(&state.census)?,
            snapshots,
        })
    }

    fn score_rows(
        &self,
        writer: &WriterState,
        inputs: &[(Row, u32)],
        weights: &CudaSlice<i16>,
    ) -> Result<Vec<Decision>> {
        let mut decisions = Vec::with_capacity(inputs.len());
        if inputs.is_empty() {
            return Ok(decisions);
        }
        let scratch_bytes = usize::try_from(d11::kernel::host_record_sizes().0)?;
        let mut scratch = self
            .stream
            .alloc_zeros::<u8>(scratch_bytes * inputs.len().min(128))?;
        for rows in inputs.chunks(128) {
            let words = rows
                .iter()
                .flat_map(|(row, gate)| row.row_words(*gate))
                .collect::<Vec<_>>();
            let packed = self.stream.clone_htod(&words)?;
            let n = rows.len() as u32;
            let mut output = self.stream.alloc_zeros::<u32>(OUT_WORDS * rows.len())?;
            let mut call = self.stream.launch_builder(&self.score);
            call.arg(&writer.shape)
                .arg(&writer.table)
                .arg(&writer.cells)
                .arg(&writer.routes)
                .arg(&writer.gates)
                .arg(&writer.plans)
                .arg(&writer.trunk)
                .arg(&self.roots)
                .arg(&self.classes)
                .arg(weights)
                .arg(&packed)
                .arg(&n)
                .arg(&mut scratch)
                .arg(&mut output);
            // Each row receives disjoint initialized scratch and output records.
            unsafe { call.launch(grid(rows.len(), 32))? };
            let output = self.stream.clone_dtoh(&output)?;
            for ((row, marker_gate), raw) in rows.iter().zip(output.chunks_exact(OUT_WORDS)) {
                decisions.push(Decision::decode(row, *marker_gate, raw)?);
            }
        }
        Ok(decisions)
    }

    fn observed(
        &self,
        writer: &WriterState,
        inputs: &[(Row, u32)],
        weights: &[i16],
    ) -> Result<Vec<Decision>> {
        valid_weights(weights)?;
        let device_weights = self.stream.clone_htod(weights)?;
        if self.stream.clone_dtoh(&device_weights)? != weights {
            return Err("device loaded different signed weight words".into());
        }
        let mut observed = self.score_rows(writer, inputs, &device_weights)?;
        let mut lexical_inputs = Vec::new();
        let mut owners = Vec::new();
        for (index, ((row, _), observation)) in inputs.iter().zip(&observed).enumerate() {
            if row.prefix.len() == CONTEXT || !observation.arms.any_space() {
                continue;
            }
            let mut extra = row.clone();
            extra.prefix.push(SPACE);
            if row.raw_target != Some(SPACE) || row.target == SPACE {
                extra.gold_pos = None;
                extra.rival_pos = None;
            }
            lexical_inputs.push((extra, NONE));
            owners.push(index);
        }
        for (row, lexical) in
            owners
                .into_iter()
                .zip(self.score_rows(writer, &lexical_inputs, &device_weights)?)
        {
            let opening = &observed[row];
            if lexical.candidates != opening.candidates
                || lexical.winner_pos != opening.winner_pos
                || lexical.no_read_score != opening.no_read_score
            {
                return Err("offset-one read changed the causal factual score set".into());
            }
            observed[row].arms.save_lexical(lexical.arms);
        }
        Ok(observed)
    }
}

fn some(word: u32) -> Option<u32> {
    (word != NONE).then_some(word)
}

impl Decision {
    fn decode(row: &Row, marker: u32, raw: &[u32]) -> Result<Self> {
        let error = || {
            format!(
                "invalid CUDA decision for {}: {:?}",
                row.pair_id,
                &raw[..HEADER_WORDS]
            )
        };
        let scoped = raw[1] != NONE && raw[2] != NONE;
        let expected_unscoped =
            matches!(row.cohort.as_str(), "single_paragraph" | "finite_unscoped");
        if raw[0] != 0
            || (raw[1] == NONE) != (raw[2] == NONE)
            || scoped == expected_unscoped
            || (scoped
                && (raw[1] as usize != row.context_start || raw[2] as usize != row.context_end))
            || (!scoped && (raw[3] != NONE || raw[5] != 0 || raw[7] != NONE || raw[8] != NONE))
        {
            return Err(error().into());
        }
        let bounds = if scoped {
            row.context_start..row.context_end
        } else {
            0..0
        };
        let mut candidates = Vec::with_capacity(bounds.len());
        for pos in bounds {
            let offset = HEADER_WORDS + pos * CANDIDATE_WORDS;
            let record = &raw[offset..offset + CANDIDATE_WORDS];
            let eligible = record[0] == 1;
            if record[0] > 1
                || eligible != some(record[2]).is_some()
                || (eligible && !(1..=CONTEXT as u32).contains(&record[2]))
                || (eligible
                    && !record[4..]
                        .iter()
                        .enumerate()
                        .all(|(slot, &value)| value / 32 == slot as u32))
                || record[3] > u16::MAX as u32
            {
                return Err(error().into());
            }
            candidates.push(Candidate {
                pos: pos as u32,
                id: row.prefix[pos],
                eligible,
                score: eligible.then_some(record[1] as i32),
                rank: some(record[2]),
                features: if eligible {
                    Some(record[4..].try_into()?)
                } else {
                    None
                },
                geometry: record[3],
            });
        }
        let blank_first = row.prefix.last() == Some(&COLON)
            && row.raw_target == Some(SPACE)
            && row.target != SPACE;
        let read_arm = |index: usize, source_pos: Option<u32>| -> Result<Arm> {
            let offset = 9 + index * ARM_WORDS;
            let arm = &raw[offset..offset + ARM_WORDS];
            if arm[0] != 0
                || arm[1] >= VOCAB
                || arm[2] >= VOCAB
                || source_pos.is_some_and(|pos| row.prefix[pos as usize] != arm[1])
                || (source_pos.is_none() && marker != NONE && arm[1] != marker)
            {
                return Err(error().into());
            }
            Ok(Arm {
                immediate: Emission {
                    source_pos,
                    gate_id: arm[1],
                    emitted_id: arm[2],
                    first_admitted_cell: some(arm[3]),
                },
                lexical: None,
            })
        };
        let winner_pos = some(raw[3]);
        let arms = Arms {
            learned: read_arm(0, winner_pos)?,
            correct: read_arm(1, row.gold_pos.filter(|_| scoped && !blank_first))?,
            swapped: read_arm(2, row.rival_pos.filter(|_| scoped && !blank_first))?,
            absent: read_arm(3, None)?,
            marker: read_arm(4, None)?,
        };
        if arms.absent != arms.marker {
            return Err("forced NoRead and ordinary marker fallback disagree".into());
        }
        Ok(Self {
            parsed_start: some(raw[1]),
            parsed_end: some(raw[2]),
            no_read_score: raw[6] as i32,
            winner_pos,
            best_score: (raw[5] != 0).then_some(raw[4] as i32),
            best_ties: raw[5],
            candidates,
            gold_rank: some(raw[7]),
            rival_rank: some(raw[8]),
            arms,
        })
    }
}

impl Arms {
    fn any_space(&self) -> bool {
        [
            &self.learned,
            &self.correct,
            &self.swapped,
            &self.absent,
            &self.marker,
        ]
        .into_iter()
        .any(|arm| arm.immediate.emitted_id == SPACE)
    }

    fn save_lexical(&mut self, after_space: Self) {
        for (before, after) in [
            (&mut self.learned, after_space.learned),
            (&mut self.correct, after_space.correct),
            (&mut self.swapped, after_space.swapped),
            (&mut self.absent, after_space.absent),
            (&mut self.marker, after_space.marker),
        ] {
            if before.immediate.emitted_id == SPACE {
                before.lexical = Some(after.immediate);
            }
        }
    }
}

fn weight_bytes(weights: &[i16]) -> Vec<u8> {
    weights
        .iter()
        .flat_map(|weight| weight.to_le_bytes())
        .collect()
}

fn valid_weights(weights: &[i16]) -> Result<()> {
    if weights.len() != STATE_WORDS
        || weights
            .iter()
            .any(|&word| i32::from(word).abs() > WEIGHT_BOUND)
    {
        return Err("score weights violate native signed bounds".into());
    }
    Ok(())
}

fn save_weights(path: &Path, weights: &[i16]) -> Result<()> {
    valid_weights(weights)?;
    let mut file = create(path)?;
    file.write_all(&weight_bytes(weights))?;
    file.flush()?;
    Ok(())
}

fn read_weights(path: &Path) -> Result<Vec<i16>> {
    let bytes = fs::read(path)?;
    if bytes.len() != 2 * STATE_WORDS {
        return Err("wrong byte length for native score state".into());
    }
    let weights = bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    valid_weights(&weights)?;
    Ok(weights)
}

fn validate_group_inputs(train: &mut [Row], absent: &mut [Row]) -> Result<()> {
    train.sort_by(|a, b| {
        (&a.group_id, a.assignment, a.side).cmp(&(&b.group_id, b.assignment, b.side))
    });
    absent.sort_by(|a, b| a.group_id.cmp(&b.group_id));
    if train.len() != TRAIN_ROWS || absent.len() != TRAIN_ROWS / 4 {
        return Err("incorrect positive/absent training census".into());
    }
    for (group, missing) in train.chunks_exact(4).zip(absent.iter()) {
        if group.iter().enumerate().any(|(i, row)| {
            row.assignment != (i / 2) as u32
                || row.side != (i % 2) as u32
                || row.gold_pos.is_none()
                || row.rival_pos.is_none()
                || row.target == row.rival
                || row.group_id != missing.group_id
        }) || missing.gold_pos.is_some()
        {
            return Err(format!("bad crossed TRAIN group {}", missing.group_id).into());
        }
    }
    if absent
        .windows(2)
        .any(|pair| pair[0].group_id == pair[1].group_id)
    {
        return Err("TRAIN group id repeated".into());
    }
    Ok(())
}

pub fn fit(
    image: &str,
    base: &str,
    classes: &str,
    donor_file: &str,
    contrast_file: &str,
    absent_file: &str,
    out: &str,
) -> Result<()> {
    let donor = read_donor(donor_file)?;
    let mut contrasts = read_panel(contrast_file)?;
    let mut absent = read_panel(absent_file)?;
    validate_group_inputs(&mut contrasts, &mut absent)?;
    let out = Path::new(out);
    fs::create_dir(out)?;
    let device = Device::open(image, base, classes)?;
    let start = Instant::now();
    let fits = device.fit_batches(&donor, &contrasts, &absent, BATCHES)?;
    let elapsed = start.elapsed().as_millis();
    let [ordinary, unparsed, nonlexical, absent_donor, multiple, unique, contrast, hard, trained_absent, empty_absent, bad] =
        fits.census.as_slice()
    else {
        return Err("incorrect device census extent".into());
    };
    if ordinary + unparsed + nonlexical + absent_donor + multiple + unique != DONOR_ROWS as u32
        || *contrast != TRAIN_ROWS as u32
        || *hard > *contrast
        || trained_absent + empty_absent != (TRAIN_ROWS / 4) as u32
        || *bad != 0
    {
        return Err(format!("invalid training census {:?}", fits.census).into());
    }
    for slot in 0..FEATURES {
        let span = slot * 32..(slot + 1) * 32;
        if fits.donor[span.clone()].iter().sum::<i32>() != *unique as i32
            || fits.contrast[span].iter().sum::<i32>()
                != 2 * *contrast as i32 - 8 * *trained_absent as i32
        {
            return Err(format!("signed update mass mismatch in slot {slot}").into());
        }
    }
    let no_read = STATE_WORDS - 1;
    if fits.donor[no_read] != -(*unique as i32)
        || fits.contrast[no_read] != 8 * *trained_absent as i32 - 2 * *contrast as i32
    {
        return Err("incorrect NoRead update mass".into());
    }
    for ((&native, &d), &c) in fits.weights.iter().zip(&fits.donor).zip(&fits.contrast) {
        if d.abs() > DONOR_ROWS as i32
            || c.abs() > CONTRAST_BOUND
            || i32::from(native)
                != (c + d.clamp(-DONOR_CAP, DONOR_CAP)).clamp(-WEIGHT_BOUND, WEIGHT_BOUND)
        {
            return Err("device counters exceeded the registered fit bound".into());
        }
    }
    let state_path = out.join(STATE_FILE);
    save_weights(&state_path, &fits.weights)?;
    if read_weights(&state_path)? != fits.weights {
        return Err("lossless state reload differs".into());
    }
    let census = CENSUS_LABELS
        .into_iter()
        .zip(&fits.census)
        .collect::<BTreeMap<_, _>>();
    write_json(
        &out.join("fit.json"),
        &json!({
            "status": "pass",
            "elapsed_ms": elapsed,
            "batches": BATCHES,
            "donor_rows": DONOR_ROWS,
            "contrast_rows": TRAIN_ROWS,
            "absent_rows": absent.len(),
            "census": census,
            "donor_bound": DONOR_ROWS,
            "contrast_bound": CONTRAST_BOUND,
            "donor_contribution_bound": DONOR_CAP,
            "weight_bound": WEIGHT_BOUND,
            "score_bound": WEIGHT_BOUND * FEATURES as i32,
            "donor_raw": fits.donor,
            "contrast_raw": fits.contrast,
            "snapshot_sha256": fits.snapshots,
            "weights_sha256": format!("{:x}", Sha256::digest(weight_bytes(&fits.weights))),
        }),
    )
}

impl Baseline {
    fn check(&self, row: &Row) -> Result<()> {
        if self.group_id != row.group_id
            || self.pair_id != row.pair_id
            || self.cohort != row.cohort
            || self.assignment != row.assignment
            || self.side != row.side
            || self.target != row.target
            || self.rival != row.rival
            || self.selected >= VOCAB
            || self.emitted >= VOCAB
        {
            return Err(format!("mismatched frozen marker observation for {}", row.pair_id).into());
        }
        Ok(())
    }
}

fn wire(row: &Row, decision: Decision) -> Result<serde_json::Value> {
    let mut raw = serde_json::to_value(row)?;
    let analyzed = serde_json::to_value(decision)?;
    let raw = raw.as_object_mut().ok_or("row failed JSON encoding")?;
    let analyzed = analyzed
        .as_object()
        .ok_or("decision failed JSON encoding")?;
    raw.extend(analyzed.clone());
    Ok(serde_json::Value::Object(std::mem::take(raw)))
}

pub fn score(
    image: &str,
    base: &str,
    classes: &str,
    held: &str,
    marker: &str,
    weights: &str,
    out: &str,
) -> Result<()> {
    let rows = read_panel(held)?;
    let baseline = read_json::<Baseline>(marker)?;
    if rows.len() != baseline.len() {
        return Err("frozen marker rows do not cover the held inputs".into());
    }
    for (marker, row) in baseline.iter().zip(&rows) {
        marker.check(row)?;
    }
    let inputs = rows
        .into_iter()
        .zip(&baseline)
        .map(|(row, prior)| (row, prior.selected))
        .collect::<Vec<_>>();
    let out = Path::new(out);
    fs::create_dir(out)?;
    let device = Device::open(image, base, classes)?;
    let writer = device.writer_state(base)?;
    let state = Path::new(weights).join(STATE_FILE);
    let weights = read_weights(&state)?;
    let start = Instant::now();
    let observed = device.observed(&writer, &inputs, &weights)?;
    for (actual, marker) in observed.iter().zip(&baseline) {
        if actual.arms.marker.immediate.emitted_id != marker.emitted {
            return Err(format!("frozen CUDA writer mismatch for {}", marker.pair_id).into());
        }
    }
    let reversed = inputs.iter().rev().cloned().collect::<Vec<_>>();
    let replay = device.observed(&writer, &reversed, &read_weights(&state)?)?;
    if !observed.iter().eq(replay.iter().rev()) {
        return Err("held reverse-order device replay differs".into());
    }
    let observations = inputs
        .iter()
        .map(|(row, _)| row)
        .zip(observed)
        .map(|(row, decision)| wire(row, decision))
        .collect::<Result<Vec<_>>>()?;
    write_rows(&out.join("observations.jsonl"), &observations)?;
    write_json(
        &out.join("score.json"),
        &json!({
            "status": "pass",
            "rows": observations.len(),
            "elapsed_ms": start.elapsed().as_millis(),
            "reverse_order_exact": true,
            "frozen_marker_writer_exact": true,
            "weight_sha256": format!("{:x}", Sha256::digest(weight_bytes(&weights))),
        }),
    )
}
