// Reuse the audited #4318 loader, launch ABIs, and native-proposal checks.
// These local types satisfy its stand-alone driver's data contract.
#![allow(dead_code)]

include!("../../../codex-copy-id/device/src/gpu.rs");

const ARMS: usize = 8;
const ARM_NAMES: [&str; ARMS] = [
    "generate",
    "learned",
    "correct",
    "swapped",
    "same_id_wrong_occurrence",
    "invalid",
    "missing",
    "marker",
];
const CORRECT_COPY: usize = 2;
const COPY_SHA: &str = "51f6ffd519f6f75a2126d438366827f0184358c697f2550bf6bcb17ff6175576";

pub(super) struct Row {
    pair_id: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    gold_pos: Option<u32>,
    rival_pos: Option<u32>,
    wrong_same_id_pos: Option<u32>,
}

impl Row {
    fn opening(row: &PanelRow) -> Self {
        Self {
            pair_id: row.pair_id.clone(),
            prefix: row.prefix.clone(),
            target: row.target,
            rival: row.target,
            gold_pos: Some(row.gold_pos),
            rival_pos: None,
            wrong_same_id_pos: None,
        }
    }

    fn self_feed(&self, emitted: u32) -> Self {
        let mut prefix = self.prefix[usize::from(self.prefix.len() == CONTEXT)..].to_vec();
        prefix.push(emitted);
        Self {
            pair_id: self.pair_id.clone(),
            prefix,
            target: self.target,
            rival: self.rival,
            gold_pos: None,
            rival_pos: None,
            wrong_same_id_pos: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct WriterEvent {
    read_pos: Option<u32>,
    claim_pos: Option<u32>,
    gate_id: u32,
    source_id: Option<u32>,
    source_distance: Option<u32>,
    source_admitted: bool,
    minimum_distance: u32,
    first_admitted_cell: Option<u32>,
    native_proposal: u32,
    committed_id: u32,
    copied: bool,
    read_code: u32,
    recurrence_code: u32,
    causal_successor: Option<(u32, u32)>,
    read_vector: [i32; 32],
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Arm {
    first: WriterEvent,
    successor: Option<WriterEvent>,
    terminal: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub(super) struct Observation<'a> {
    pair_id: &'a str,
    prefix: &'a [u32],
    arms: BTreeMap<&'static str, Arm>,
}

const INSTRUCTION_STRIDE: usize = CONTEXT + 1;

struct EndingKernels {
    train: CudaFunction,
    pack: CudaFunction,
    score: CudaFunction,
    commit: CudaFunction,
}

pub(super) struct EndingDevice {
    stream: Arc<CudaStream>,
    kernels: EndingKernels,
    roots: CudaSlice<u32>,
}

struct Instructions {
    records: Vec<u32>,
    count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Commit {
    id: u32,
    source: usize,
}

impl Instructions {
    fn pack<'a>(rows: impl ExactSizeIterator<Item = &'a [u32]>) -> Result<Self> {
        let count = u32::try_from(rows.len())?;
        let mut records = vec![0; rows.len() * INSTRUCTION_STRIDE];
        for (row, record) in rows.zip(records.chunks_exact_mut(INSTRUCTION_STRIDE)) {
            if row.len() > CONTEXT {
                return Err("device instruction exceeds native context".into());
            }
            record[0] = u32::try_from(row.len())?;
            record[1..=row.len()].copy_from_slice(row);
        }
        Ok(Self { records, count })
    }

    fn from_panel(rows: &[&PanelRow]) -> Result<Self> {
        let slices: Vec<_> = rows
            .iter()
            .map(|row| row.spans().map(|(_, span)| span))
            .collect::<Result<_>>()?;
        Self::pack(slices.into_iter())
    }
}

impl EndingKernels {
    fn open(stream: &Arc<CudaStream>, path: &str) -> Result<Self> {
        let bytes = fs::read(path)?;
        if !bytes.starts_with(b"\x7fELF") {
            return Err(format!("expected AOT ELF module at {path}").into());
        }
        let module = stream.context().load_module(Ptx::from_binary(bytes))?;
        Ok(Self {
            train: module.load_function("ending_train")?,
            pack: module.load_function("ending_pack")?,
            score: module.load_function("ending_score")?,
            commit: module.load_function("ending_commit")?,
        })
    }
}

impl EndingDevice {
    fn context() -> Result<Arc<CudaStream>> {
        let context = CudaContext::new(0)?;
        if context.compute_capability()? != (8, 9) {
            return Err("registered code images require sm_89".into());
        }
        Ok(context.default_stream())
    }

    pub(super) fn open(image: &str, base: &str) -> Result<Self> {
        let (name, hash) = BASE[7];
        let frozen = checked(Path::new(base).join(name), hash)?;
        let roots = words(&frozen);
        if frozen.len() != VOCAB as usize * 4 || roots.iter().any(|&root| root > 0xffff) {
            return Err("invalid packed native roots".into());
        }
        Self::from_roots(Self::context()?, image, &roots)
    }

    fn from_roots(stream: Arc<CudaStream>, image: &str, roots: &[u32]) -> Result<Self> {
        Ok(Self {
            kernels: EndingKernels::open(&stream, image)?,
            roots: stream.clone_htod(roots)?,
            stream,
        })
    }

    fn accumulate(&self, instructions: &Instructions, deltas: &[i32]) -> Result<CudaSlice<i32>> {
        if deltas.len() != instructions.count as usize || deltas.is_empty() {
            return Err("invalid fit instruction/target cardinality".into());
        }
        let records = self.stream.clone_htod(&instructions.records)?;
        let deltas = self.stream.clone_htod(deltas)?;
        let mut accumulator = self.stream.alloc_zeros::<i32>(WORDS)?;
        let mut call = self.stream.launch_builder(&self.kernels.train);
        call.arg(&self.roots)
            .arg(&records)
            .arg(&deltas)
            .arg(&instructions.count)
            .arg(&mut accumulator);
        // Validated token IDs index the matching root table. One thread per
        // instruction applies bounded signed atomic updates to WORDS buckets.
        unsafe { call.launch(grid(instructions.count as usize))? };
        Ok(accumulator)
    }

    fn pack(&self, accumulator: &CudaSlice<i32>) -> Result<Vec<i16>> {
        if accumulator.len() != WORDS {
            return Err("incorrect accumulator width".into());
        }
        let mut weights = self.stream.alloc_zeros::<i16>(WORDS)?;
        let mut refused = self.stream.alloc_zeros::<u32>(1)?;
        let mut call = self.stream.launch_builder(&self.kernels.pack);
        call.arg(accumulator).arg(&mut weights).arg(&mut refused);
        // One thread per bucket and a single atomic refusal word.
        unsafe { call.launch(grid(WORDS))? };
        if self.stream.clone_dtoh(&refused)? != [0] {
            return Err("signed i16 instruction-state overflow".into());
        }
        let weights = self.stream.clone_dtoh(&weights)?;
        let exact = self.stream.clone_dtoh(accumulator)?;
        if !weights
            .iter()
            .zip(exact)
            .all(|(&weight, count)| i32::from(weight) == count)
        {
            return Err("packed instruction state differs from fitted counts".into());
        }
        Ok(weights)
    }

    pub(super) fn fit(&self, rows: &[&PanelRow]) -> Result<(Vec<i16>, usize, usize)> {
        let deltas: Vec<_> = rows
            .iter()
            .map(|row| {
                row.ideal()
                    .map(|ideal| if ideal == PERIOD { 1 } else { -1 })
            })
            .collect::<Result<_>>()?;
        let period = deltas.iter().filter(|&&delta| delta == 1).count();
        let stop = deltas.len() - period;
        if period == 0 || period != stop {
            return Err("training requires balanced nonempty stop and period classes".into());
        }
        let accum = self.accumulate(&Instructions::from_panel(rows)?, &deltas)?;
        Ok((self.pack(&accum)?, stop, period))
    }

    fn scores(&self, instructions: &Instructions, weights: &[i16]) -> Result<Vec<i32>> {
        if weights.len() != WORDS {
            return Err("incorrect instruction-state width".into());
        }
        let input = self.stream.clone_htod(&instructions.records)?;
        let weights = self.stream.clone_htod(weights)?;
        let mut scores = self
            .stream
            .alloc_zeros::<i32>(instructions.count as usize)?;
        let mut call = self.stream.launch_builder(&self.kernels.score);
        call.arg(&self.roots)
            .arg(&input)
            .arg(&weights)
            .arg(&instructions.count)
            .arg(&mut scores);
        // Prefix-derived instruction tokens and the frozen fitted weights are
        // the only prediction inputs to this launch.
        unsafe { call.launch(grid(instructions.count as usize))? };
        Ok(self.stream.clone_dtoh(&scores)?)
    }

    fn commits(
        &self,
        scores: &[i32],
        peers: &[u32],
        ideals: &[u32],
        verified: &[u32],
        native: &[u32],
    ) -> Result<Vec<Commit>> {
        let len = scores.len();
        if len == 0
            || [peers.len(), ideals.len(), verified.len(), native.len()]
                .iter()
                .any(|&n| n != len)
            || peers.iter().any(|&index| index as usize >= len)
        {
            return Err("invalid second-step launch shape".into());
        }
        let count = u32::try_from(len)?;
        let scores = self.stream.clone_htod(scores)?;
        let peers = self.stream.clone_htod(peers)?;
        let ideals = self.stream.clone_htod(ideals)?;
        let verified = self.stream.clone_htod(verified)?;
        let native = self.stream.clone_htod(native)?;
        let mut commits = self
            .stream
            .alloc_zeros::<u32>(len * ENDING_ARMS.len() * 2)?;
        let mut call = self.stream.launch_builder(&self.kernels.commit);
        call.arg(&scores)
            .arg(&peers)
            .arg(&ideals)
            .arg(&verified)
            .arg(&native)
            .arg(&count)
            .arg(&mut commits);
        // Pair indices are checked, and there are exactly two output words
        // per arm per row. Scores and oracle labels are separate arguments.
        unsafe { call.launch(grid(len))? };
        self.stream
            .clone_dtoh(&commits)?
            .chunks_exact(2)
            .enumerate()
            .map(|(index, pair)| {
                let source = pair[1] as usize;
                if pair[0] >= VOCAB
                    || (source != 0
                        && (source != index % ENDING_ARMS.len()
                            || !matches!(pair[0], EOS | PERIOD)))
                {
                    return Err("device emitted an invalid ending or source".into());
                }
                Ok(Commit {
                    id: pair[0],
                    source,
                })
            })
            .collect()
    }
}

fn load_weights(path: &str) -> Result<Vec<i16>> {
    let bytes = fs::read(path)?;
    if bytes.len() != WORDS * 2 {
        return Err("expected 8192 bytes of signed little-endian instruction weights".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|word| i16::from_le_bytes([word[0], word[1]]))
        .collect())
}

fn opening_verified(row: &PanelRow, event: &WriterEvent) -> bool {
    event.copied
        && event.committed_id == row.target
        && event.read_pos == Some(row.gold_pos)
        && event.claim_pos == Some(row.gold_pos)
}

fn verify_native(event: &WriterEvent) -> Result<()> {
    if event.copied || event.claim_pos.is_some() || event.committed_id != event.native_proposal {
        return Err("copy state persisted into ordinary continuation".into());
    }
    Ok(())
}

fn mismatch(steps: &[Step], expected: &[u32]) -> Option<usize> {
    steps
        .iter()
        .enumerate()
        .find(|(index, step)| expected.get(*index) != Some(&step.writer.committed_id))
        .map(|(index, _)| index + 1)
        .or_else(|| (steps.len() < expected.len()).then_some(steps.len() + 1))
}

pub(super) fn replay_endings<'a>(
    image: &str,
    files: &Files<'_>,
    weights: &str,
    rows: impl Iterator<Item = &'a PanelRow>,
) -> Result<Vec<EndingObservation<'a>>> {
    if digest(files.copy)? != COPY_SHA {
        return Err("frozen #4318 copy cubin digest mismatch".into());
    }
    let frozen = Device::open(files)?;
    let endings = EndingDevice::from_roots(
        frozen.stream.clone(),
        image,
        &frozen.stream.clone_dtoh(&frozen.roots)?,
    )?;
    let rows: Vec<_> = rows.collect();
    let peers = peers(&rows)?;
    let scores = endings.scores(&Instructions::from_panel(&rows)?, &load_weights(weights)?)?;
    let initial: Vec<_> = rows.iter().map(|row| Row::opening(row)).collect();
    let openings = frozen.events(&initial.iter().collect::<Vec<_>>(), ARMS)?;
    let openings: Vec<_> = openings
        .into_iter()
        .skip(CORRECT_COPY)
        .step_by(ARMS)
        .collect();
    let verified: Vec<_> = rows
        .iter()
        .zip(&openings)
        .map(|(row, event)| u32::from(opening_verified(row, event)))
        .collect();
    if let Some(index) = verified.iter().position(|&checked| checked == 0) {
        let row = rows[index];
        return Err(format!(
            "supplied copy changed at {} side {}: {:?}",
            row.pair_id, row.side, openings[index]
        )
        .into());
    }
    let seconds: Vec<_> = initial
        .iter()
        .zip(&openings)
        .map(|(row, event)| row.self_feed(event.committed_id))
        .collect();
    let proposals = frozen.events(&seconds.iter().collect::<Vec<_>>(), 1)?;
    let native: Vec<_> = proposals
        .iter()
        .map(|event| verify_native(event).map(|()| event.native_proposal))
        .collect::<Result<_>>()?;
    let ideals: Vec<_> = rows.iter().map(|row| row.ideal()).collect::<Result<_>>()?;
    let commits = endings.commits(&scores, &peers, &ideals, &verified, &native)?;

    // A distinct continuation is required for each distinct actual prefix.
    // Arms that commit the same ID share the identical frozen evaluation.
    let mut third_inputs = Vec::new();
    let mut third_indices = Vec::new();
    for (row, commits) in seconds.iter().zip(commits.chunks_exact(ENDING_ARMS.len())) {
        let mut row_inputs = BTreeMap::new();
        for commit in commits {
            let index = (commit.id != EOS).then(|| {
                *row_inputs.entry(commit.id).or_insert_with(|| {
                    let next = third_inputs.len();
                    third_inputs.push(row.self_feed(commit.id));
                    next
                })
            });
            third_indices.push(index);
        }
    }
    let third = frozen.events(&third_inputs.iter().collect::<Vec<_>>(), 1)?;
    for event in &third {
        verify_native(event)?;
    }

    let mut observations = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        let mut arms = BTreeMap::new();
        for (arm, name) in ENDING_ARMS.into_iter().enumerate() {
            let flat = index * ENDING_ARMS.len() + arm;
            let commit = commits[flat];
            let mut second = proposals[index].clone();
            second.committed_id = commit.id;
            let mut steps = vec![
                Step {
                    writer: openings[index].clone(),
                    selected_source: "copy",
                },
                Step {
                    writer: second,
                    selected_source: ENDING_ARMS[commit.source],
                },
            ];
            if let Some(third_index) = third_indices[flat] {
                steps.push(Step {
                    writer: third[third_index].clone(),
                    selected_source: "native",
                });
            }
            arms.insert(
                name,
                Trajectory {
                    terminal: steps
                        .last()
                        .is_some_and(|step| step.writer.committed_id == EOS),
                    first_mismatch: mismatch(&steps, &row.expected),
                    steps,
                },
            );
        }
        observations.push(EndingObservation {
            pair_id: &row.pair_id,
            side: row.side,
            prefix: &row.prefix,
            gold_pos: row.gold_pos,
            target: row.target,
            expected: &row.expected,
            family: &row.family,
            score: scores[index],
            peer_score: scores[peers[index] as usize],
            arms,
        });
    }
    Ok(observations)
}

pub(super) fn finite(image: &str) -> Result<()> {
    let roots = [0x0789, 0x1234, 0x00af];
    let device = EndingDevice::from_roots(EndingDevice::context()?, image, &roots)?;
    let instructions = Instructions::pack([&[0, 0, 1][..], &[1, 2, 1, 2]].into_iter())?;
    let weights = device.pack(&device.accumulate(&instructions, &[1, -1])?)?;
    let mut expected = vec![0; WORDS];
    for bucket in [1926, 3908, 2770] {
        expected[bucket] = 1;
    }
    for bucket in [174, 3745, 4059] {
        expected[bucket] = -1;
    }
    if weights != expected {
        return Err("finite fit disagrees on presence, orientation, or cancellation".into());
    }
    let held = Instructions::pack([&[0, 1][..], &[2, 1], &[1], &[1, 0]].into_iter())?;
    let scores = device.scores(&held, &weights)?;
    if scores != [2, -2, 0, 1] {
        return Err("finite positive, negative, and tied scores disagree".into());
    }
    let peers = [1, 0, 3, 2];
    let ideals = [EOS, PERIOD, EOS, PERIOD];
    let verified = [1, 1, 1, 0];
    let native = [110, 120, 130, 140];
    let commits = device.commits(&scores, &peers, &ideals, &verified, &native)?;
    let expected = [
        [(110, 0), (PERIOD, 1), (EOS, 2), (EOS, 3), (PERIOD, 4)],
        [(120, 0), (EOS, 1), (PERIOD, 2), (PERIOD, 3), (EOS, 4)],
        [(130, 0), (130, 0), (PERIOD, 2), (EOS, 3), (PERIOD, 4)],
        [(140, 0); 5],
    ];
    if !commits
        .iter()
        .map(|commit| (commit.id, commit.source))
        .eq(expected.into_iter().flatten())
    {
        return Err("finite matched swap, one-step sources, or copy guard disagree".into());
    }
    let zero = device.scores(&held, &vec![0; WORDS])?;
    if zero != [0; 4] {
        return Err("finite zero weights generated nonzero scores".into());
    }
    let zero_commits = device.commits(&zero, &peers, &ideals, &verified, &native)?;
    for (row, proposal) in zero_commits.chunks_exact(ENDING_ARMS.len()).zip(native) {
        if row[..3]
            != [Commit {
                id: proposal,
                source: 0,
            }; 3]
        {
            return Err("finite zero-weight choice changed native fallback".into());
        }
    }
    let mut limits = vec![0i32; WORDS];
    limits[0] = i32::from(i16::MIN);
    limits[WORDS - 1] = i32::from(i16::MAX);
    device.pack(&device.stream.clone_htod(&limits)?)?;
    for index in [0, WORDS - 1] {
        let mut overflow = limits.clone();
        overflow[index] += overflow[index].signum();
        let error = device
            .pack(&device.stream.clone_htod(&overflow)?)
            .err()
            .ok_or("finite signed i16 overflow was accepted")?;
        if error.to_string() != "signed i16 instruction-state overflow" {
            return Err(error);
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": "uor-r4.ending-device-finite/1",
            "fit_and_presence": "PASS",
            "scores": scores,
            "matched_swap_and_copy_guard": "PASS",
            "zero_scores": zero,
            "zero_fallback": "PASS",
            "signed_i16_bounds_and_overflow": "PASS",
            "ending_cubin_sha256": digest(image)?,
        })
    );
    Ok(())
}
