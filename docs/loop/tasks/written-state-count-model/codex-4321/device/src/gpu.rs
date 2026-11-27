// Source-pin the audited copy runner and use its unchanged frozen launch ABI.
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

#[derive(Clone)]
pub struct Row {
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
            target: row.ids[0],
            rival: row.ids[0],
            gold_pos: Some(row.positions[0]),
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
            target: EOS,
            rival: EOS,
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
    pub(super) committed_id: u32,
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
pub struct Observation<'a> {
    pair_id: &'a str,
    prefix: &'a [u32],
    arms: BTreeMap<&'static str, Arm>,
}

const INSTRUCTION_STRIDE: usize = CONTEXT + 1;
const COMMIT_STRIDE: usize = CONTEXT + 10;
const PERIOD_STAGE: u32 = 4;
const SOURCES: [&str; 4] = ["native", "verified_copy", "selected_eos", "supplied_period"];

struct Kernel {
    train: CudaFunction,
    score: CudaFunction,
    commit: CudaFunction,
    roots: CudaSlice<u32>,
}

#[derive(Debug, PartialEq, Eq)]
struct Commit {
    id: u32,
    source: u32,
    next: u32,
    refusal: u32,
}

impl Commit {
    fn step(
        self,
        mut writer: WriterEvent,
        stage: u32,
        score: Option<i32>,
        pos: u32,
        id: u32,
    ) -> Step {
        writer.committed_id = self.id;
        let requested = self.source == 1 || self.refusal == 3;
        Step {
            writer,
            selected_source: SOURCES[self.source as usize],
            stage: (stage < STAGES as u32).then_some(stage),
            action_score: score,
            requested_pos: requested.then_some(pos),
            requested_id: requested.then_some(id),
            refusal: self.refusal,
        }
    }
}

struct Instructions {
    rows: CudaSlice<u32>,
    count: u32,
}

impl Instructions {
    fn from_slices(stream: &Arc<CudaStream>, slices: &[&[u32]]) -> Result<Self> {
        let count = u32::try_from(slices.len())?;
        if slices.is_empty()
            || slices.iter().any(|slice| {
                slice.is_empty() || slice.len() > CONTEXT || slice.iter().any(|&id| id >= VOCAB)
            })
        {
            return Err("invalid instructions for GPU scoring".into());
        }
        let mut packed = vec![0; slices.len() * INSTRUCTION_STRIDE];
        for (slice, record) in slices
            .iter()
            .zip(packed.chunks_exact_mut(INSTRUCTION_STRIDE))
        {
            record[0] = u32::try_from(slice.len())?;
            record[1..=slice.len()].copy_from_slice(slice);
        }
        Ok(Self {
            rows: stream.clone_htod(&packed)?,
            count,
        })
    }

    fn from_panel(stream: &Arc<CudaStream>, rows: &[&PanelRow]) -> Result<Self> {
        let slices = rows
            .iter()
            .map(|row| row.boundaries().map(|(_, _, directive)| directive))
            .collect::<Result<Vec<_>>>()?;
        Self::from_slices(stream, &slices)
    }
}

fn context() -> Result<Arc<CudaStream>> {
    let context = CudaContext::new(0)?;
    if context.compute_capability()? != (8, 9) {
        return Err("registered choice image requires sm_89".into());
    }
    Ok(context.default_stream())
}

fn frozen_roots(base: &str) -> Result<Vec<u32>> {
    let (name, sha) = BASE[7];
    let input = checked(Path::new(base).join(name), sha)?;
    let roots = words(&input);
    if input.len() != VOCAB as usize * 4 || roots.iter().any(|&root| root > 0xffff) {
        return Err("invalid packed native roots".into());
    }
    Ok(roots)
}

impl Kernel {
    fn open(stream: &Arc<CudaStream>, path: &str, sha: &str, roots: &[u32]) -> Result<Self> {
        let bytes = checked(path, sha)?;
        if !bytes.starts_with(b"\x7fELF") {
            return Err(format!("expected AOT ELF module at {path}").into());
        }
        let module = stream.context().load_module(Ptx::from_binary(bytes))?;
        Ok(Self {
            train: module.load_function("choice_train")?,
            score: module.load_function("choice_score")?,
            commit: module.load_function("choice_commit")?,
            roots: stream.clone_htod(roots)?,
        })
    }

    fn fit(
        &self,
        stream: &Arc<CudaStream>,
        instructions: &Instructions,
        metadata: &[[u32; 4]],
    ) -> Result<(CudaSlice<i16>, Vec<u32>)> {
        if metadata.len() != instructions.count as usize
            || metadata
                .iter()
                .any(|row| row[0] >= instructions.count || row[1] > 2 || row[2] > 1 || row[3] > 1)
        {
            return Err("invalid metadata for GPU fit".into());
        }
        let metadata = stream.clone_htod(metadata.as_flattened())?;
        let mut state = stream.alloc_zeros::<i16>(STATE_WORDS)?;
        let mut audit = stream.alloc_zeros::<u32>(STAGES * 3)?;
        let mut call = stream.launch_builder(&self.train);
        call.arg(&self.roots)
            .arg(&instructions.rows)
            .arg(&metadata)
            .arg(&instructions.count)
            .arg(&mut state)
            .arg(&mut audit);
        // Exactly one worker for each disjoint stage segment. Token indices,
        // metadata peer offsets and all buffer widths were validated above.
        unsafe { call.launch(grid(STAGES))? };
        let audit = stream.clone_dtoh(&audit)?;
        if audit.chunks_exact(3).any(|stage| stage[0] != 0) {
            return Err("signed native training update overflowed; state refused".into());
        }
        Ok((state, audit))
    }

    fn scores(
        &self,
        stream: &Arc<CudaStream>,
        instructions: &Instructions,
        state: &CudaSlice<i16>,
    ) -> Result<CudaSlice<i32>> {
        let mut scores = stream.alloc_zeros::<i32>(instructions.count as usize * STAGES)?;
        let mut call = stream.launch_builder(&self.score);
        call.arg(&self.roots)
            .arg(&instructions.rows)
            .arg(state)
            .arg(&instructions.count)
            .arg(&mut scores);
        // One bounded row of instruction buckets and three i32 scores per thread.
        unsafe { call.launch(grid(instructions.count as usize))? };
        Ok(scores)
    }

    fn select(
        &self,
        stream: &Arc<CudaStream>,
        records: &[u32],
        scores: &CudaSlice<i32>,
    ) -> Result<Vec<Commit>> {
        let len = records.len() / COMMIT_STRIDE;
        let count = u32::try_from(len)?;
        let score_rows = u32::try_from(scores.len() / STAGES)?;
        if len == 0
            || records.len() % COMMIT_STRIDE != 0
            || score_rows == 0
            || scores.len() != score_rows as usize * STAGES
        {
            return Err("invalid commit ABI shape".into());
        }
        let input = stream.clone_htod(records)?;
        let mut output = stream.alloc_zeros::<u32>(len * 4)?;
        let mut call = stream.launch_builder(&self.commit);
        call.arg(&input)
            .arg(scores)
            .arg(&score_rows)
            .arg(&count)
            .arg(&mut output);
        // Each ABI record has a full bounded prefix; the kernel checks row,
        // stage, and candidate indices before each dynamic read.
        unsafe { call.launch(grid(len))? };
        stream
            .clone_dtoh(&output)?
            .chunks_exact(4)
            .map(|words| {
                if words[0] >= VOCAB
                    || words[1] >= SOURCES.len() as u32
                    || (words[2] != NONE && words[2] >= STAGES as u32)
                    || words[3] > 3
                {
                    return Err("invalid CUDA commit event".into());
                }
                Ok(Commit {
                    id: words[0],
                    source: words[1],
                    next: words[2],
                    refusal: words[3],
                })
            })
            .collect()
    }
}

pub(super) fn load_weights(path: &str) -> Result<Vec<i16>> {
    let bytes = fs::read(path)?;
    if bytes.len() != STATE_WORDS * 2 {
        return Err(
            "expected exactly three stage segments of 4096 i16 weights and one i16 bias".into(),
        );
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|word| i16::from_le_bytes([word[0], word[1]]))
        .collect())
}

fn choice(score: i32) -> &'static str {
    match score.cmp(&0) {
        std::cmp::Ordering::Less => "eos",
        std::cmp::Ordering::Greater => "copy",
        std::cmp::Ordering::Equal => "native",
    }
}

#[derive(Default, Serialize)]
struct StageFit {
    reachable: u32,
    exact: u32,
    abstained: u32,
    individual_updates: u32,
    difference_updates: u32,
}

#[derive(Default, Serialize)]
struct PairFit {
    oriented: u32,
    reversed: u32,
    score_ties: u32,
    same_decision: u32,
}

pub(super) fn train(
    image: &str,
    image_sha: &str,
    base: &str,
    rows: &[PanelRow],
    input_sha256: &str,
    out: &Path,
) -> Result<()> {
    let started = Instant::now();
    let stream = context()?;
    let kernel = Kernel::open(&stream, image, image_sha, &frozen_roots(base)?)?;
    let rows: Vec<_> = rows.iter().collect();
    let peers = peers(&rows)?;
    let instructions = Instructions::from_panel(&stream, &rows)?;
    let metadata: Vec<_> = rows
        .iter()
        .zip(&peers)
        .map(|(row, &peer)| [peer, row.requested(), row.side, row.contrast])
        .collect();
    let (state, audit) = kernel.fit(&stream, &instructions, &metadata)?;
    let scores = stream.clone_dtoh(&kernel.scores(&stream, &instructions, &state)?)?;
    let mut stages = [
        StageFit::default(),
        StageFit::default(),
        StageFit::default(),
    ];
    let mut contrasts = [PairFit::default(), PairFit::default()];
    for (index, (row, &peer)) in rows.iter().zip(&peers).enumerate() {
        let requested = row.requested() as usize;
        for (stage, stats) in stages.iter_mut().enumerate().take(requested + 1) {
            let raw = scores[index * STAGES + stage];
            stats.reachable += 1;
            stats.exact += u32::from(if stage == requested { raw < 0 } else { raw > 0 });
            stats.abstained += u32::from(raw == 0);
        }
        if row.side == 0 {
            let contrast = row.contrast as usize;
            let stats = &mut contrasts[contrast];
            let (stop, copy) = (
                scores[index * STAGES + contrast],
                scores[peer as usize * STAGES + contrast],
            );
            stats.oriented += u32::from(stop < 0 && copy > 0);
            stats.reversed += u32::from(stop > 0 && copy < 0);
            stats.score_ties += u32::from(stop == copy);
            stats.same_decision += u32::from(choice(stop) == choice(copy));
        }
    }
    for (stats, audit) in stages.iter_mut().zip(audit.chunks_exact(3)) {
        stats.individual_updates = audit[1];
        stats.difference_updates = audit[2];
    }
    let weights_path = out.join("weights.i16le");
    let bytes: Vec<u8> = stream
        .clone_dtoh(&state)?
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect();
    let weights_sha256 = format!("{:x}", Sha256::digest(&bytes));
    fs::write(&weights_path, bytes)?;
    let receipt = serde_json::json!({
        "training_order": "panel order then side-0 panel order for pair differences, independently by stage",
        "epochs": 8, "margin": 8, "pair_margin": 16, "initial_state": "zero",
        "rows": rows.len(), "pairs": rows.len() / 2, "stages": stages, "contrasts": contrasts,
        "input_sha256": input_sha256, "image_sha256": image_sha, "roots_sha256": BASE[7].1,
        "weights_sha256": weights_sha256, "elapsed_ms": started.elapsed().as_millis()
    });
    fs::write(out.join("train.json"), serde_json::to_vec_pretty(&receipt)?)?;
    println!("{receipt}");
    Ok(())
}

fn native_only(event: &WriterEvent) -> Result<()> {
    if event.copied || event.claim_pos.is_some() || event.committed_id != event.native_proposal {
        return Err("copy state persisted into the fresh native writer".into());
    }
    Ok(())
}

pub(super) struct Replayer {
    device: Device,
    kernel: Kernel,
    state: CudaSlice<i16>,
}

struct Active {
    row: usize,
    arm: usize,
    stage: u32,
    prefix: Row,
    steps: Vec<Step>,
}

fn opening_step(writer: WriterEvent) -> Step {
    Step {
        writer,
        selected_source: "copy",
        stage: None,
        action_score: None,
        requested_pos: None,
        requested_id: None,
        refusal: 0,
    }
}

fn duplicate_step(step: &Step) -> Step {
    Step {
        writer: step.writer.clone(),
        selected_source: step.selected_source,
        stage: step.stage,
        action_score: step.action_score,
        requested_pos: step.requested_pos,
        requested_id: step.requested_id,
        refusal: step.refusal,
    }
}

impl Replayer {
    pub(super) fn open(files: &Files<'_>, image: &str, sha: &str, state: &[i16]) -> Result<Self> {
        if digest(files.copy)? != COPY_SHA || state.len() != STATE_WORDS {
            return Err("frozen #4318 copy cubin digest or choice state width mismatch".into());
        }
        let device = Device::open(files)?;
        let kernel = Kernel::open(&device.stream, image, sha, &frozen_roots(files.base)?)?;
        let state = device.stream.clone_htod(state)?;
        Ok(Self {
            device,
            kernel,
            state,
        })
    }

    pub(super) fn replay<'a>(
        &mut self,
        rows: impl Iterator<Item = &'a PanelRow>,
    ) -> Result<Vec<ChoiceObservation<'a>>> {
        let rows: Vec<_> = rows.collect();
        let peers = peers(&rows)?;
        let stream = &self.device.stream;
        let instructions = Instructions::from_panel(stream, &rows)?;
        let gpu_scores = self.kernel.scores(stream, &instructions, &self.state)?;
        let scores = stream.clone_dtoh(&gpu_scores)?;
        let fences = rows
            .iter()
            .map(|row| row.boundaries().map(|(fence, _, _)| fence))
            .collect::<Result<Vec<_>>>()?;
        let originals: Vec<_> = rows.iter().map(|row| Row::opening(row)).collect();
        let opening: Vec<_> = self
            .device
            .events(&originals.iter().collect::<Vec<_>>(), ARMS)?
            .into_iter()
            .skip(CORRECT_COPY)
            .step_by(ARMS)
            .collect();
        let second_inputs: Vec<_> = originals
            .iter()
            .zip(&opening)
            .map(|(input, open)| input.self_feed(open.committed_id))
            .collect();
        let second = self
            .device
            .events(&second_inputs.iter().collect::<Vec<_>>(), 1)?;
        let mut period_records = vec![0; rows.len() * COMMIT_STRIDE];
        for (((row, open), proposal), record) in rows
            .iter()
            .zip(&opening)
            .zip(&second)
            .zip(period_records.chunks_exact_mut(COMMIT_STRIDE))
        {
            native_only(proposal)?;
            if !open.copied
                || open.committed_id != row.ids[0]
                || open.source_id != Some(row.ids[0])
                || open.claim_pos != Some(row.positions[0])
                || open.read_pos != Some(row.positions[0])
            {
                return Err(format!(
                    "opening source authentication failed at {} side {}",
                    row.pair_id, row.side
                )
                .into());
            }
            record[..4].copy_from_slice(&[proposal.native_proposal, PERIOD_STAGE, 0, 1]);
        }
        let periods = self.kernel.select(stream, &period_records, &gpu_scores)?;
        let mut active = Vec::new();
        for (row, (((prefix, open), writer), period)) in second_inputs
            .iter()
            .zip(opening)
            .zip(second)
            .zip(periods)
            .enumerate()
        {
            if period
                != (Commit {
                    id: PERIOD,
                    source: 3,
                    next: 0,
                    refusal: 0,
                })
            {
                return Err("CUDA changed the supplied period".into());
            }
            let post_period = prefix.self_feed(period.id);
            let initial_stage = period.next;
            let first = opening_step(open);
            let second = period.step(writer, PERIOD_STAGE, None, NONE, NONE);
            for arm in 0..ARMS_NAMES.len() {
                active.push(Active {
                    row,
                    arm,
                    stage: if arm == 0 { NONE } else { initial_stage },
                    prefix: post_period.clone(),
                    steps: vec![duplicate_step(&first), duplicate_step(&second)],
                });
            }
        }
        for _ in 2..CAP {
            let mut unique = BTreeMap::new();
            let mut inputs = Vec::new();
            let mut dispatch = Vec::new();
            for (index, state) in active.iter().enumerate() {
                if state
                    .steps
                    .last()
                    .is_some_and(|step| step.writer.committed_id == EOS)
                {
                    continue;
                }
                let writer_index = *unique
                    .entry((state.row, &state.prefix.prefix))
                    .or_insert_with(|| {
                        let index = inputs.len();
                        inputs.push(&state.prefix);
                        index
                    });
                dispatch.push((index, writer_index));
            }
            if dispatch.is_empty() {
                break;
            }
            let proposals = self.device.events(&inputs, 1)?;
            let mut records = vec![0; dispatch.len() * COMMIT_STRIDE];
            for (&(index, writer_index), record) in
                dispatch.iter().zip(records.chunks_exact_mut(COMMIT_STRIDE))
            {
                let state = &active[index];
                let row = rows[state.row];
                let peer = peers[state.row];
                let proposal = &proposals[writer_index];
                native_only(proposal)?;
                let (position, id) = if state.stage == NONE {
                    (NONE, NONE)
                } else {
                    let source = state.stage as usize + 1;
                    (row.positions[source], row.ids[source])
                };
                record[..10].copy_from_slice(&[
                    proposal.native_proposal,
                    state.stage,
                    state.arm as u32,
                    state.row as u32,
                    peer,
                    if state.arm == 3 { row.requested() } else { 0 },
                    if state.arm == 4 {
                        rows[peer as usize].requested()
                    } else {
                        0
                    },
                    position,
                    id,
                    fences[state.row] as u32,
                ]);
                record[10..10 + state.prefix.prefix.len()].copy_from_slice(&state.prefix.prefix);
            }
            let commits = self.kernel.select(stream, &records, &gpu_scores)?;
            for (((index, writer_index), commit), record) in dispatch
                .into_iter()
                .zip(commits)
                .zip(records.chunks_exact(COMMIT_STRIDE))
            {
                let state = &mut active[index];
                if matches!(commit.refusal, 1 | 2) || commit.source == 3 {
                    return Err("invalid CUDA release transition".into());
                }
                let score_index = match state.arm {
                    1 => state.row,
                    2 => peers[state.row] as usize,
                    _ => 0,
                };
                let score = (state.stage != NONE && matches!(state.arm, 1 | 2))
                    .then(|| scores[score_index * STAGES + state.stage as usize]);
                state.prefix = state.prefix.self_feed(commit.id);
                state.stage = commit.next;
                state.steps.push(commit.step(
                    proposals[writer_index].clone(),
                    record[1],
                    score,
                    record[7],
                    record[8],
                ));
            }
        }
        let mut results = Vec::with_capacity(rows.len());
        let mut active = active.into_iter();
        for (index, row) in rows.into_iter().enumerate() {
            let mut arms = BTreeMap::new();
            for name in ARMS_NAMES {
                let arm = active.next().ok_or("missing completed trajectory")?;
                arms.insert(name, Trajectory::new(arm.steps, &row.expected)?);
            }
            let stage = row.contrast;
            let own_score = scores[index * STAGES + stage as usize];
            let peer_score = scores[peers[index] as usize * STAGES + stage as usize];
            results.push(ChoiceObservation {
                row,
                common: Common {
                    stage,
                    own_score,
                    peer_score,
                    own_choice: choice(own_score),
                    peer_choice: choice(peer_score),
                },
                arms,
            });
        }
        Ok(results)
    }
}

pub(super) fn finite(path: &str, sha: &str) -> Result<serde_json::Value> {
    let stream = context()?;
    let mut roots = vec![0; VOCAB as usize];
    // IDs 121 and 122 intentionally occupy the same geometric bucket.
    roots[121] = 1;
    roots[122] = 1;
    roots[130] = 2;
    roots[131] = 4;
    roots[132] = 8;
    let kernel = Kernel::open(&stream, path, sha, &roots)?;
    let instructions =
        Instructions::from_slices(&stream, &[&[121, 121], &[130, 130], &[131], &[132]])?;
    let zero = stream.alloc_zeros::<i16>(STATE_WORDS)?;
    let zero = kernel.scores(&stream, &instructions, &zero)?;
    if stream.clone_dtoh(&zero)? != [0; 4 * STAGES] {
        return Err("zero state failed to abstain".into());
    }
    let (state, updates) = kernel.fit(
        &stream,
        &instructions,
        &[[1, 0, 0, 0], [0, 1, 1, 0], [3, 1, 0, 1], [2, 2, 1, 1]],
    )?;
    let trained = kernel.scores(&stream, &instructions, &state)?;
    let values = stream.clone_dtoh(&trained)?;
    // The second stage includes wrong-sign intermediate margins. A final
    // sign-only check would miss skipped updates there.
    let expected = [-9, -1, -4, 13, -9, -4, 9, -8, -4, 9, 9, -8];
    if values != expected {
        return Err(format!("planted signed fit is {values:?}, expected {expected:?}").into());
    }

    let cases: [([u32; 10], [u32; 4]); 13] = [
        ([73, 4, 0, 1, 0, 0, 0, NONE, NONE, 0], [PERIOD, 3, 0, 0]),
        ([73, 4, 0, 0, 0, 0, 0, NONE, NONE, 0], [73, 0, NONE, 1]),
        ([73, 0, 1, 0, 1, 0, 1, 1, 121, 4], [EOS, 2, NONE, 0]),
        ([73, 0, 1, 1, 0, 1, 0, 1, 121, 4], [121, 1, 1, 0]),
        ([73, 0, 2, 0, 1, 0, 1, 1, 121, 4], [121, 1, 1, 0]),
        ([73, 0, 2, 1, 0, 1, 0, 1, 121, 4], [EOS, 2, NONE, 0]),
        ([73, 1, 1, 3, 2, 2, 1, 2, 130, 4], [130, 1, 2, 0]),
        ([73, 2, 4, 3, 2, 2, 1, 3, 131, 4], [EOS, 2, NONE, 0]),
        ([73, 0, 3, 1, 0, 1, 0, 1, 122, 4], [73, 0, NONE, 3]),
        ([73, 0, 3, 1, 0, 1, 0, 0, 121, 4], [73, 0, NONE, 3]),
        ([73, 0, 3, 1, 0, 1, 0, 4, 121, 4], [73, 0, NONE, 3]),
        ([73, 0, 3, 1, 0, 1, 0, 5, 0, 6], [73, 0, NONE, 3]),
        ([EOS, NONE, 0, 0, 0, 0, 0, NONE, NONE, 0], [EOS, 0, NONE, 0]),
    ];
    let mut records = vec![0; cases.len() * COMMIT_STRIDE];
    for (record, (input, _)) in records.chunks_exact_mut(COMMIT_STRIDE).zip(cases) {
        record[..10].copy_from_slice(&input);
        record[10..15].copy_from_slice(&[122, 121, 130, 131, 121]);
    }
    let commits = kernel.select(&stream, &records, &trained)?;
    for (index, (event, (_, expected))) in commits.iter().zip(cases).enumerate() {
        if [event.id, event.source, event.next, event.refusal] != expected {
            return Err(
                format!("finite commit {index} is {event:?}, expected {expected:?}").into(),
            );
        }
    }
    let abstained = kernel.select(
        &stream,
        &records[2 * COMMIT_STRIDE..7 * COMMIT_STRIDE],
        &zero,
    )?;
    if abstained.iter().any(|event| {
        *event
            != (Commit {
                id: 73,
                source: 0,
                next: NONE,
                refusal: 0,
            })
    }) {
        return Err("zero release score failed to preserve native fallback and expiry".into());
    }
    Ok(serde_json::json!({
        "pass": true, "image_sha256": sha, "trained_scores": values, "stage_updates": updates,
        "controls": ["planted_paired_both_contrasts", "trained_vs_zero", "actual_swap",
            "supplied_period", "refused_opening", "same_root_wrong_id", "wrong_person_position",
            "original_fence", "zero_id", "native_eos", "native_fallback_expires"]
    }))
}
