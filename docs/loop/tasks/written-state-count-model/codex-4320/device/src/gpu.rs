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

const COMPLETION_STRIDE: usize = CONTEXT + 7;
const COMMIT_WORDS: usize = 4;
const TRACE_WORDS: usize = 1719;
const ROUTE_WORDS: usize = 44;
const CANDIDATE_WORDS: usize = 18;
const PROBES: usize = 7;
const SELECTED_SOURCES: [&str; 5] = [
    "native",
    "correct",
    "swapped",
    "supplied_next",
    "supplied_period",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Selected {
    id: u32,
    source: u32,
    terminal: bool,
    refusal: u32,
}

impl Selected {
    fn checked(self) -> Result<Self> {
        if self.refusal != 0 {
            return Err(format!("device selection rejected with code {}", self.refusal).into());
        }
        Ok(self)
    }

    fn step(self, mut writer: WriterEvent) -> Step {
        writer.committed_id = self.id;
        Step {
            writer,
            selected_source: SELECTED_SOURCES[self.source as usize],
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct RouteNode {
    node_idx: u32,
    action: u32,
    carried_root: u32,
    span: [u32; 2],
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct RoutedLevel {
    block: usize,
    level: usize,
    gate_code: u32,
    root: u32,
    nodes: Vec<RouteNode>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct RankedToken {
    id: u32,
    distance: u32,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Candidate {
    id: u32,
    score: u64,
    terms: [u32; 6],
    recurrence_count: u32,
    unigram: u32,
    in_pool: bool,
    cell: u32,
    distance: u32,
    admitted: bool,
    writer_rank: Option<u32>,
    distance_ties: u32,
    admitted_ties: u32,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub(super) struct ThirdTrace {
    read_code: u32,
    recurrence_code: u32,
    mixed_token: u32,
    gate_id: u32,
    writer_id: u32,
    winning_score: u64,
    pointer_predecessor: Option<(u32, u32)>,
    pointer_successor: Option<(u32, u32)>,
    pair_successor: Option<(u32, u32)>,
    pair_distance: Option<u32>,
    read_vector: [i32; 32],
    recurrence: [i32; 32],
    recurrence_sum: [i32; 32],
    recurrent_frames: [i32; 4],
    routed_nodes: Vec<RoutedLevel>,
    admitted_cells: Vec<u32>,
    shortlist: Vec<RankedToken>,
    candidates: Vec<Candidate>,
    sensitivity: [u32; 3],
    pool_size: u32,
    positive_candidates: u32,
    writer_candidates: u32,
}

fn joined_u64(lo: u32, hi: u32) -> u64 {
    u64::from(lo) | (u64::from(hi) << 32)
}

fn signed<const N: usize>(input: &[u32]) -> Result<[i32; N]> {
    let words: [u32; N] = input.try_into()?;
    Ok(words.map(|word| word as i32))
}

impl ThirdTrace {
    fn decode(row: &Row, writer: &WriterEvent, raw: &[u32]) -> Result<Self> {
        let [status, read_code, recurrence_code, mixed_token, gate_id, writer_id, orbit, pair, pair_distance, effective, lo, hi] =
            <[u32; 12]>::try_from(&raw[..12])?;
        if status != 0 {
            return Err(format!("third trace rejected {} with code {status}", row.pair_id).into());
        }
        let located = |pos: u32| -> Result<Option<(u32, u32)>> {
            optional(pos)
                .map(|pos| {
                    row.prefix
                        .get(pos as usize)
                        .map(|&id| (pos, id))
                        .ok_or_else(|| "trace source outside actual prefix".into())
                })
                .transpose()
        };
        let pointer_successor = located(effective)?;
        let pointer_predecessor = match pointer_successor {
            Some((pos, _)) => located(pos.checked_sub(1).ok_or("source lacked a predecessor")?)?,
            None => None,
        };
        let admitted_count = raw[144] as usize;
        let shortlist_count = raw[161] as usize;
        let candidate_count = raw[1586] as usize;
        if admitted_count > 16
            || !(1..=8).contains(&shortlist_count)
            || candidate_count > PROBES
            || read_code != writer.read_code
            || recurrence_code != writer.recurrence_code
            || gate_id != writer.gate_id
            || writer_id != writer.native_proposal
            || signed::<32>(&raw[108..140])? != writer.read_vector
            || located(orbit)? != writer.causal_successor
            || orbit != effective
            || raw[145] != writer.first_admitted_cell.unwrap_or(NONE)
            || raw[162] != writer_id
            || raw[163] != writer.minimum_distance
        {
            return Err(
                format!("diagnostic disagrees with frozen writer at {}", row.pair_id).into(),
            );
        }
        let mut routed_nodes = Vec::new();
        for (index, route) in raw[178..1586].chunks_exact(ROUTE_WORDS).enumerate() {
            if route[0] == 0 {
                continue;
            }
            let count = route[2] as usize;
            if count == 0 || count > 8 {
                return Err("invalid route count".into());
            }
            routed_nodes.push(RoutedLevel {
                block: index / 8,
                level: index % 8,
                gate_code: route[1],
                root: route[3],
                nodes: route[4..4 + count * 5]
                    .chunks_exact(5)
                    .map(|node| RouteNode {
                        node_idx: node[0],
                        action: node[1],
                        carried_root: node[2],
                        span: [node[3], node[4]],
                    })
                    .collect(),
            });
        }
        let shortlist = raw[162..162 + 2 * shortlist_count]
            .chunks_exact(2)
            .map(|token| RankedToken {
                id: token[0],
                distance: token[1],
            })
            .collect();
        let candidates = raw[1587..1587 + candidate_count * CANDIDATE_WORDS]
            .chunks_exact(CANDIDATE_WORDS)
            .map(|token| {
                Ok(Candidate {
                    id: token[0],
                    score: joined_u64(token[1], token[2]),
                    terms: token[3..9].try_into()?,
                    recurrence_count: token[9],
                    unigram: token[10],
                    in_pool: token[11] == 1,
                    cell: token[12],
                    distance: token[13],
                    admitted: token[14] == 1,
                    writer_rank: (token[15] != 0).then_some(token[15]),
                    distance_ties: token[16],
                    admitted_ties: token[17],
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            read_code,
            recurrence_code,
            mixed_token,
            gate_id,
            writer_id,
            winning_score: joined_u64(lo, hi),
            pointer_predecessor,
            pointer_successor,
            pair_successor: located(pair)?,
            pair_distance: optional(pair_distance),
            read_vector: signed(&raw[12..44])?,
            recurrence: signed(&raw[44..76])?,
            recurrence_sum: signed(&raw[76..108])?,
            recurrent_frames: signed(&raw[140..144])?,
            routed_nodes,
            admitted_cells: raw[145..145 + admitted_count].to_vec(),
            shortlist,
            candidates,
            sensitivity: raw[1713..1716].try_into()?,
            pool_size: raw[1716],
            positive_candidates: raw[1717],
            writer_candidates: raw[1718],
        })
    }
}

struct CompletionKernels {
    select: CudaFunction,
    trace: CudaFunction,
}

impl CompletionKernels {
    fn open(stream: &Arc<CudaStream>, path: &str, sha: &str) -> Result<Self> {
        let bytes = checked(path, sha)?;
        if !bytes.starts_with(b"\x7fELF") {
            return Err(format!("expected AOT ELF module at {path}").into());
        }
        let module = stream.context().load_module(Ptx::from_binary(bytes))?;
        Ok(Self {
            select: module.load_function("postperiod_select")?,
            trace: module.load_function("postperiod_trace")?,
        })
    }

    fn select(
        &self,
        stream: &Arc<CudaStream>,
        records: &[u32],
        step: u32,
    ) -> Result<Vec<Selected>> {
        let rows = records.len() / COMPLETION_STRIDE;
        let count = u32::try_from(rows)?;
        if count == 0 || records.len() % COMPLETION_STRIDE != 0 || !matches!(step, 2 | 3) {
            return Err("invalid completion selection shape".into());
        }
        let inputs = stream.clone_htod(records)?;
        let mut outputs = stream.alloc_zeros::<u32>(rows * COMPLETION_ARMS.len() * COMMIT_WORDS)?;
        let mut call = stream.launch_builder(&self.select);
        call.arg(&inputs).arg(&count).arg(&step).arg(&mut outputs);
        // Each row has its own output, and the kernel validates pair and source bounds.
        unsafe { call.launch(grid(rows))? };
        let outputs = stream.clone_dtoh(&outputs)?;
        let selected = outputs
            .chunks_exact(COMPLETION_ARMS.len() * COMMIT_WORDS)
            .flat_map(|row| {
                row.chunks_exact(COMMIT_WORDS)
                    .take(if step == 2 { 1 } else { 4 })
            });
        selected
            .map(|words| {
                if words[0] >= VOCAB
                    || words[1] as usize >= SELECTED_SOURCES.len()
                    || words[2] != u32::from(words[0] == EOS)
                    || words[3] > 3
                {
                    return Err("invalid device completion decision".into());
                }
                Ok(Selected {
                    id: words[0],
                    source: words[1],
                    terminal: words[2] == 1,
                    refusal: words[3],
                })
            })
            .collect()
    }

    fn traces(
        &self,
        device: &Device,
        rows: &[&PanelRow],
        inputs: &[Row],
        native: &[WriterEvent],
    ) -> Result<Vec<ThirdTrace>> {
        let mut result = Vec::with_capacity(inputs.len());
        for ((rows, inputs), native) in rows
            .chunks(CHUNK)
            .zip(inputs.chunks(CHUNK))
            .zip(native.chunks(CHUNK))
        {
            let count = u32::try_from(inputs.len())?;
            let mut records = vec![0; inputs.len() * COMPLETION_STRIDE];
            for (((row, input), event), record) in rows
                .iter()
                .zip(inputs)
                .zip(native)
                .zip(records.chunks_exact_mut(COMPLETION_STRIDE))
            {
                record[0] = u32::try_from(input.prefix.len())?;
                record[1] = if let Some(pos) = event.read_pos {
                    *input
                        .prefix
                        .get(pos as usize)
                        .ok_or("ordinary read outside prefix")?
                } else {
                    NONE
                };
                record[2] = row.next_id;
                record[7..7 + input.prefix.len()].copy_from_slice(&input.prefix);
            }
            let stream = &device.stream;
            let records = stream.clone_htod(&records)?;
            let mut scratch = stream.alloc_zeros::<u32>(inputs.len() * SCRATCH_WORDS)?;
            let mut raw = stream.alloc_zeros::<u32>(inputs.len() * TRACE_WORDS)?;
            let mut call = stream.launch_builder(&self.trace);
            call.arg(&device.shape)
                .arg(&device.table)
                .arg(&device.cells)
                .arg(&device.routes)
                .arg(&device.gates)
                .arg(&device.plans)
                .arg(&device.trunk)
                .arg(&records)
                .arg(&count)
                .arg(&mut scratch)
                .arg(&mut raw);
            // The exact frozen table dimensions and cubins are verified before launch.
            unsafe { call.launch(grid(inputs.len()))? };
            let raw = stream.clone_dtoh(&raw)?;
            for ((row, event), words) in
                inputs.iter().zip(native).zip(raw.chunks_exact(TRACE_WORDS))
            {
                result.push(ThirdTrace::decode(row, event, words)?);
            }
        }
        Ok(result)
    }
}

fn native_only(event: &WriterEvent) -> Result<()> {
    if event.copied || event.claim_pos.is_some() || event.committed_id != event.native_proposal {
        return Err("copy state persisted into ordinary continuation".into());
    }
    Ok(())
}

pub(super) fn replay_completion<'a>(
    files: &Files<'_>,
    image_path: &str,
    image_sha: &str,
    rows: impl Iterator<Item = &'a PanelRow>,
) -> Result<Vec<CompletionObservation<'a>>> {
    if digest(files.copy)? != COPY_SHA {
        return Err("frozen #4318 copy cubin digest mismatch".into());
    }
    let device = Device::open(files)?;
    let kernels = CompletionKernels::open(&device.stream, image_path, image_sha)?;
    let rows: Vec<_> = rows.collect();
    let peers = peers(&rows)?;
    let input: Vec<_> = rows.iter().map(|row| Row::opening(row)).collect();
    let opening: Vec<_> = device
        .events(&input.iter().collect::<Vec<_>>(), ARMS)?
        .into_iter()
        .skip(CORRECT_COPY)
        .step_by(ARMS)
        .collect();
    let second_inputs: Vec<_> = input
        .iter()
        .zip(&opening)
        .map(|(row, event)| row.self_feed(event.committed_id))
        .collect();
    let second = device.events(&second_inputs.iter().collect::<Vec<_>>(), 1)?;
    let mut records = vec![0; rows.len() * COMPLETION_STRIDE];
    for (((row, open), proposal), record) in rows
        .iter()
        .zip(&opening)
        .zip(&second)
        .zip(records.chunks_exact_mut(COMPLETION_STRIDE))
    {
        native_only(proposal)?;
        if !open.copied
            || open.committed_id != row.target
            || open.claim_pos != Some(row.gold_pos)
            || open.read_pos != Some(row.gold_pos)
        {
            return Err(format!(
                "supplied opening changed at {} side {}",
                row.pair_id, row.side
            )
            .into());
        }
        record[0] = proposal.native_proposal;
        record[6] = 1;
    }
    let period = kernels.select(&device.stream, &records, 2)?;
    let mut third_inputs = Vec::with_capacity(rows.len());
    for (input, selection) in second_inputs.iter().zip(&period) {
        let selection = selection.checked()?;
        if selection.id != PERIOD || selection.source != 4 || selection.terminal {
            return Err("device changed supplied period".into());
        }
        third_inputs.push(input.self_feed(selection.id));
    }
    let third = device.events(&third_inputs.iter().collect::<Vec<_>>(), 1)?;
    for (((row, &peer), proposal), record) in rows
        .iter()
        .zip(&peers)
        .zip(&third)
        .zip(records.chunks_exact_mut(COMPLETION_STRIDE))
    {
        native_only(proposal)?;
        let (fence, _) = row.boundaries()?;
        record[..7].copy_from_slice(&[
            proposal.native_proposal,
            row.side,
            peer,
            row.next_pos,
            row.next_id,
            u32::try_from(fence)?,
            u32::try_from(row.prefix.len())?,
        ]);
        record[7..7 + row.prefix.len()].copy_from_slice(&row.prefix);
    }
    let selections = kernels.select(&device.stream, &records, 3)?;
    let traces = kernels.traces(&device, &rows, &third_inputs, &third)?;

    // Share a fourth evaluation only when the actual token prefixes are equal.
    let mut fourth_inputs = Vec::new();
    let mut fourth_indices = Vec::with_capacity(selections.len());
    for (input, arms) in third_inputs
        .iter()
        .zip(selections.chunks_exact(COMPLETION_ARMS.len()))
    {
        let mut unique = BTreeMap::new();
        for selection in arms {
            let selection = selection.checked()?;
            fourth_indices.push((!selection.terminal).then(|| {
                *unique.entry(selection.id).or_insert_with(|| {
                    let index = fourth_inputs.len();
                    fourth_inputs.push(input.self_feed(selection.id));
                    index
                })
            }));
        }
    }
    let fourth = device.events(&fourth_inputs.iter().collect::<Vec<_>>(), 1)?;
    for event in &fourth {
        native_only(event)?;
    }

    let mut observations = Vec::with_capacity(rows.len());
    for ((index, row), third_trace) in rows.into_iter().enumerate().zip(traces) {
        let mut arms = BTreeMap::new();
        for (arm, name) in COMPLETION_ARMS.into_iter().enumerate() {
            let flat = index * COMPLETION_ARMS.len() + arm;
            let mut steps = vec![
                Step {
                    writer: opening[index].clone(),
                    selected_source: "copy",
                },
                period[index].step(second[index].clone()),
                selections[flat].step(third[index].clone()),
            ];
            if let Some(last) = fourth_indices[flat] {
                steps.push(Step {
                    writer: fourth[last].clone(),
                    selected_source: "native",
                });
            }
            arms.insert(name, Trajectory::new(steps, &row.expected)?);
        }
        observations.push(CompletionObservation {
            row,
            third_trace,
            arms,
        });
    }
    Ok(observations)
}

pub(super) fn finite(path: &str, sha: &str) -> Result<()> {
    let context = CudaContext::new(0)?;
    if context.compute_capability()? != (8, 9) {
        return Err("registered code images require sm_89".into());
    }
    let stream = context.default_stream();
    let kernels = CompletionKernels::open(&stream, path, sha)?;
    let mut rows = vec![0; 4 * COMPLETION_STRIDE];
    let planted = [
        [73, 0, 1, 1, 121, 2, 3, 99, 121, 14],
        [35, 1, 0, 1, 121, 2, 3, 99, 121, 14],
        [49, 0, 3, 1, 130, 2, 3, 99, 130, 14],
        [EOS, 1, 2, 1, 130, 2, 3, 99, 130, 14],
    ];
    for (record, sample) in rows.chunks_exact_mut(COMPLETION_STRIDE).zip(planted) {
        record[..sample.len()].copy_from_slice(&sample);
    }
    let check = |records: &[u32], step, expected: &[(u32, u32, u32)]| -> Result<()> {
        let actual = kernels.select(&stream, records, step)?;
        let expect: Vec<_> = expected
            .iter()
            .map(|&(id, source, refusal)| Selected {
                id,
                source,
                terminal: id == EOS,
                refusal,
            })
            .collect();
        if actual != expect {
            return Err(format!("finite selection mismatch: {actual:?} vs {expect:?}").into());
        }
        Ok(())
    };
    let valid = [
        (73, 0, 0),
        (EOS, 1, 0),
        (73, 0, 0),
        (EOS, 3, 0),
        (35, 0, 0),
        (35, 0, 0),
        (EOS, 2, 0),
        (121, 3, 0),
        (49, 0, 0),
        (EOS, 1, 0),
        (49, 0, 0),
        (EOS, 3, 0),
        (EOS, 0, 0),
        (EOS, 0, 0),
        (EOS, 2, 0),
        (130, 3, 0),
    ];
    check(&rows, 3, &valid)?;
    let mut wrong = valid;
    wrong[7] = (35, 0, 3);
    for (column, value) in [(3, NONE), (4, 122), (3, 2)] {
        let mut changed = rows.clone();
        changed[COMPLETION_STRIDE + column] = value;
        check(&changed, 3, &wrong)?;
    }
    let mut bad_peer = valid;
    bad_peer[1..4].fill((73, 0, 2));
    bad_peer[5..8].fill((35, 0, 2));
    rows[COMPLETION_STRIDE + 2] = 3;
    check(&rows, 3, &bad_peer)?;
    rows[6] = 1;
    check(
        &rows[..2 * COMPLETION_STRIDE],
        2,
        &[(PERIOD, 4, 0), (35, 0, 1)],
    )
}
