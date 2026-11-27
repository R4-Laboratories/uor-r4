use super::{
    fs, paired_rows, peers, BufWriter, OpenOptions, PanelRow, Path, Result, Write, BASE_WORDS,
    CONTEXT, POSITION_WORDS, STAGES, STATE_WORDS, VOCAB, WORDS,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

const ROOTS_SHA: &str = "875a0857dfef840d88a64c90a5777a6154a1674cde08a71429e2cca93fb7bb5a";
const BALANCED_SHA: &str = "45d517f31074aa8061307a36f5d522e3af4bb464b434e5e6ff1779f4dd82c05e";
const STRIDE: usize = CONTEXT + 1;
const TRACE_HEADER: usize = 11;
const TRACE_TOKEN_WORDS: usize = 9;
const LINK_BITS_OFFSET: usize = TRACE_HEADER + CONTEXT * TRACE_TOKEN_WORDS;
const TRACE_STRIDE: usize = LINK_BITS_OFFSET + WORDS / 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Difference = 0,
    Position = 1,
    Link = 2,
}

impl Arm {
    fn fit_range(self) -> std::ops::Range<usize> {
        match self {
            Self::Difference => 0..WORDS,
            Self::Position => BASE_WORDS..POSITION_WORDS,
            Self::Link => POSITION_WORDS..STATE_WORDS,
        }
    }

    fn artifact_words(self) -> usize {
        match self {
            Self::Difference => BASE_WORDS,
            Self::Position => POSITION_WORDS,
            Self::Link => STATE_WORDS,
        }
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked(path: impl AsRef<Path>, expected: &str) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    if sha(&bytes) != expected {
        return Err(format!("digest mismatch at {}", path.display()).into());
    }
    Ok(bytes)
}

pub(super) fn load_weights(path: &str) -> Result<(String, Vec<i16>, usize)> {
    let bytes = fs::read(path)?;
    let count = bytes.len() / 2;
    if bytes.len() % 2 != 0 || !matches!(count, BASE_WORDS | POSITION_WORDS | STATE_WORDS) {
        return Err("expected three i16 stage segments, with position and link residual tables optional in that order".into());
    }
    let mut words = Vec::with_capacity(STATE_WORDS);
    words.extend(
        bytes
            .chunks_exact(2)
            .map(|word| i16::from_le_bytes([word[0], word[1]])),
    );
    words.resize(STATE_WORDS, 0);
    Ok((sha(&bytes), words, count))
}

fn frozen_roots(base: &str) -> Result<Vec<u32>> {
    let bytes = checked(Path::new(base).join("roots"), ROOTS_SHA)?;
    let roots: Vec<_> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    if bytes.len() != VOCAB as usize * 4 || roots.iter().any(|&root| root > 0xffff) {
        return Err("invalid packed native roots".into());
    }
    Ok(roots)
}

fn image(stream: &CudaStream, [path, digest]: [&str; 2]) -> Result<Arc<CudaModule>> {
    let cubin = checked(path, digest)?;
    if !cubin.starts_with(b"\x7fELF") {
        return Err("expected an AOT ELF cubin".into());
    }
    Ok(stream.context().load_module(Ptx::from_binary(cubin))?)
}

fn grid(count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(32), 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn pack<'a>(rows: impl IntoIterator<Item = &'a [u32]>) -> Result<Vec<u32>> {
    let mut packed = Vec::new();
    for row in rows {
        if row.is_empty() || row.len() > CONTEXT || row.iter().any(|&id| id >= VOCAB) {
            return Err("invalid bounded token row".into());
        }
        packed.push(u32::try_from(row.len())?);
        packed.extend_from_slice(row);
        packed.resize(packed.len() + CONTEXT - row.len(), 0);
    }
    if packed.is_empty() {
        return Err("empty GPU input".into());
    }
    Ok(packed)
}

struct Device {
    stream: Arc<CudaStream>,
    roots: CudaSlice<u32>,
    host_roots: Vec<u32>,
    scope_trace: CudaFunction,
    scope_fit: CudaFunction,
    score: CudaFunction,
    frozen_score: Option<CudaFunction>,
}

struct Batch {
    instructions: CudaSlice<u32>,
    packed: Vec<u32>,
    trace: Vec<i32>,
    scores: Vec<i32>,
}

impl Device {
    fn open(current: [&str; 2], frozen: Option<[&str; 2]>, roots: &[u32]) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        if ctx.compute_capability()? != (8, 9) || roots.len() != VOCAB as usize {
            return Err("expected registered sm_89 device and packed root table".into());
        }
        let stream = ctx.default_stream();
        let module = image(&stream, current)?;
        let frozen_score = frozen
            .map(|identity| -> Result<CudaFunction> {
                Ok(image(&stream, identity)?.load_function("choice_score")?)
            })
            .transpose()?;
        Ok(Self {
            roots: stream.clone_htod(roots)?,
            host_roots: roots.to_vec(),
            scope_trace: module.load_function("scope_trace")?,
            scope_fit: module.load_function("scope_fit")?,
            score: module.load_function("choice_score")?,
            frozen_score,
            stream,
        })
    }

    fn scores(
        &self,
        scorer: &CudaFunction,
        instructions: &CudaSlice<u32>,
        states: &CudaSlice<i16>,
        count: u32,
    ) -> Result<Vec<i32>> {
        let mut output = self.stream.alloc_zeros::<i32>(count as usize * STAGES)?;
        let mut call = self.stream.launch_builder(scorer);
        call.arg(&self.roots)
            .arg(instructions)
            .arg(states)
            .arg(&count)
            .arg(&mut output);
        // The packed rows passed here have bounded/vocabulary-checked tokens and
        // addressed states have exactly the declared native table sizes.
        unsafe { call.launch(grid(count))? };
        Ok(self.stream.clone_dtoh(&output)?)
    }

    fn record(&self, prefixes: &[u32], directives: &[u32], words: &[i16]) -> Result<Batch> {
        let count = u32::try_from(prefixes.len() / STRIDE)?;
        let states = self.stream.clone_htod(words)?;
        let prefixes = self.stream.clone_htod(prefixes)?;
        let mut instructions = self.stream.alloc_zeros::<u32>(count as usize * STRIDE)?;
        let mut trace = self
            .stream
            .alloc_zeros::<i32>(count as usize * TRACE_STRIDE)?;
        let mut call = self.stream.launch_builder(&self.scope_trace);
        call.arg(&self.roots)
            .arg(&prefixes)
            .arg(&states)
            .arg(&count)
            .arg(&mut instructions)
            .arg(&mut trace);
        // Each row is independently bounded and emits to disjoint fixed-width slices.
        unsafe { call.launch(grid(count))? };
        let trace = self.stream.clone_dtoh(&trace)?;
        if let Some((row, error)) = trace
            .chunks_exact(TRACE_STRIDE)
            .enumerate()
            .find(|(_, row)| row[0] != 0)
        {
            return Err(format!(
                "device directive extraction refused row {row} code {}",
                error[0]
            )
            .into());
        }
        let packed = self.stream.clone_dtoh(&instructions)?;
        if packed != directives {
            return Err(
                "device raw-prefix extraction diverged from historical scorer slice".into(),
            );
        }
        let scores = self.scores(&self.score, &instructions, &states, count)?;
        for (index, (detail, score)) in trace
            .chunks_exact(TRACE_STRIDE)
            .zip(scores.chunks_exact(STAGES))
            .enumerate()
        {
            if detail[4..7] != *score {
                return Err(
                    format!("trace did not reconstruct all scorer words at row {index}").into(),
                );
            }
        }
        if let Some(oracle) = &self.frozen_score {
            let frozen_input = self.stream.clone_htod(directives)?;
            let frozen_scores = self.scores(oracle, &frozen_input, &states, count)?;
            for (index, (detail, frozen)) in trace
                .chunks_exact(TRACE_STRIDE)
                .zip(frozen_scores.chunks_exact(STAGES))
                .enumerate()
            {
                if [detail[9], detail[5], detail[6]] != frozen {
                    return Err(format!(
                        "trace differs from untouched frozen scorer at row {index}"
                    )
                    .into());
                }
            }
        }
        Ok(Batch {
            instructions,
            packed,
            trace,
            scores,
        })
    }

    fn fit(
        &self,
        instructions: &CudaSlice<u32>,
        pairs: &[[u32; 2]],
        arm: Arm,
        words: &[i16],
    ) -> Result<(Vec<i16>, [u32; 5])> {
        let n = u32::try_from(pairs.len())?;
        let rows = instructions.len() / STRIDE;
        if n == 0 || pairs.iter().flatten().any(|&index| index as usize >= rows) {
            return Err("expected valid first-period fit pairs".into());
        }
        let indices = self.stream.clone_htod(pairs.as_flattened())?;
        let mode = arm as u32;
        let mut states = self.stream.clone_htod(words)?;
        let mut audit = self.stream.alloc_zeros::<u32>(5)?;
        let mut call = self.stream.launch_builder(&self.scope_fit);
        call.arg(&self.roots)
            .arg(instructions)
            .arg(&indices)
            .arg(&n)
            .arg(&mode)
            .arg(&mut states)
            .arg(&mut audit);
        // Exactly one CUDA worker owns the first-period native table and sequential update order.
        unsafe { call.launch(grid(1))? };
        let audit: [u32; 5] = self
            .stream
            .clone_dtoh(&audit)?
            .try_into()
            .map_err(|_| "invalid fit counters")?;
        if audit[0] != 0 {
            return Err("i16 fit overflowed; state refused".into());
        }
        Ok((self.stream.clone_dtoh(&states)?, audit))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Visit {
    id: u32,
    root: u32,
    root_bucket: u32,
    root_term: Option<i16>,
    adjacency_bucket: Option<u32>,
    adjacency_term: Option<i16>,
    scope_root_bucket: u32,
    scope_root_term: Option<i16>,
    scope_adjacency_bucket: Option<u32>,
    scope_adjacency_term: Option<i16>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct RowTrace<'a> {
    pair_id: &'a str,
    family: &'a str,
    side: u32,
    contrast: u32,
    fence: usize,
    instruction_start: usize,
    bias: i16,
    baseline_score: i32,
    residual_score: i32,
    link_score: i32,
    link_terms: Vec<Term>,
    scores: [i32; STAGES],
    peer_scores: [i32; STAGES],
    visits: Vec<Visit>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct Term {
    bucket: u32,
    weight: i16,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct PairTrace<'a> {
    pair_id: &'a str,
    family: &'a str,
    contrast: u32,
    stop_score: i32,
    copy_score: i32,
    stop_only: Vec<Term>,
    copy_only: Vec<Term>,
    same_support: bool,
    all_differing_weights_zero: bool,
    cancellation: bool,
    correct_flip: bool,
    reversed_flip: bool,
    wrong_stop_sign: bool,
    wrong_copy_sign: bool,
    stop_abstained: bool,
    copy_abstained: bool,
    same_decision: bool,
    score_delta: i32,
    scope_same_support: bool,
    scope_stop_only: Vec<Term>,
    scope_copy_only: Vec<Term>,
    link_same_support: bool,
    link_stop_only: Vec<Term>,
    link_copy_only: Vec<Term>,
    identical_instruction: bool,
}

fn support(visits: &[Visit], scope: bool) -> BTreeMap<u32, i16> {
    visits
        .iter()
        .flat_map(|visit| {
            if scope {
                [
                    (Some(visit.scope_root_bucket), visit.scope_root_term),
                    (visit.scope_adjacency_bucket, visit.scope_adjacency_term),
                ]
            } else {
                [
                    (Some(visit.root_bucket), visit.root_term),
                    (visit.adjacency_bucket, visit.adjacency_term),
                ]
            }
        })
        .filter_map(|(bucket, weight)| bucket.zip(weight))
        .collect()
}

fn only(left: &BTreeMap<u32, i16>, right: &BTreeMap<u32, i16>) -> Vec<Term> {
    left.iter()
        .filter(|(bucket, _)| !right.contains_key(bucket))
        .map(|(&bucket, &weight)| Term { bucket, weight })
        .collect()
}

fn sum(terms: &[Term]) -> i32 {
    terms.iter().map(|term| i32::from(term.weight)).sum()
}

fn link_support(trace: &RowTrace<'_>) -> BTreeMap<u32, i16> {
    trace
        .link_terms
        .iter()
        .map(|term| (term.bucket, term.weight))
        .collect()
}

fn analyze<'a>(stop: &RowTrace<'a>, copy: &RowTrace<'a>) -> Result<PairTrace<'a>> {
    let stop_base = support(&stop.visits, false);
    let copy_base = support(&copy.visits, false);
    let stop_only = only(&stop_base, &copy_base);
    let copy_only = only(&copy_base, &stop_base);
    let same_support = stop_only.is_empty() && copy_only.is_empty();
    let all_differing_weights_zero = !same_support
        && stop_only
            .iter()
            .chain(&copy_only)
            .all(|term| term.weight == 0);
    let score_delta = copy.baseline_score - stop.baseline_score;
    if score_delta != sum(&copy_only) - sum(&stop_only) || stop.bias != copy.bias {
        return Err("exclusive signed terms did not reconstruct the base score gap".into());
    }
    let stop_scope = support(&stop.visits, true);
    let copy_scope = support(&copy.visits, true);
    let scope_stop_only = only(&stop_scope, &copy_scope);
    let scope_copy_only = only(&copy_scope, &stop_scope);
    if copy.residual_score - stop.residual_score != sum(&scope_copy_only) - sum(&scope_stop_only) {
        return Err("exclusive signed terms did not reconstruct the residual score gap".into());
    }
    let stop_link = link_support(stop);
    let copy_link = link_support(copy);
    let link_stop_only = only(&stop_link, &copy_link);
    let link_copy_only = only(&copy_link, &stop_link);
    if copy.link_score - stop.link_score != sum(&link_copy_only) - sum(&link_stop_only) {
        return Err("exclusive signed terms did not reconstruct the link score gap".into());
    }
    let (stop_score, copy_score) = (stop.scores[0], copy.scores[0]);
    Ok(PairTrace {
        pair_id: stop.pair_id,
        family: stop.family,
        contrast: stop.contrast,
        stop_score,
        copy_score,
        stop_only,
        copy_only,
        same_support,
        all_differing_weights_zero,
        cancellation: score_delta == 0 && !same_support && !all_differing_weights_zero,
        correct_flip: stop_score < 0 && copy_score > 0,
        reversed_flip: stop_score > 0 && copy_score < 0,
        wrong_stop_sign: stop_score > 0,
        wrong_copy_sign: copy_score < 0,
        stop_abstained: stop_score == 0,
        copy_abstained: copy_score == 0,
        same_decision: stop_score.signum() == copy_score.signum(),
        score_delta,
        scope_same_support: scope_stop_only.is_empty() && scope_copy_only.is_empty(),
        scope_stop_only,
        scope_copy_only,
        link_same_support: link_stop_only.is_empty() && link_copy_only.is_empty(),
        link_stop_only,
        link_copy_only,
        identical_instruction: stop
            .visits
            .iter()
            .map(|visit| visit.id)
            .eq(copy.visits.iter().map(|visit| visit.id)),
    })
}

fn term(word: i32) -> Result<Option<i16>> {
    if word == i32::MIN {
        return Ok(None);
    }
    Ok(Some(i16::try_from(word)?))
}

fn bucket(word: i32) -> Result<Option<u32>> {
    match word {
        -1 => Ok(None),
        0..=4095 => Ok(Some(word as u32)),
        _ => Err("invalid traced bucket".into()),
    }
}

fn hash(root: u16) -> u32 {
    u32::from((root ^ (root >> 7)) & 2047)
}

fn phase(distance: usize) -> u16 {
    (distance as u16).wrapping_mul(0x9e37)
}

fn validate_visits(trace: &RowTrace<'_>, words: &[i16], roots: &[u32]) -> Result<()> {
    use std::collections::BTreeSet;
    let mut seen = [BTreeSet::new(), BTreeSet::new()];
    let mut linked = BTreeSet::new();
    let mut prior = None::<u16>;
    let read_word = |seen: &mut BTreeSet<u32>, address: Option<u32>, offset: usize| {
        address
            .filter(|&address| seen.insert(address))
            .map(|address| words[offset + address as usize])
    };
    for (index, visit) in trace.visits.iter().enumerate() {
        let root = u16::try_from(roots[visit.id as usize])?;
        let anchor = phase(trace.visits.len() - index);
        let adjacent = prior.map(|previous| previous.rotate_left(5) ^ root);
        let base_root = hash(root);
        let base_pair = adjacent.map(|pair| 2048 + hash(pair));
        let pos_root = hash(root ^ anchor);
        let pos_pair = adjacent.map(|pair| 2048 + hash(pair ^ anchor));
        let expected = Visit {
            id: visit.id,
            root: u32::from(root),
            root_bucket: base_root,
            root_term: read_word(&mut seen[0], Some(base_root), 0),
            adjacency_bucket: base_pair,
            adjacency_term: read_word(&mut seen[0], base_pair, 0),
            scope_root_bucket: pos_root,
            scope_root_term: read_word(&mut seen[1], Some(pos_root), BASE_WORDS),
            scope_adjacency_bucket: pos_pair,
            scope_adjacency_term: read_word(&mut seen[1], pos_pair, BASE_WORDS),
        };
        if *visit != expected {
            return Err(format!(
                "trace visit/address/word disagreement for {} side {} at directive token {index}",
                trace.pair_id, trace.side
            )
            .into());
        }
        for delta in 2..=index.min(16) {
            let other_id = trace.visits[index - delta].id;
            let other = u16::try_from(roots[other_id as usize])?;
            linked.insert(hash(other.rotate_left(5) ^ root ^ phase(delta)));
            linked.insert(2048 + hash(root.rotate_left(5) ^ other ^ phase(delta)));
        }
        prior = Some(root);
    }
    let expected = linked.into_iter().map(|bucket| Term {
        bucket,
        weight: words[POSITION_WORDS + bucket as usize],
    });
    if !trace.link_terms.iter().copied().eq(expected) {
        return Err("link trace differs from the bounded host address/word reference".into());
    }
    Ok(())
}

fn trace_row<'a>(
    row: &'a PanelRow,
    index: usize,
    peer: usize,
    batch: &Batch,
    words: &[i16],
    roots: &[u32],
) -> Result<RowTrace<'a>> {
    let detail = &batch.trace[index * TRACE_STRIDE..][..TRACE_STRIDE];
    let directive = &batch.packed[index * STRIDE + 1..][..usize::try_from(detail[3])?];
    let visits = directive
        .iter()
        .zip(detail[TRACE_HEADER..LINK_BITS_OFFSET].chunks_exact(TRACE_TOKEN_WORDS))
        .map(|(&id, visit)| -> Result<Visit> {
            Ok(Visit {
                id,
                root: u32::try_from(visit[0])?,
                root_bucket: bucket(visit[1])?.ok_or("missing traced root bucket")?,
                adjacency_bucket: bucket(visit[2])?,
                scope_root_bucket: bucket(visit[3])?
                    .ok_or("missing traced residual root bucket")?,
                scope_adjacency_bucket: bucket(visit[4])?,
                root_term: term(visit[5])?,
                adjacency_term: term(visit[6])?,
                scope_root_term: term(visit[7])?,
                scope_adjacency_term: term(visit[8])?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut link_terms = Vec::new();
    for (word_index, &word) in detail[LINK_BITS_OFFSET..].iter().enumerate() {
        let mut seen = word as u32;
        while seen != 0 {
            let bucket = word_index * 32 + seen.trailing_zeros() as usize;
            link_terms.push(Term {
                bucket: u32::try_from(bucket)?,
                weight: words[POSITION_WORDS + bucket],
            });
            seen &= seen - 1;
        }
    }
    if sum(&link_terms) != detail[10] {
        return Err("device link support did not reconstruct link score".into());
    }
    let trace = RowTrace {
        pair_id: &row.pair_id,
        family: &row.family,
        side: row.side,
        contrast: row.contrast,
        fence: usize::try_from(detail[1])?,
        instruction_start: usize::try_from(detail[2])?,
        bias: i16::try_from(detail[7])?,
        baseline_score: detail[9],
        residual_score: detail[8],
        link_score: detail[10],
        link_terms,
        scores: batch.scores[index * STAGES..][..STAGES].try_into()?,
        peer_scores: batch.scores[peer * STAGES..][..STAGES].try_into()?,
        visits,
    };
    let (fence, start, _) = row.boundaries()?;
    if (trace.fence, trace.instruction_start) != (fence, start) {
        return Err("device prefix boundaries differ from historical scorer".into());
    }
    validate_visits(&trace, words, roots)?;
    for scope in [false, true] {
        let mut seen = std::collections::BTreeSet::new();
        let mut total = if scope { 0 } else { i32::from(trace.bias) };
        for visit in &trace.visits {
            let entries = if scope {
                [
                    (Some(visit.scope_root_bucket), visit.scope_root_term),
                    (visit.scope_adjacency_bucket, visit.scope_adjacency_term),
                ]
            } else {
                [
                    (Some(visit.root_bucket), visit.root_term),
                    (visit.adjacency_bucket, visit.adjacency_term),
                ]
            };
            for (bucket, term) in entries {
                if bucket.is_some_and(|bucket| seen.insert(bucket)) != term.is_some() {
                    return Err("device trace first-visit marker diverged".into());
                }
                total += i32::from(term.unwrap_or_default());
            }
        }
        let expected = if scope {
            trace.residual_score
        } else {
            trace.baseline_score
        };
        if total != expected {
            return Err("device trace terms did not sum to their stage-zero score".into());
        }
    }
    Ok(trace)
}

fn pack_panel(rows: &[&PanelRow]) -> Result<(Vec<u32>, Vec<u32>)> {
    let prefixes = pack(rows.iter().map(|row| row.prefix.as_slice()))?;
    let directives = rows
        .iter()
        .map(|row| row.boundaries().map(|(_, _, tokens)| tokens))
        .collect::<Result<Vec<_>>>()?;
    Ok((prefixes, pack(directives)?))
}

fn trace_panel<'a>(
    device: &Device,
    rows: &[&'a PanelRow],
    state: &[i16],
) -> Result<(Batch, Vec<RowTrace<'a>>)> {
    let (prefixes, directives) = pack_panel(rows)?;
    let batch = device.record(&prefixes, &directives, state)?;
    let mate = paired_rows(rows, true)?;
    let traces = rows
        .iter()
        .zip(mate)
        .enumerate()
        .map(|(index, (&row, other))| {
            trace_row(
                row,
                index,
                other as usize,
                &batch,
                state,
                &device.host_roots,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((batch, traces))
}

fn paired_traces<'a>(rows: &[&'a PanelRow], traces: &[RowTrace<'a>]) -> Result<Vec<PairTrace<'a>>> {
    let mate = paired_rows(rows, true)?;
    rows.iter()
        .zip(mate)
        .enumerate()
        .filter(|(_, (row, _))| row.side == 0)
        .map(|(index, (_, other))| analyze(&traces[index], &traces[other as usize]))
        .collect()
}

fn write_rows(path: &Path, rows: &[impl Serialize]) -> Result<()> {
    let mut output = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
    for row in rows {
        serde_json::to_writer(&mut output, row)?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}

pub(super) fn trace(
    [image, image_sha, frozen, frozen_sha]: [&str; 4],
    base: &str,
    rows: &[PanelRow],
    input_sha: &str,
    weights: &str,
    output: &Path,
) -> Result<()> {
    let (weights_sha, words, word_count) = load_weights(weights)?;
    let device = Device::open(
        [image, image_sha],
        Some([frozen, frozen_sha]),
        &frozen_roots(base)?,
    )?;
    let forward: Vec<_> = rows.iter().collect();
    let backward: Vec<_> = rows.iter().rev().collect();
    let (_, traces) = trace_panel(&device, &forward, &words)?;
    let (_, mut reverse) = trace_panel(&device, &backward, &words)?;
    reverse.reverse();
    if traces != reverse {
        return Err("forward and reverse GPU prefix/feature traces differ".into());
    }
    let pairs = paired_traces(&forward, &traces)?;
    fs::create_dir(output)?;
    write_rows(&output.join("rows.jsonl"), &traces)?;
    write_rows(&output.join("pairs.jsonl"), &pairs)?;
    let receipt = serde_json::json!({
        "pass": true, "rows": rows.len(), "pairs": pairs.len(), "fit_words": word_count,
        "input_sha256": input_sha, "weights_sha256": weights_sha, "roots_sha256": ROOTS_SHA,
        "image_sha256": image_sha, "frozen_image_sha256": frozen_sha,
        "differentials": ["raw_prefix_boundary", "exact_address_and_signed_word_per_visit",
            "exact_frozen_score_words",
            "exact_candidate_score_words", "forward_reverse_full_traces"],
        "identical_support": pairs.iter().filter(|pair| pair.same_support).count(),
        "zero_differential_weights": pairs.iter().filter(|pair| pair.all_differing_weights_zero).count(),
        "cancellations": pairs.iter().filter(|pair| pair.cancellation).count(),
        "scoped_identical_support": pairs.iter().filter(|pair| pair.scope_same_support).count(),
        "linked_identical_support": pairs.iter().filter(|pair| pair.link_same_support).count(),
        "identical_instruction_slices": pairs.iter().filter(|pair| pair.identical_instruction).count(),
        "rows_sha256": sha(&fs::read(output.join("rows.jsonl"))?),
        "pairs_sha256": sha(&fs::read(output.join("pairs.jsonl"))?)
    });
    fs::write(
        output.join("trace.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}

fn labels(rows: &[&PanelRow]) -> Result<Vec<[u32; 2]>> {
    let mate = peers(rows)?;
    rows.iter()
        .zip(mate)
        .enumerate()
        .filter(|(_, (row, _))| row.contrast == 0 && row.side == 0)
        .map(|(index, (_, peer))| Ok([u32::try_from(index)?, peer]))
        .collect()
}

pub(super) fn fit(
    image: [&str; 2],
    base: &str,
    rows: &[PanelRow],
    input_sha: &str,
    weights: &str,
    arm: &str,
    output: &Path,
) -> Result<()> {
    let mode = match arm {
        "scope" => Arm::Position,
        "difference" => Arm::Difference,
        "link" => Arm::Link,
        _ => return Err("fit arm must be link, scope or difference".into()),
    };
    let (weights_sha, initial, count) = load_weights(weights)?;
    if weights_sha != BALANCED_SHA || count != BASE_WORDS {
        return Err("fit must start from the exact registered #4322 balanced weights".into());
    }
    let rows: Vec<_> = rows.iter().collect();
    let device = Device::open(image, None, &frozen_roots(base)?)?;
    let pairs = labels(&rows)?;
    let (before, before_traces) = trace_panel(&device, &rows, &initial)?;
    let (state, audit) = device.fit(&before.instructions, &pairs, mode, &initial)?;
    let (repeat, repeat_audit) = device.fit(&before.instructions, &pairs, mode, &initial)?;
    let active = mode.fit_range();
    if state != repeat
        || audit != repeat_audit
        || initial[..active.start] != state[..active.start]
        || initial[active.end..] != state[active.end..]
    {
        return Err("repeatability or frozen-state boundary failed".into());
    }
    let (_, after_traces) = trace_panel(&device, &rows, &state)?;
    let words = mode.artifact_words();
    let bytes: Vec<_> = state[..words]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let changes = initial
        .iter()
        .zip(&state)
        .filter(|(left, right)| left != right)
        .count();
    let correct_pairs = |traces: &[RowTrace<'_>]| {
        pairs
            .iter()
            .filter(|[stop, copy]| {
                traces[*stop as usize].scores[0] < 0 && traces[*copy as usize].scores[0] > 0
            })
            .count()
    };
    let receipt = serde_json::json!({
        "pass": true, "arm": arm, "rows": rows.len(), "pairs": pairs.len(),
        "epochs": 8, "margin": 8, "changed_words": changes,
        "fitted_words": words, "fitted_bytes": bytes.len(),
        "before_correct_train_pairs": correct_pairs(&before_traces),
        "after_correct_train_pairs": correct_pairs(&after_traces),
        "stop_exclusive_updates": audit[1], "copy_exclusive_updates": audit[2],
        "unresolved_margin_pair_visits": audit[3], "unresolved_same_support_visits": audit[4],
        "input_sha256": input_sha, "starting_weights_sha256": weights_sha,
        "weights_sha256": sha(&bytes), "roots_sha256": ROOTS_SHA, "image_sha256": image[1],
        "later_stages_byte_identical": true, "initial_bias_byte_identical": true,
        "repeat_identical": true
    });
    fs::create_dir(output)?;
    fs::write(output.join("weights.i16le"), bytes)?;
    fs::write(
        output.join("train.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}

mod finite;
pub(super) use finite::finite;
