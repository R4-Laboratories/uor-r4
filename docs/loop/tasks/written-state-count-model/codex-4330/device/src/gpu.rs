use crate::{
    input::{eligible, pack, Geometry, Row},
    require, Result, CONTEXT, EPOCHS, FEATURES, NONE, STATE_WORDS,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx, PushKernelArg,
};
use serde::Serialize;
use std::{fs, path::Path, sync::Arc};

const ARMS: usize = 3;
const HEADER: usize = 11;
const CANDIDATE_WORDS: usize = FEATURES + 2;
const ARM_WORDS: usize = HEADER + CONTEXT * CANDIDATE_WORDS;
const FIT_WORDS: usize = 7;
const BATCH_SIZE: usize = 96;

pub struct Device {
    stream: Arc<CudaStream>,
    fit: CudaFunction,
    score: CudaFunction,
    roots: CudaSlice<u32>,
    classes: CudaSlice<u8>,
    pub name: String,
    pub image_sha: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct FitEvent {
    pub epoch: usize,
    pub case_id: String,
    pub gold_pos: Option<u16>,
    pub competing_pos: Option<u16>,
    pub desired_score: i32,
    pub competitor_score: Option<i32>,
    pub no_read_score: i16,
    pub margin: Option<i32>,
    pub update: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Fit {
    pub snapshots: Vec<Vec<i16>>,
    pub events: Vec<FitEvent>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct Candidate {
    pub pos: u16,
    pub id: u16,
    pub score: i32,
    pub features: [u16; FEATURES],
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct Observation {
    pub fact_bounds: Option<[u16; 2]>,
    pub query_bounds: Option<[u16; 2]>,
    pub winner_pos: Option<u16>,
    pub best_score: Option<i32>,
    pub best_ties: u16,
    pub no_read_score: i16,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct Decision {
    pub case_id: String,
    pub old_framed: Observation,
    pub expanded_frozen: Observation,
    pub expanded_fitted: Observation,
}

fn optional_position(word: u32) -> Result<Option<u16>> {
    if word == NONE {
        return Ok(None);
    }
    require(
        word < CONTEXT as u32,
        "GPU reported an out-of-range position",
    )?;
    Ok(Some(word as u16))
}

fn grid(rows: usize, width: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(width), 1, 1),
        block_dim: (width, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Device {
    pub fn open(image: &Path, geometry: &Geometry) -> Result<Self> {
        let image = fs::read(image)?;
        require(
            image.starts_with(b"\x7fELF"),
            "expected an ahead-of-time cubin ELF",
        )?;
        let image_sha = crate::input::sha(&image);
        let context = CudaContext::new(0)?;
        require(
            context.compute_capability()? == (8, 9),
            "device must be sm_89",
        )?;
        let name = context.name()?;
        let stream = context.default_stream();
        let module = context.load_module(Ptx::from_binary(image))?;
        Ok(Self {
            fit: module.load_function("fit_whole_turn")?,
            score: module.load_function("score_whole_turn")?,
            roots: stream.clone_htod(&geometry.roots)?,
            classes: stream.clone_htod(&geometry.classes)?,
            stream,
            name,
            image_sha,
        })
    }

    pub fn fit(&self, rows: &[Row], initial: &[i16]) -> Result<Fit> {
        let train = self.stream.clone_htod(&pack(rows, true))?;
        let length = u32::try_from(rows.len())?;
        let mut weights = self.stream.clone_htod(initial)?;
        let mut log = self
            .stream
            .alloc_zeros::<u32>(EPOCHS * rows.len() * FIT_WORDS)?;
        let mut snapshots = self.stream.alloc_zeros::<i16>(STATE_WORDS * EPOCHS)?;
        let mut call = self.stream.launch_builder(&self.fit);
        call.arg(&self.roots)
            .arg(&self.classes)
            .arg(&train)
            .arg(&length)
            .arg(&mut weights)
            .arg(&mut log)
            .arg(&mut snapshots);
        // Device and host share the explicit scalar and buffer types above.
        unsafe { call.launch(grid(1, 1))? };
        let log = self.stream.clone_dtoh(&log)?;
        let snapshots = self.stream.clone_dtoh(&snapshots)?;
        let weights = self.stream.clone_dtoh(&weights)?;
        require(
            snapshots[STATE_WORDS * (EPOCHS - 1)..] == weights,
            "last device snapshot differs from final learned state",
        )?;
        let snapshots = snapshots
            .chunks_exact(STATE_WORDS)
            .map(|state| {
                crate::input::check_weights(state)?;
                Ok(state.to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut events = Vec::with_capacity(rows.len() * EPOCHS);
        for (index, words) in log.chunks_exact(FIT_WORDS).enumerate() {
            let row = &rows[index % rows.len()];
            let gold_pos = optional_position(words[1])?;
            let competing_pos = optional_position(words[2])?;
            let desired_score = words[3] as i32;
            let no_read_score = i16::try_from(words[5] as i32)?;
            let competitor_score = (words[4] as i32 != i32::MIN).then_some(words[4] as i32);
            let margin = competitor_score.map(|score| desired_score - score);
            let update = words[6] == 1;
            let [start, end] = row.user_bounds;
            require(
                words[0] == 0
                    && gold_pos == row.target
                    && competing_pos
                        .is_none_or(|p| (start..end).contains(&p) && Some(p) != gold_pos)
                    && (gold_pos.is_some() || desired_score == i32::from(no_read_score))
                    && (gold_pos.is_none() || competitor_score.is_some())
                    && (gold_pos.is_some()
                        || competitor_score.is_some() == competing_pos.is_some())
                    && words[6] <= 1
                    && update == margin.is_some_and(|delta| delta < crate::MARGIN),
                &format!("malformed GPU fit event for {}", row.case_id),
            )?;
            events.push(FitEvent {
                epoch: 1 + index / rows.len(),
                case_id: row.case_id.clone(),
                gold_pos,
                competing_pos,
                desired_score,
                competitor_score,
                no_read_score,
                margin,
                update,
            });
        }
        Ok(Fit { snapshots, events })
    }

    pub fn score(
        &self,
        rows: &[&Row],
        initial: &[i16],
        fitted: &[i16],
        classes: &[u8],
    ) -> Result<Vec<Decision>> {
        let initial_device = self.stream.clone_htod(initial)?;
        let fitted_device = self.stream.clone_htod(fitted)?;
        let mut decisions = Vec::with_capacity(rows.len());
        for batch in rows.chunks(BATCH_SIZE) {
            let packed = self
                .stream
                .clone_htod(&pack(batch.iter().copied(), false))?;
            let length = batch.len() as u32;
            let mut out = self
                .stream
                .alloc_zeros::<u32>(batch.len() * ARMS * ARM_WORDS)?;
            let mut call = self.stream.launch_builder(&self.score);
            call.arg(&self.roots)
                .arg(&self.classes)
                .arg(&packed)
                .arg(&length)
                .arg(&initial_device)
                .arg(&fitted_device)
                .arg(&mut out);
            // Each thread reads one guarded prefix and writes disjoint records.
            unsafe { call.launch(grid(batch.len(), 32))? };
            let result = self.stream.clone_dtoh(&out)?;
            for (row, words) in batch.iter().zip(result.chunks_exact(ARMS * ARM_WORDS)) {
                require(words[9] == 0, "GPU rejected the trusted whole-turn scope")?;
                let old_framed = observation(row, classes, initial, &words[..ARM_WORDS])?;
                let expanded_frozen =
                    observation(row, classes, initial, &words[ARM_WORDS..2 * ARM_WORDS])?;
                let expanded_fitted = observation(row, classes, fitted, &words[2 * ARM_WORDS..])?;
                let [start, end] = row.user_bounds;
                let whole_query = Some([start.max(end.saturating_sub(32)), end]);
                require(
                    expanded_frozen.fact_bounds == Some(row.user_bounds)
                        && expanded_fitted.fact_bounds == Some(row.user_bounds)
                        && expanded_frozen.query_bounds == whole_query
                        && expanded_fitted.query_bounds == whole_query
                        && expanded_frozen.candidates.len() == expanded_fitted.candidates.len()
                        && expanded_frozen
                            .candidates
                            .iter()
                            .zip(&expanded_fitted.candidates)
                            .all(|(a, b)| {
                                a.pos == b.pos && a.id == b.id && a.features == b.features
                            }),
                    &format!(
                        "different eligible address or feature in expanded arms for {}",
                        row.case_id
                    ),
                )?;
                decisions.push(Decision {
                    case_id: row.case_id.clone(),
                    old_framed,
                    expanded_frozen,
                    expanded_fitted,
                });
            }
        }
        Ok(decisions)
    }
}

fn observation(row: &Row, classes: &[u8], weights: &[i16], words: &[u32]) -> Result<Observation> {
    // Span endpoints are allowed to equal the source-current window length.
    let bounds = |lo: u32, hi: u32| -> Result<Option<[u16; 2]>> {
        if lo == NONE && hi == NONE {
            return Ok(None);
        }
        require(
            lo < hi && hi <= row.prefix.len() as u32,
            "GPU span is out of bounds",
        )?;
        Ok(Some([lo as u16, hi as u16]))
    };
    let fact_bounds = bounds(words[0], words[1])?;
    let query_bounds = bounds(words[2], words[3])?;
    let winner_pos = optional_position(words[4])?;
    let best_ties = u16::try_from(words[5])?;
    let no_read_score = i16::try_from(words[6] as i32)?;
    let best_score = (best_ties != 0).then_some(words[7] as i32);
    let size = words[8] as usize;
    require(
        size <= CONTEXT
            && fact_bounds.is_some() == query_bounds.is_some()
            && no_read_score == weights[STATE_WORDS - 1],
        "GPU candidate header is invalid",
    )?;
    let eligible_positions = fact_bounds
        .into_iter()
        .flat_map(|[lo, hi]| lo..hi)
        .filter(|&position| eligible(classes[row.prefix[usize::from(position)] as usize]))
        .collect::<Vec<_>>();
    require(
        eligible_positions.len() == size,
        "GPU did not emit exactly the eligible span",
    )?;
    let mut candidates = Vec::with_capacity(size);
    for (item, pos) in words[HEADER..HEADER + size * CANDIDATE_WORDS]
        .chunks_exact(CANDIDATE_WORDS)
        .zip(eligible_positions)
    {
        require(
            item[0] == u32::from(pos),
            "GPU occurrence addresses are out of order",
        )?;
        let id = row.prefix[usize::from(pos)] as u16;
        let mut features = [0; FEATURES];
        for (slot, (&word, field)) in item[2..].iter().zip(&mut features).enumerate() {
            require(
                word / 32 == slot as u32,
                "GPU feature is outside its typed slot",
            )?;
            *field = word as u16;
        }
        let score = item[1] as i32;
        require(
            score
                == features
                    .iter()
                    .map(|&field| i32::from(weights[usize::from(field)]))
                    .sum::<i32>(),
            "GPU score and active i16 words disagree",
        )?;
        candidates.push(Candidate {
            pos,
            id,
            score,
            features,
        });
    }
    let observed_best = candidates.iter().map(|c| c.score).max();
    let mut best_positions = candidates.iter().filter(|c| Some(c.score) == observed_best);
    let first = best_positions.next();
    let count = usize::from(first.is_some()) + best_positions.count();
    let observed_winner = first
        .filter(|candidate| count == 1 && candidate.score > i32::from(no_read_score))
        .map(|candidate| candidate.pos);
    require(
        best_score == observed_best
            && usize::from(best_ties) == count
            && winner_pos == observed_winner,
        "GPU winner did not respect exact score ties and NoRead",
    )?;
    Ok(Observation {
        fact_bounds,
        query_bounds,
        winner_pos,
        best_score,
        best_ties,
        no_read_score,
        candidates,
    })
}
