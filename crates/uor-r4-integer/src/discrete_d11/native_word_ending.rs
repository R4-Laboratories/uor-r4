//! One native stop-or-period decision after an authenticated original-word copy.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    native_copy::{self, Action, WriterEvent, EOS},
    native_occurrence, native_opening, pack_4block_roots, HashingReader, IntegerDiscreteModel,
    MAGIC,
};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.lut-discrete-native-word-ending/1";
pub const DECODING: &str = "uor-r4.native-word-ending/1";
pub const BASE_SHA256: &str = "591ea4a21e39942ac86625a8903ff90e2e6055b382966e740787211edcfe73c8";
const BASE_BYTES: u64 = 1_202_143_657;
const BASE_HEADER_BYTES: usize = 358;
const BASE_HEADER_SHA256: &str = "fd99baa3ba5a1977490334f8bcec6452570476a2d4b9e026e91bc8e5620225f5";
const MAX_HEADER_BYTES: usize = 1024;

const PERIOD: u32 = 16;
const QUESTION: u32 = 33;
const LF: u32 = 201;
const HALF: usize = 2048;
const WEIGHTS: usize = 2 * HALF;
const LINK_WINDOW: usize = 16;
pub const WEIGHT_BYTES: usize = WEIGHTS * 2;
pub const REPLAY_STEPS: usize = 3;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    base_bytes: u64,
    base_sha256: String,
    state_bytes: usize,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Clone, Debug)]
pub(super) struct State {
    weights: Box<[i16; WEIGHTS]>,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct Identity<'a> {
    pub base_sha256: &'static str,
    pub state_sha256: &'a str,
    pub fit_sha256: &'a str,
    pub state_bytes: usize,
}

pub fn identity(model: &IntegerDiscreteModel) -> Option<Identity<'_>> {
    model.word_ending.as_ref().map(|state| Identity {
        base_sha256: BASE_SHA256,
        state_sha256: &state.state_sha256,
        fit_sha256: &state.fit_sha256,
        state_bytes: WEIGHT_BYTES,
    })
}

pub(super) fn canonical_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl State {
    fn read(reader: &mut dyn Read, envelope: &Envelope) -> Result<Self> {
        let mut bytes = [0; WEIGHT_BYTES];
        reader.read_exact(&mut bytes)?;
        if hex::encode(Sha256::digest(bytes)) != envelope.state_sha256 {
            return Err(invalid("native word ending state digest mismatch"));
        }
        let mut weights = Box::new([0; WEIGHTS]);
        for (weight, bytes) in weights.iter_mut().zip(bytes.chunks_exact(2)) {
            *weight = i16::from_le_bytes([bytes[0], bytes[1]]);
        }
        Ok(Self {
            weights,
            state_sha256: envelope.state_sha256.clone(),
            fit_sha256: envelope.fit_sha256.clone(),
        })
    }

    fn score(&self, prefix: &[u32], roots: &[[u8; 4]]) -> i32 {
        feature_sum(prefix, roots, &self.weights[..], 0)
    }
}

pub(super) fn feature_sum(prefix: &[u32], roots: &[[u8; 4]], weights: &[i16], score: i32) -> i32 {
    weighted_instruction(prefix, weights, score, |instruction, tally| {
        adjacent_features(instruction, roots, false, tally);
    })
}

pub(super) fn scope_sum(prefix: &[u32], roots: &[[u8; 4]], weights: &[i16]) -> i32 {
    weighted_instruction(prefix, weights, 0, |instruction, tally| {
        adjacent_features(instruction, roots, true, tally);
    })
}

pub(super) fn link_sum(prefix: &[u32], roots: &[[u8; 4]], weights: &[i16]) -> i32 {
    addressed_link_sum(prefix, roots, weights, 0)
}

pub(super) fn addressed_link_sum(
    prefix: &[u32],
    roots: &[[u8; 4]],
    weights: &[i16],
    address_xor: usize,
) -> i32 {
    weighted_instruction(prefix, weights, 0, |instruction, tally| {
        let mut previous = [0; LINK_WINDOW];
        for (index, &id) in instruction.iter().enumerate() {
            let later = pack_4block_roots(roots[id as usize]);
            for distance in 2..=index.min(LINK_WINDOW) {
                let earlier = previous[(index - distance) % LINK_WINDOW];
                let phase = position_phase(distance);
                tally.include(bucket(ordered_roots(earlier, later) ^ phase) ^ address_xor);
                tally.include((HALF + bucket(ordered_roots(later, earlier) ^ phase)) ^ address_xor);
            }
            previous[index % LINK_WINDOW] = later;
        }
    })
}

struct InstructionScore<'a> {
    weights: &'a [i16],
    seen: [u64; WEIGHTS / u64::BITS as usize],
    score: i32,
}

impl InstructionScore<'_> {
    fn include(&mut self, bucket: usize) {
        let cell = &mut self.seen[bucket >> 6];
        let bit = 1u64 << (bucket & 63);
        if (*cell & bit) == 0 {
            *cell |= bit;
            self.score += i32::from(self.weights[bucket]);
        }
    }
}

fn weighted_instruction(
    prefix: &[u32],
    weights: &[i16],
    score: i32,
    visit: impl FnOnce(&[u32], &mut InstructionScore<'_>),
) -> i32 {
    let Some(instruction) = instruction_tokens(prefix) else {
        return 0;
    };
    let mut tally = InstructionScore {
        weights,
        seen: [0; WEIGHTS / u64::BITS as usize],
        score,
    };
    visit(instruction, &mut tally);
    tally.score
}

fn adjacent_features(
    instruction: &[u32],
    roots: &[[u8; 4]],
    bind_position: bool,
    tally: &mut InstructionScore<'_>,
) {
    let mut previous: Option<u16> = None;
    for (index, &id) in instruction.iter().enumerate() {
        let root = pack_4block_roots(roots[id as usize]);
        let position = if bind_position {
            // Bind each root to its one-based distance from the right boundary.
            position_phase(instruction.len() - index)
        } else {
            0
        };
        tally.include(bucket(root ^ position));
        if let Some(previous) = previous {
            tally.include(HALF + bucket(ordered_roots(previous, root) ^ position));
        }
        previous = Some(root);
    }
}

fn position_phase(distance: usize) -> u16 {
    (distance as u16).wrapping_mul(0x9e37)
}

fn ordered_roots(earlier: u16, later: u16) -> u16 {
    earlier.rotate_left(5) ^ later
}

fn instruction_tokens(prefix: &[u32]) -> Option<&[u32]> {
    let before = before_instruction(prefix)?;
    Some(&prefix[before.len()..prefix.len() - native_opening::ASSISTANT_MARKER.len()])
}

pub(super) fn before_instruction(prefix: &[u32]) -> Option<&[u32]> {
    let body = prefix.strip_suffix(&native_opening::ASSISTANT_MARKER)?;
    let separator = body.windows(2).rposition(|pair| pair == [LF, LF])?;
    let question = &body[separator + 2..];
    let boundary = question.iter().rposition(|&id| id == QUESTION)?;
    Some(&body[..separator + 2 + boundary + 1])
}

fn bucket(code: u16) -> usize {
    usize::from(code ^ (code >> 7)) & (HALF - 1)
}

pub fn score(model: &IntegerDiscreteModel, prefix: &[u32]) -> Result<i32> {
    model.validate_prefix(prefix)?;
    let state = model.word_ending.as_ref().ok_or_else(missing_state)?;
    Ok(state.score(prefix, &model.codebook_roots))
}

fn missing_state() -> crate::IntegerError {
    invalid("native word ending requires a packed word-ending artifact")
}

pub(super) fn read_from(
    reader: &mut dyn Read,
    header: &[u8],
    expected_body: Option<u64>,
) -> Result<IntegerDiscreteModel> {
    if header.len() > MAX_HEADER_BYTES {
        return Err(invalid("native word ending header exceeds size bound"));
    }
    let envelope: Envelope = serde_json::from_slice(header)?;
    if envelope.schema != SCHEMA
        || envelope.base_bytes != BASE_BYTES
        || envelope.base_sha256 != BASE_SHA256
        || envelope.state_bytes != WEIGHT_BYTES
        || expected_body.is_some_and(|len| len != BASE_BYTES + WEIGHT_BYTES as u64)
        || !canonical_digest(&envelope.state_sha256)
        || !canonical_digest(&envelope.fit_sha256)
    {
        return Err(invalid("invalid native word ending artifact bindings"));
    }
    let state = State::read(reader, &envelope)?;
    let mut base = HashingReader::new((&mut *reader).take(BASE_BYTES));
    let mut base_header = [0; BASE_HEADER_BYTES];
    base.read_exact(&mut base_header)?;
    if hex::encode(Sha256::digest(base_header)) != BASE_HEADER_SHA256 {
        return Err(invalid(
            "native word ending requires the pinned #4315 header",
        ));
    }
    let mut model = native_occurrence::read_from(
        &mut base,
        &base_header[16..],
        Some(BASE_BYTES - BASE_HEADER_BYTES as u64),
    )?;
    if base.inner.limit() != 0 || base.finalize() != BASE_SHA256 || reader.read(&mut [0])? != 0 {
        return Err(invalid("native word ending base digest or length mismatch"));
    }
    model.word_ending = Some(state);
    Ok(model)
}

/// Write the 4,096 signed little-endian words and byte-identical #4315 artifact.
/// Publish the destination only after writing and verification succeed.
pub fn write_artifact(
    base: impl Read,
    mut weights: impl Read,
    fit_sha256: [u8; 32],
    mut output: impl Write,
) -> Result<()> {
    let mut bytes = [0; WEIGHT_BYTES];
    weights.read_exact(&mut bytes)?;
    if weights.read(&mut [0])? != 0 {
        return Err(invalid(
            "native word ending weight input has trailing bytes",
        ));
    }
    let envelope = Envelope {
        schema: SCHEMA.to_owned(),
        base_bytes: BASE_BYTES,
        base_sha256: BASE_SHA256.to_owned(),
        state_bytes: WEIGHT_BYTES,
        state_sha256: hex::encode(Sha256::digest(bytes)),
        fit_sha256: hex::encode(fit_sha256),
    };
    let header = serde_json::to_vec(&envelope)?;
    output.write_all(MAGIC)?;
    output.write_all(&(header.len() as u64).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(&bytes)?;
    let mut base = HashingReader::new(base);
    if std::io::copy(&mut (&mut base).take(BASE_BYTES), &mut output)? != BASE_BYTES
        || base.read(&mut [0])? != 0
        || base.finalize() != BASE_SHA256
    {
        return Err(invalid(
            "native word ending frozen base digest or length mismatch",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectedSource {
    Copy,
    Native,
    Learned,
    Swapped,
    Oracle,
    WrongOracle,
}

#[derive(Clone, Copy)]
struct Ending {
    token: Option<u32>,
    source: SelectedSource,
}

impl Ending {
    fn from_score(score: i32, source: SelectedSource) -> Self {
        let token = match score.cmp(&0) {
            std::cmp::Ordering::Less => Some(EOS),
            std::cmp::Ordering::Equal => None,
            std::cmp::Ordering::Greater => Some(PERIOD),
        };
        Self { token, source }
    }
}

enum Phase {
    Opening(Action),
    AfterCopy,
    Ordinary,
    Terminal,
}

#[derive(Debug, Serialize)]
pub struct Event {
    #[serde(flatten)]
    pub writer: WriterEvent,
    pub selected_source: SelectedSource,
}

/// An opted-in ordinary answer. An exact source, when supplied, intervenes
/// on the first step only. The caller appends each actual commit before stepping.
pub struct Session<'a> {
    model: &'a IntegerDiscreteModel,
    score: i32,
    ending: Ending,
    phase: Phase,
}

impl<'a> Session<'a> {
    pub fn new(
        model: &'a IntegerDiscreteModel,
        prompt: &[u32],
        supplied_source: Option<u16>,
    ) -> Result<Self> {
        let score = score(model, prompt)?;
        let opening = supplied_source.map_or(native_copy::LEARNED, |pos| {
            native_copy::supplied(Some(pos), Some(pos))
        });
        Ok(Self::with_ending(
            model,
            score,
            opening,
            Ending::from_score(score, SelectedSource::Learned),
        ))
    }

    fn with_ending(
        model: &'a IntegerDiscreteModel,
        score: i32,
        opening: Action,
        ending: Ending,
    ) -> Self {
        Self {
            model,
            score,
            ending,
            phase: Phase::Opening(opening),
        }
    }

    pub fn score(&self) -> i32 {
        self.score
    }

    pub fn terminal(&self) -> bool {
        matches!(self.phase, Phase::Terminal)
    }

    pub fn step(&mut self, prefix: &[u32]) -> Result<Event> {
        let action = match self.phase {
            Phase::Opening(action) => action,
            Phase::AfterCopy | Phase::Ordinary => native_copy::GENERATE,
            Phase::Terminal => return Err(invalid("native word ending session has reached EOS")),
        };
        let mut writer = native_copy::evaluate(self.model, prefix, action)?;
        let selected_source = if writer.copied {
            SelectedSource::Copy
        } else if matches!(self.phase, Phase::AfterCopy) {
            if let Some(token) = self.ending.token {
                writer.committed_id = token;
                self.ending.source
            } else {
                SelectedSource::Native
            }
        } else {
            SelectedSource::Native
        };
        self.phase = if writer.committed_id == EOS {
            Phase::Terminal
        } else if writer.copied {
            Phase::AfterCopy
        } else {
            Phase::Ordinary
        };
        Ok(Event {
            writer,
            selected_source,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct Trajectory {
    pub steps: Vec<Event>,
    pub terminal: bool,
    pub first_mismatch: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct Arms {
    pub native: Trajectory,
    pub learned: Trajectory,
    pub swapped: Trajectory,
    pub oracle: Trajectory,
    pub wrong_oracle: Trajectory,
}

#[derive(Debug, Serialize)]
pub struct Replay {
    pub score: i32,
    pub peer_score: i32,
    pub arms: Arms,
}

/// Replay all five ending choices with the same verified-copy action and
/// ordinary writer. Only the swapped arm reads the matched peer's real score.
pub fn observe(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    peer_prefix: &[u32],
    source: u16,
    expected: &[u32],
) -> Result<Replay> {
    let own_score = score(model, prefix)?;
    let peer_score = score(model, peer_prefix)?;
    let (right, wrong) = match expected {
        [_, EOS] => (EOS, PERIOD),
        [_, PERIOD, EOS] => (PERIOD, EOS),
        _ => return Err(invalid("expected a word followed by EOS or period and EOS")),
    };
    let opening = native_copy::supplied(Some(source), Some(source));
    let run = |ending| -> Result<Trajectory> {
        let mut session = Session::with_ending(model, own_score, opening, ending);
        let mut causal = prefix.to_vec();
        let mut steps = Vec::with_capacity(REPLAY_STEPS);
        for _ in 0..REPLAY_STEPS {
            let event = session.step(&causal)?;
            let id = event.writer.committed_id;
            steps.push(event);
            if session.terminal() {
                break;
            }
            if causal.len() == model.shape().context as usize {
                causal.remove(0);
            }
            causal.push(id);
        }
        let first_mismatch = (0..steps.len().max(expected.len()))
            .find(|&idx| {
                steps.get(idx).map(|step| step.writer.committed_id) != expected.get(idx).copied()
            })
            .map(|idx| idx + 1);
        Ok(Trajectory {
            steps,
            terminal: session.terminal(),
            first_mismatch,
        })
    };
    Ok(Replay {
        score: own_score,
        peer_score,
        arms: Arms {
            native: run(Ending::from_score(0, SelectedSource::Native))?,
            learned: run(Ending::from_score(own_score, SelectedSource::Learned))?,
            swapped: run(Ending::from_score(peer_score, SelectedSource::Swapped))?,
            oracle: run(Ending {
                token: Some(right),
                source: SelectedSource::Oracle,
            })?,
            wrong_oracle: run(Ending {
                token: Some(wrong),
                source: SelectedSource::WrongOracle,
            })?,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(body: &[u32]) -> Vec<u32> {
        body.iter()
            .copied()
            .chain(native_opening::ASSISTANT_MARKER)
            .collect()
    }

    #[test]
    fn instruction_uses_the_last_question_boundary_in_the_current_section() {
        let prompt = prefix(&[QUESTION, LF, LF, QUESTION, 2, QUESTION, 5, 6]);
        assert_eq!(instruction_tokens(&prompt), Some(&[5, 6][..]));
        assert_eq!(instruction_tokens(&prefix(&[QUESTION, LF, LF, 5])), None);
        assert_eq!(instruction_tokens(&prefix(&[QUESTION, 5])), None);
        assert_eq!(instruction_tokens(&prompt[..prompt.len() - 1]), None);
    }

    #[test]
    fn signed_state_round_trip_counts_each_root_bucket_once_with_ordered_pairs() {
        let mut bytes = [0; WEIGHT_BYTES];
        for (index, value) in [(1, -23i16), (2, 42), (HALF + 34, 300), (HALF + 65, -20)] {
            bytes[2 * index..2 * index + 2].copy_from_slice(&value.to_le_bytes());
        }
        let mut envelope = Envelope {
            schema: SCHEMA.to_owned(),
            base_bytes: BASE_BYTES,
            base_sha256: BASE_SHA256.to_owned(),
            state_bytes: WEIGHT_BYTES,
            state_sha256: hex::encode(Sha256::digest(bytes)),
            fit_sha256: "11".repeat(32),
        };
        let state = State::read(&mut bytes.as_slice(), &envelope).unwrap();
        let roots = [[0; 4], [1, 0, 0, 0], [2, 0, 0, 0]];
        assert_eq!(
            state.score(&prefix(&[LF, LF, QUESTION, 1, 2, 1, 2]), &roots),
            299
        );
        assert_eq!(state.score(&prefix(&[LF, LF, QUESTION, 2, 1]), &roots), -1);
        assert_eq!(state.score(&prefix(&[LF, LF, QUESTION]), &roots), 0);
        envelope.state_sha256 = "00".repeat(32);
        assert!(State::read(&mut bytes.as_slice(), &envelope).is_err());
    }

    #[test]
    fn scope_position_separates_a_root_and_adjacency_set_collision() {
        let roots = [[0; 4], [1, 0, 0, 0], [2, 0, 0, 0]];
        let left = prefix(&[LF, LF, QUESTION, 1, 1, 2, 1]);
        let right = prefix(&[LF, LF, QUESTION, 1, 2, 1, 1]);
        let base = [1; WEIGHTS];
        assert_eq!(feature_sum(&left, &roots, &base, 0), 5);
        assert_eq!(feature_sum(&right, &roots, &base, 0), 5);

        let mut residual = [0; WEIGHTS];
        residual[785] = -17;
        residual[786] = 23;
        residual[3124] = -11;
        residual[3159] = 41;
        assert_eq!(scope_sum(&left, &roots, &residual), -28);
        assert_eq!(scope_sum(&right, &roots, &residual), 64);
        assert_eq!(scope_sum(&[1, 1, 2, 1], &roots, &residual), 0);
    }

    #[test]
    fn scope_links_read_a_signed_nonlocal_selector_argument_interaction() {
        let roots = std::array::from_fn::<_, 7, _>(|id| [id as u8, 0, 0, 0]);
        let mut residual = [0; WEIGHTS];
        // Each planted pair spans three positions. Both oriented buckets contribute.
        for (index, weight) in [
            (819, -7),
            (2929, -4),
            (820, 13),
            (2960, 4),
            (851, 17),
            (2930, 5),
            (852, -19),
            (2963, -4),
        ] {
            residual[index] = weight;
        }
        for (selector, argument, expected) in [(1, 3, -11), (1, 4, 17), (2, 3, 22), (2, 4, -23)] {
            let prompt = prefix(&[LF, LF, QUESTION, selector, 5, 6, argument]);
            assert_eq!(link_sum(&prompt, &roots, &residual), expected);
        }
        assert_eq!(link_sum(&[1, 5, 6, 3], &roots, &residual), 0);
    }

    #[test]
    fn scope_links_count_each_oriented_bucket_once_within_the_distance_bounds() {
        let roots = [[1, 0, 0, 0]];
        let mut body = vec![LF, LF, QUESTION];
        body.extend([0; 20]);
        // Distances 2 through 16 give 15 unique keys in each half for this root.
        assert_eq!(link_sum(&prefix(&body), &roots, &[1; WEIGHTS]), 30);
    }

    #[test]
    fn release_residual_swaps_the_oriented_bucket_halves_after_b() {
        let roots = std::array::from_fn::<_, 7, _>(|id| [id as u8, 0, 0, 0]);
        let prompt = prefix(&[LF, LF, QUESTION, 1, 5, 6, 3]);
        let mut residual = [0; WEIGHTS];
        residual[819] = -7;
        residual[2929] = -4;
        assert_eq!(addressed_link_sum(&prompt, &roots, &residual, 0), -11);
        assert_eq!(addressed_link_sum(&prompt, &roots, &residual, HALF), 0);
        residual[819 ^ HALF] = -17;
        residual[2929 ^ HALF] = 23;
        assert_eq!(addressed_link_sum(&prompt, &roots, &residual, 0), -11);
        assert_eq!(addressed_link_sum(&prompt, &roots, &residual, HALF), 6);
    }
}
