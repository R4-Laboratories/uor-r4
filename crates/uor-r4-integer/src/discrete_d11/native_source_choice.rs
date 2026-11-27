//! Opt-in native EOS or authenticated original-source decisions after the first period.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    native_copy::{self, WriterEvent, EOS},
    native_occurrence, native_word_ending, HashingReader, IntegerDiscreteModel, MAGIC,
};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.lut-discrete-native-source-choice/1";
pub const DECODING: &str = "uor-r4.native-source-choice/1";
pub const BASE_SHA256: &str = "93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a";
const BASE_BYTES: u64 = 1_202_152_204;
const MAX_HEADER_BYTES: usize = 1024;

const PERIOD: u32 = 16;
const FEATURE_WORDS: usize = 4096;
const STAGE_WORDS: usize = FEATURE_WORDS + 1;
const STAGES: usize = 3;
const WEIGHTS: usize = STAGE_WORDS * STAGES;
pub const WEIGHT_BYTES: usize = WEIGHTS * 2;
const POSITION_STATE_WORDS: usize = WEIGHTS + FEATURE_WORDS;
const POSITION_STATE_BYTES: usize = POSITION_STATE_WORDS * 2;
const LINK_STATE_WORDS: usize = POSITION_STATE_WORDS + FEATURE_WORDS;
const LINK_STATE_BYTES: usize = LINK_STATE_WORDS * 2;
const CAP: usize = 5;

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
    weights: Box<[i16]>,
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
    model.source_choice.as_ref().map(|state| Identity {
        base_sha256: BASE_SHA256,
        state_sha256: &state.state_sha256,
        fit_sha256: &state.fit_sha256,
        state_bytes: state.weights.len() * 2,
    })
}

impl State {
    fn read(reader: &mut dyn Read, envelope: &Envelope) -> Result<Self> {
        let mut bytes = vec![0; envelope.state_bytes];
        reader.read_exact(&mut bytes)?;
        if hex::encode(Sha256::digest(&bytes)) != envelope.state_sha256 {
            return Err(invalid("native source choice state digest mismatch"));
        }
        let weights = bytes
            .chunks_exact(2)
            .map(|word| i16::from_le_bytes([word[0], word[1]]))
            .collect();
        Ok(Self {
            weights,
            state_sha256: envelope.state_sha256.clone(),
            fit_sha256: envelope.fit_sha256.clone(),
        })
    }
}

pub(super) fn read_from(
    reader: &mut dyn Read,
    header: &[u8],
    expected_body: Option<u64>,
) -> Result<IntegerDiscreteModel> {
    if header.len() > MAX_HEADER_BYTES {
        return Err(invalid("native source choice header exceeds size bound"));
    }
    let envelope: Envelope = serde_json::from_slice(header)?;
    if envelope.schema != SCHEMA
        || envelope.base_bytes != BASE_BYTES
        || envelope.base_sha256 != BASE_SHA256
        || !valid_state_bytes(envelope.state_bytes)
        || expected_body.is_some_and(|len| len != BASE_BYTES + envelope.state_bytes as u64)
        || !native_word_ending::canonical_digest(&envelope.state_sha256)
        || !native_word_ending::canonical_digest(&envelope.fit_sha256)
    {
        return Err(invalid("invalid native source choice artifact bindings"));
    }
    let state = State::read(reader, &envelope)?;
    let mut base = HashingReader::new((&mut *reader).take(BASE_BYTES));
    let mut base_header = [0; 355];
    base.read_exact(&mut base_header)?;
    if &base_header[..8] != MAGIC || base_header[8..16] != 339u64.to_le_bytes() {
        return Err(invalid("source choice requires the pinned #4319 envelope"));
    }
    let mut model = native_word_ending::read_from(
        &mut base,
        &base_header[16..],
        Some(BASE_BYTES - base_header.len() as u64),
    )?;
    if base.inner.limit() != 0 || base.finalize() != BASE_SHA256 || reader.read(&mut [0])? != 0 {
        return Err(invalid(
            "native source choice base digest or length mismatch",
        ));
    }
    model.source_choice = Some(state);
    Ok(model)
}

fn valid_state_bytes(len: usize) -> bool {
    matches!(len, WEIGHT_BYTES | POSITION_STATE_BYTES | LINK_STATE_BYTES)
}

/// Write the three signed little-endian action rows, optional first-release
/// position and link rows (in that order), and byte-identical #4319 artifact.
/// Publish the destination only after writing and verification succeed.
pub fn write_artifact(
    base: impl Read,
    mut weights: impl Read,
    fit_sha256: [u8; 32],
    mut output: impl Write,
) -> Result<()> {
    let mut bytes = Vec::with_capacity(LINK_STATE_BYTES);
    weights
        .by_ref()
        .take((LINK_STATE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if !valid_state_bytes(bytes.len()) {
        return Err(invalid(
            "native source choice weights need the original action rows and zero, one or two complete first-release rows",
        ));
    }
    let envelope = Envelope {
        schema: SCHEMA.to_owned(),
        base_bytes: BASE_BYTES,
        base_sha256: BASE_SHA256.to_owned(),
        state_bytes: bytes.len(),
        state_sha256: hex::encode(Sha256::digest(&bytes)),
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
            "native source choice frozen base digest or length mismatch",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
    Copy,
    Eos,
    Native,
}

impl Choice {
    fn from_score(score: i32) -> Self {
        match score.cmp(&0) {
            std::cmp::Ordering::Greater => Self::Copy,
            std::cmp::Ordering::Less => Self::Eos,
            std::cmp::Ordering::Equal => Self::Native,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Source {
    Copy,
    SuppliedPeriod,
    Native,
    VerifiedCopy,
    SelectedEos,
}

#[derive(Serialize)]
pub struct Event {
    #[serde(flatten)]
    pub writer: WriterEvent,
    selected_source: Source,
    stage: Option<usize>,
    action_score: Option<i32>,
    requested_pos: Option<u32>,
    requested_id: Option<u32>,
    refusal: u8,
}

impl Event {
    fn new(writer: WriterEvent, selected_source: Source) -> Self {
        Self {
            writer,
            selected_source,
            stage: None,
            action_score: None,
            requested_pos: None,
            requested_id: None,
            refusal: 0,
        }
    }
}

#[derive(Serialize)]
pub struct Trajectory {
    pub steps: Vec<Event>,
    pub terminal: bool,
    first_mismatch: Option<usize>,
}

#[derive(Clone, Copy)]
enum Control<'a> {
    Native,
    Learned(&'a [u32]),
    Oracle(usize),
}

fn score(state: &State, model: &IntegerDiscreteModel, prefix: &[u32], stage: usize) -> i32 {
    let words = &state.weights[stage * STAGE_WORDS..][..STAGE_WORDS];
    let mut score = native_word_ending::feature_sum(
        prefix,
        &model.codebook_roots,
        &words[..FEATURE_WORDS],
        i32::from(words[FEATURE_WORDS]),
    );
    if stage == 0 {
        if let Some(position) = state.weights.get(WEIGHTS..POSITION_STATE_WORDS) {
            score += native_word_ending::scope_sum(prefix, &model.codebook_roots, position);
        }
        if let Some(link) = state.weights.get(POSITION_STATE_WORDS..LINK_STATE_WORDS) {
            score += native_word_ending::link_sum(prefix, &model.codebook_roots, link);
        }
    }
    score
}

fn verified_source(model: &IntegerDiscreteModel, prefix: &[u32], position: u32) -> Option<u32> {
    let position = u16::try_from(position).ok()?;
    let candidate = model.occurrence.as_ref()?.choose(
        prefix,
        &model.codebook_roots,
        native_occurrence::OpeningChoice::Supplied(position),
    )?;
    let id = u32::from(candidate.token);
    (prefix.get(usize::from(position)) == Some(&id) && ![0, PERIOD, EOS].contains(&id))
        .then_some(id)
}

fn run(
    state: &State,
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    positions: [u32; 4],
    control: Control<'_>,
) -> Result<Trajectory> {
    model.validate_prefix(prefix)?;
    if prefix.len() + CAP - 1 > model.shape().context as usize {
        return Err(invalid(
            "source-choice prompt needs room for five transitions",
        ));
    }
    let first_id = verified_source(model, prefix, positions[0])
        .ok_or_else(|| invalid("opening address is not a verified original factual source"))?;
    let position = Some(positions[0] as u16);
    let first = native_copy::evaluate(model, prefix, native_copy::supplied(position, position))?;
    if !first.copied || first.committed_id != first_id {
        return Err(invalid(
            "opening writer did not authenticate the original source",
        ));
    }
    let mut causal = Vec::with_capacity(prefix.len() + CAP);
    causal.extend_from_slice(prefix);
    causal.push(first_id);
    let mut second = native_copy::evaluate(model, &causal, native_copy::GENERATE)?;
    second.committed_id = PERIOD;
    causal.push(PERIOD);

    let mut steps = Vec::with_capacity(CAP);
    steps.push(Event::new(first, Source::Copy));
    steps.push(Event::new(second, Source::SuppliedPeriod));
    let mut stage = (!matches!(control, Control::Native)).then_some(0);
    let mut terminal = false;
    while steps.len() < CAP && !terminal {
        let writer = native_copy::evaluate(model, &causal, native_copy::GENERATE)?;
        let mut event = Event::new(writer, Source::Native);
        event.stage = stage;
        if let Some(current) = stage {
            let choice = match control {
                Control::Learned(instruction) => {
                    let score = score(state, model, instruction, current);
                    event.action_score = Some(score);
                    Choice::from_score(score)
                }
                Control::Oracle(count) if current < count => Choice::Copy,
                Control::Oracle(_) => Choice::Eos,
                Control::Native => unreachable!(),
            };
            stage = None;
            match choice {
                Choice::Copy => {
                    let requested = positions[current + 1];
                    event.requested_pos = Some(requested);
                    event.requested_id = prefix.get(requested as usize).copied();
                    let verified = verified_source(model, prefix, requested)
                        .filter(|_| !positions[..=current].contains(&requested));
                    if let Some(id) = verified {
                        event.writer.committed_id = id;
                        event.selected_source = Source::VerifiedCopy;
                        stage = (current + 1 < STAGES).then_some(current + 1);
                    } else {
                        event.refusal = 3;
                    }
                }
                Choice::Eos => {
                    event.writer.committed_id = EOS;
                    event.selected_source = Source::SelectedEos;
                }
                Choice::Native => {}
            }
        }
        terminal = event.writer.committed_id == EOS;
        causal.push(event.writer.committed_id);
        steps.push(event);
    }
    Ok(Trajectory {
        steps,
        terminal,
        first_mismatch: None,
    })
}

/// Serve the learned action from the current prompt and four original factual
/// positions, in the supplied order.
pub fn serve(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    positions: [u32; 4],
) -> Result<Trajectory> {
    let state = checked_state(model)?;
    run(state, model, prefix, positions, Control::Learned(prefix))
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub pair_id: String,
    pub side: usize,
    pub family: String,
    pub contrast: usize,
    pub prefix: Vec<u32>,
    pub positions: [u32; 4],
    pub ids: [u32; 4],
    pub expected: Vec<u32>,
}

impl Row {
    fn validate(&self, model: &IntegerDiscreteModel) -> Result<usize> {
        model.validate_prefix(&self.prefix)?;
        let count = self.expected.len().saturating_sub(3);
        if self.side > 1 || self.contrast > 1 || count != self.contrast + self.side {
            return Err(invalid(
                "invalid source-choice contrast, side or expected length",
            ));
        }
        for (i, (&pos, &id)) in self.positions.iter().zip(&self.ids).enumerate() {
            if verified_source(model, &self.prefix, pos) != Some(id)
                || self.positions[..i].contains(&pos)
            {
                return Err(invalid("invalid or repeated original factual address"));
            }
        }
        let expected = std::iter::once(self.ids[0])
            .chain(std::iter::once(PERIOD))
            .chain(self.ids[1..1 + count].iter().copied())
            .chain(std::iter::once(EOS));
        if !expected.eq(self.expected.iter().copied()) {
            return Err(invalid("invalid expected source-choice trajectory"));
        }
        Ok(count)
    }
}

#[derive(Serialize)]
struct Common {
    stage: usize,
    own_score: i32,
    peer_score: i32,
    own_choice: Choice,
    peer_choice: Choice,
}

#[derive(Serialize)]
struct Arms {
    native: Trajectory,
    learned: Trajectory,
    swapped: Trajectory,
    correct: Trajectory,
    opposite: Trajectory,
}

#[derive(Serialize)]
pub struct Replay {
    common: Common,
    arms: Arms,
}

fn checked_state(model: &IntegerDiscreteModel) -> Result<&State> {
    model
        .source_choice
        .as_ref()
        .ok_or_else(|| invalid("source choice requires its packed state"))
}

pub fn observe(model: &IntegerDiscreteModel, row: &Row, peer: &Row) -> Result<Replay> {
    let state = checked_state(model)?;
    let count = row.validate(model)?;
    let peer_count = peer.validate(model)?;
    if row.pair_id != peer.pair_id
        || row.side == peer.side
        || row.family != peer.family
        || row.contrast != peer.contrast
        || row.positions != peer.positions
        || row.ids != peer.ids
        || row.prefix == peer.prefix
        || native_word_ending::before_instruction(&row.prefix)
            != native_word_ending::before_instruction(&peer.prefix)
    {
        return Err(invalid(
            "source-choice peers must share original facts and question",
        ));
    }
    let stage = count.min(peer_count);
    let own_score = score(state, model, &row.prefix, stage);
    let peer_score = score(state, model, &peer.prefix, stage);
    let replay = |control| -> Result<Trajectory> {
        let mut trajectory = run(state, model, &row.prefix, row.positions, control)?;
        trajectory.first_mismatch = (0..trajectory.steps.len().max(row.expected.len()))
            .find(|&i| {
                trajectory.steps.get(i).map(|step| step.writer.committed_id)
                    != row.expected.get(i).copied()
            })
            .map(|i| i + 1);
        Ok(trajectory)
    };
    Ok(Replay {
        common: Common {
            stage,
            own_score,
            peer_score,
            own_choice: Choice::from_score(own_score),
            peer_choice: Choice::from_score(peer_score),
        },
        arms: Arms {
            native: replay(Control::Native)?,
            learned: replay(Control::Learned(&row.prefix))?,
            swapped: replay(Control::Learned(&peer.prefix))?,
            correct: replay(Control::Oracle(count))?,
            opposite: replay(Control::Oracle(peer_count))?,
        },
    })
}
