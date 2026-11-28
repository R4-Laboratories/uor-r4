//! Shared integer source score over original, causal factual occurrences.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::dialogue::EncodedUserOpening;

use super::{native_opening, pack_4block_roots, HashingReader, IntegerDiscreteModel, MAGIC};
use crate::{invalid, Result};

mod features;
pub mod lexical;
pub mod lexical_commit;
use features::Prompt;
pub use features::{classify_token, Candidate, ACTIVE, WEIGHTS};

pub const SCHEMA: &str = "uor-r4.lut-discrete-native-occurrence/1";
pub const SPEC_SHA256: &str = "dfa61825eeae77042d8f6e541e801f55e5e65e1f53946ca9f4dd72813f6e6acb";
pub const CLASSES_SHA256: &str = "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23";
pub const MARKER_SHA256: &str = "d782af466c2ebdfba1304158a3f995b50937c434de6ee5a505f8c49cc944f143";
const MARKER_HEADER_BYTES: usize = 369;
const MARKER_HEADER_SHA256: &str =
    "f85b53c3ff955147c46710dc26198808a26719e1c103cc3f13849c0b6df325e4";
const MARKER_BYTES: u64 = 1_202_138_137;
const STATE_MAGIC: &[u8; 8] = b"UOROCC01";
const STATE_PREFIX_BYTES: usize = STATE_MAGIC.len() + 32;
pub const WEIGHT_BYTES: usize = (WEIGHTS + 1) * 2;
const STATE_BYTES: usize = STATE_PREFIX_BYTES + WEIGHT_BYTES + VOCABULARY;
const CONTEXT: usize = native_opening::CONTEXT as usize;
pub const VOCABULARY: usize = 4096;
const WEIGHT_LIMIT: i16 = 4096;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    marker_bytes: u64,
    marker_sha256: String,
    state_bytes: usize,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Clone, Debug)]
pub(super) struct State {
    weights: WeightWords,
    classes: Box<[u8; VOCABULARY]>,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Clone, Debug)]
pub(super) struct WeightWords([i16; WEIGHTS + 1]);

impl WeightWords {
    pub(super) fn parse(bytes: &[u8; WEIGHT_BYTES]) -> Result<Self> {
        let mut words = [0; WEIGHTS + 1];
        for (word, bytes) in words.iter_mut().zip(bytes.chunks_exact(2)) {
            *word = i16::from_le_bytes([bytes[0], bytes[1]]);
            if !(-WEIGHT_LIMIT..=WEIGHT_LIMIT).contains(word) {
                return Err(invalid("native occurrence weight exceeds the signed bound"));
            }
        }
        Ok(Self(words))
    }

    pub(super) fn bytes(&self) -> [u8; WEIGHT_BYTES] {
        let mut bytes = [0; WEIGHT_BYTES];
        for (word, output) in self.0.iter().zip(bytes.chunks_exact_mut(2)) {
            output.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn score(&self, candidate: Candidate) -> i32 {
        candidate
            .features
            .iter()
            .map(|&id| i32::from(self.0[usize::from(id)]))
            .sum()
    }

    fn no_read(&self) -> i16 {
        self.0[WEIGHTS]
    }

    fn best(&self, prompt: &Prompt<'_>) -> Best {
        let mut best = Best::default();
        for candidate in prompt.candidates() {
            best.consider(candidate, self.score(candidate));
        }
        best
    }
}

#[derive(Debug, Serialize)]
pub struct Identity<'a> {
    pub marker_sha256: &'static str,
    pub feature_spec_sha256: &'static str,
    pub token_classes_sha256: &'static str,
    pub state_sha256: &'a str,
    pub fit_sha256: &'a str,
    pub state_bytes: usize,
}

#[derive(Clone, Copy)]
pub(super) enum OpeningChoice {
    Learned,
    Supplied(u16),
    Fallback,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpeningDecision {
    Selected,
    FramingRejected,
    NoEligibleCandidate,
    ScoreTie,
    NoReadThreshold,
    Disabled,
}

#[derive(Debug, Serialize)]
pub struct OpeningProbe {
    pub parsed_start: Option<u16>,
    pub parsed_end: Option<u16>,
    pub decision: OpeningDecision,
    pub best_score: Option<i32>,
    pub best_ties: u16,
    pub no_read_score: i16,
}

#[derive(Debug, Serialize)]
pub struct Assessment {
    pub parsed_start: Option<u16>,
    pub parsed_end: Option<u16>,
    pub candidates: Vec<AssessmentCandidate>,
    pub winner_pos: Option<u16>,
    pub best_score: Option<i32>,
    pub best_ties: u16,
    pub no_read_score: i16,
    pub gold_rank: Option<u16>,
    pub rival_rank: Option<u16>,
}

#[derive(Debug, Serialize)]
pub struct AssessmentCandidate {
    pub pos: u16,
    pub id: u16,
    pub eligible: bool,
    pub geometry: u16,
    pub features: Option<[u16; ACTIVE]>,
    pub score: Option<i32>,
    pub rank: Option<u16>,
}

#[derive(Debug, Serialize)]
pub struct OpeningAssessment {
    pub fact_bounds: Option<[u16; 2]>,
    pub query_bounds: Option<[u16; 2]>,
    pub winner_pos: Option<u16>,
    pub best_score: Option<i32>,
    pub best_ties: u16,
    pub no_read_score: i16,
    pub candidates: Vec<ScoredCandidate>,
}

#[derive(Debug, Serialize)]
pub struct ScoredCandidate {
    pub pos: u16,
    pub id: u16,
    pub score: i32,
    pub features: [u16; ACTIVE],
}

#[derive(Clone, Debug, Serialize)]
pub struct WriterEvent {
    pub source_pos: Option<u16>,
    pub gate_id: u16,
    pub emitted_id: u32,
    pub first_admitted_cell: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArmReplay {
    #[serde(flatten)]
    pub first: WriterEvent,
    pub lexical: Option<WriterEvent>,
}

#[derive(Debug, Serialize)]
pub struct Arms {
    pub learned: ArmReplay,
    pub correct: ArmReplay,
    pub swapped: ArmReplay,
    pub absent: ArmReplay,
    pub marker: ArmReplay,
}

#[derive(Debug, Serialize)]
pub struct Replay {
    #[serde(flatten)]
    pub assessment: Assessment,
    pub arms: Arms,
}

pub(super) fn replay(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    raw_target: Option<u32>,
    target: u32,
    gold_pos: Option<u16>,
    rival_pos: Option<u16>,
) -> Result<Replay> {
    model.validate_prefix(prefix)?;
    let state = model
        .occurrence
        .as_ref()
        .ok_or_else(|| invalid("artifact has no shared source occurrence state"))?;
    let writer_event = |ids: &[u32], choice| -> Result<WriterEvent> {
        let source = state.choose(ids, &model.codebook_roots, choice);
        let step = model.evaluate_prefix_internal(ids, false, choice)?;
        let gate = step
            .trunk_pointer
            .ok_or_else(|| invalid("occurrence replay requires the frozen writer gate"))?;
        if source.is_some_and(|candidate| candidate.token != gate.selected_token) {
            return Err(invalid(
                "occurrence replay source differs from writer gate input",
            ));
        }
        Ok(WriterEvent {
            source_pos: source.map(|candidate| candidate.position),
            gate_id: gate.selected_token,
            emitted_id: step.evaluation.prediction,
            first_admitted_cell: step.evaluation.admitted_cells.first().copied(),
        })
    };
    let arm = |first, second| -> Result<ArmReplay> {
        let first = writer_event(prefix, first)?;
        let lexical = if first.emitted_id == native_opening::SPACE && prefix.len() < CONTEXT {
            let mut extended = prefix.to_vec();
            extended.push(native_opening::SPACE);
            Some(writer_event(&extended, second)?)
        } else {
            None
        };
        Ok(ArmReplay { first, lexical })
    };
    let supplied = |position: Option<u16>| {
        let selected = position.map_or(OpeningChoice::Fallback, OpeningChoice::Supplied);
        let fallback = OpeningChoice::Fallback;
        if raw_target == Some(native_opening::SPACE)
            && target != native_opening::SPACE
            && prefix.last() == Some(&native_opening::COLON)
        {
            arm(fallback, selected)
        } else {
            arm(selected, fallback)
        }
    };
    let fallback = arm(OpeningChoice::Fallback, OpeningChoice::Fallback)?;
    Ok(Replay {
        assessment: state.assess_prefix(prefix, &model.codebook_roots, gold_pos, rival_pos),
        arms: Arms {
            learned: arm(OpeningChoice::Learned, OpeningChoice::Learned)?,
            correct: supplied(gold_pos)?,
            swapped: supplied(rival_pos)?,
            absent: fallback.clone(),
            marker: fallback,
        },
    })
}

#[derive(Default)]
struct Best {
    candidate: Option<Candidate>,
    score: Option<i32>,
    ties: u16,
}

impl Best {
    fn consider(&mut self, candidate: Candidate, score: i32) {
        match self.score {
            Some(best) if score < best => (),
            Some(best) if score == best => self.ties += 1,
            _ => {
                self.candidate = Some(candidate);
                self.score = Some(score);
                self.ties = 1;
            }
        }
    }

    fn read(&self, no_read: i16) -> Option<Candidate> {
        self.candidate
            .filter(|_| self.ties == 1 && self.score > Some(i32::from(no_read)))
    }
}

impl State {
    fn new(envelope: Envelope, payload: &[u8; STATE_BYTES]) -> Result<Self> {
        let mut spec = [0; 32];
        hex::decode_to_slice(SPEC_SHA256, &mut spec)
            .map_err(|_| invalid("invalid native occurrence feature spec digest"))?;
        if &payload[..STATE_MAGIC.len()] != STATE_MAGIC
            || payload[STATE_MAGIC.len()..STATE_PREFIX_BYTES] != spec
        {
            return Err(invalid(
                "native occurrence state magic or feature spec mismatch",
            ));
        }
        let weights = WeightWords::parse(
            payload[STATE_PREFIX_BYTES..STATE_PREFIX_BYTES + WEIGHT_BYTES]
                .try_into()
                .map_err(|_| invalid("invalid native occurrence weight length"))?,
        )?;
        let classes = Box::new(
            payload[STATE_PREFIX_BYTES + WEIGHT_BYTES..]
                .try_into()
                .map_err(|_| invalid("invalid native occurrence token class length"))?,
        );
        if hex::encode(Sha256::digest(&*classes)) != CLASSES_SHA256 {
            return Err(invalid(
                "native occurrence canonical token classes mismatch",
            ));
        }
        Ok(Self {
            weights,
            classes,
            state_sha256: envelope.state_sha256,
            fit_sha256: envelope.fit_sha256,
        })
    }

    pub(super) fn identity(&self) -> Identity<'_> {
        Identity {
            marker_sha256: MARKER_SHA256,
            feature_spec_sha256: SPEC_SHA256,
            token_classes_sha256: CLASSES_SHA256,
            state_sha256: &self.state_sha256,
            fit_sha256: &self.fit_sha256,
            state_bytes: STATE_BYTES,
        }
    }

    pub(super) fn weight_bytes(&self) -> [u8; WEIGHT_BYTES] {
        self.weights.bytes()
    }

    fn user_prompt<'a>(
        &'a self,
        opening: &'a EncodedUserOpening,
        roots: &'a [[u8; 4]],
    ) -> Option<Prompt<'a>> {
        Prompt::for_user(opening.tokens(), &self.classes, roots, opening.user_range())
    }

    pub(super) fn choose_user(
        &self,
        opening: &EncodedUserOpening,
        roots: &[[u8; 4]],
        weights: &WeightWords,
    ) -> Result<Option<Candidate>> {
        let prompt = self
            .user_prompt(opening, roots)
            .ok_or_else(|| invalid("invalid serialized whole-turn opening"))?;
        Ok(weights.best(&prompt).read(weights.no_read()))
    }

    pub(super) fn probe_user(
        &self,
        opening: Option<&EncodedUserOpening>,
        roots: &[[u8; 4]],
        weights: &WeightWords,
    ) -> OpeningProbe {
        let prompt = opening.and_then(|opening| self.user_prompt(opening, roots));
        make_probe(prompt, weights, opening.is_some())
    }

    pub(super) fn assess_opening(
        &self,
        opening: &EncodedUserOpening,
        roots: &[[u8; 4]],
        expanded: Option<&WeightWords>,
    ) -> Result<OpeningAssessment> {
        let (prompt, weights) = match expanded {
            Some(weights) => (
                Some(
                    self.user_prompt(opening, roots)
                        .ok_or_else(|| invalid("invalid serialized whole-turn opening"))?,
                ),
                weights,
            ),
            None => (
                Prompt::new(opening.tokens(), &self.classes, roots),
                &self.weights,
            ),
        };
        let mut best = Best::default();
        let mut candidates = Vec::new();
        if let Some(prompt) = &prompt {
            for candidate in prompt.candidates() {
                let score = weights.score(candidate);
                best.consider(candidate, score);
                candidates.push(ScoredCandidate {
                    pos: candidate.position,
                    id: candidate.token,
                    score,
                    features: candidate.features,
                });
            }
        }
        Ok(OpeningAssessment {
            fact_bounds: prompt.as_ref().map(Prompt::fact_bounds),
            query_bounds: prompt.as_ref().map(Prompt::query_bounds),
            winner_pos: best
                .read(weights.no_read())
                .map(|candidate| candidate.position),
            best_score: best.score,
            best_ties: best.ties,
            no_read_score: weights.no_read(),
            candidates,
        })
    }

    pub(super) fn choose(
        &self,
        ids: &[u32],
        roots: &[[u8; 4]],
        choice: OpeningChoice,
    ) -> Option<Candidate> {
        if matches!(choice, OpeningChoice::Fallback) {
            return None;
        }
        let prompt = Prompt::new(ids, &self.classes, roots)?;
        if let OpeningChoice::Supplied(position) = choice {
            return prompt.candidate(usize::from(position));
        }
        self.weights.best(&prompt).read(self.weights.no_read())
    }

    pub(super) fn probe(&self, ids: &[u32], roots: &[[u8; 4]], enabled: bool) -> OpeningProbe {
        make_probe(
            Prompt::new(ids, &self.classes, roots),
            &self.weights,
            enabled,
        )
    }

    fn assess_prefix(
        &self,
        ids: &[u32],
        roots: &[[u8; 4]],
        gold_pos: Option<u16>,
        rival_pos: Option<u16>,
    ) -> Assessment {
        let mut assessment = Assessment {
            parsed_start: None,
            parsed_end: None,
            candidates: Vec::new(),
            winner_pos: None,
            best_score: None,
            best_ties: 0,
            no_read_score: self.weights.no_read(),
            gold_rank: None,
            rival_rank: None,
        };
        let Some(prompt) = Prompt::new(ids, &self.classes, roots) else {
            return assessment;
        };
        let [start, end] = prompt.fact_bounds();
        assessment.parsed_start = Some(start);
        assessment.parsed_end = Some(end);
        let mut best = Best::default();
        for pos in start..end {
            let candidate = prompt.candidate(usize::from(pos));
            let score = candidate.map(|candidate| {
                let score = self.weights.score(candidate);
                best.consider(candidate, score);
                score
            });
            let id = ids[usize::from(pos)] as u16;
            assessment.candidates.push(AssessmentCandidate {
                pos,
                id,
                eligible: score.is_some(),
                geometry: pack_4block_roots(roots[usize::from(id)]),
                features: candidate.map(|candidate| candidate.features),
                score,
                rank: None,
            });
        }
        let scores: Vec<_> = assessment
            .candidates
            .iter()
            .filter_map(|candidate| candidate.score)
            .collect();
        for candidate in &mut assessment.candidates {
            candidate.rank = candidate
                .score
                .map(|own| 1 + scores.iter().filter(|&&score| score > own).count() as u16);
            if gold_pos == Some(candidate.pos) {
                assessment.gold_rank = candidate.rank;
            }
            if rival_pos == Some(candidate.pos) {
                assessment.rival_rank = candidate.rank;
            }
        }
        assessment.winner_pos = best
            .read(self.weights.no_read())
            .map(|candidate| candidate.position);
        assessment.best_score = best.score;
        assessment.best_ties = best.ties;
        assessment
    }
}

fn make_probe(prompt: Option<Prompt<'_>>, weights: &WeightWords, enabled: bool) -> OpeningProbe {
    let best = if enabled {
        prompt.as_ref().map(|prompt| weights.best(prompt))
    } else {
        None
    };
    let decision = match &best {
        _ if !enabled => OpeningDecision::Disabled,
        None => OpeningDecision::FramingRejected,
        Some(best) if best.candidate.is_none() => OpeningDecision::NoEligibleCandidate,
        Some(best) if best.ties != 1 => OpeningDecision::ScoreTie,
        Some(best) if best.read(weights.no_read()).is_none() => OpeningDecision::NoReadThreshold,
        Some(_) => OpeningDecision::Selected,
    };
    let bounds = prompt.map(|prompt| prompt.fact_bounds());
    OpeningProbe {
        parsed_start: bounds.map(|bounds| bounds[0]),
        parsed_end: bounds.map(|bounds| bounds[1]),
        decision,
        best_score: best.as_ref().and_then(|best| best.score),
        best_ties: best.map_or(0, |best| best.ties),
        no_read_score: weights.no_read(),
    }
}

pub(super) fn read_from(
    reader: &mut dyn Read,
    header_bytes: &[u8],
    expected_body: Option<u64>,
) -> Result<IntegerDiscreteModel> {
    if header_bytes.len() > 1024 {
        return Err(invalid(
            "native occurrence envelope header exceeds size bound",
        ));
    }
    let envelope: Envelope = serde_json::from_slice(header_bytes)?;
    let body_size = MARKER_BYTES + STATE_BYTES as u64;
    if envelope.schema != SCHEMA
        || envelope.marker_sha256 != MARKER_SHA256
        || envelope.marker_bytes != MARKER_BYTES
        || envelope.state_bytes != STATE_BYTES
        || expected_body.is_some_and(|len| len != body_size)
        || !canonical_digest(&envelope.state_sha256)
        || !canonical_digest(&envelope.fit_sha256)
    {
        return Err(invalid("invalid native occurrence envelope bindings"));
    }
    let mut marker = HashingReader::new((&mut *reader).take(MARKER_BYTES));
    let mut header = [0; MARKER_HEADER_BYTES];
    marker.read_exact(&mut header)?;
    if hex::encode(Sha256::digest(header)) != MARKER_HEADER_SHA256 {
        return Err(invalid(
            "native occurrence marker header differs from pinned #4314",
        ));
    }
    let mut model = native_opening::read_from(
        &mut marker,
        &header[16..],
        Some(MARKER_BYTES - MARKER_HEADER_BYTES as u64),
    )?;
    if marker.inner.limit() != 0 || marker.finalize() != MARKER_SHA256 {
        return Err(invalid(
            "native occurrence nested marker digest or length mismatch",
        ));
    }
    let mut bytes = [0; STATE_BYTES];
    reader.read_exact(&mut bytes)?;
    if reader.read(&mut [0])? != 0 || hex::encode(Sha256::digest(bytes)) != envelope.state_sha256 {
        return Err(invalid("native occurrence state digest or length mismatch"));
    }
    model.occurrence = Some(State::new(envelope, &bytes)?);
    Ok(model)
}

fn canonical_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Losslessly pack the fitted i16 state against the exact frozen marker and writer.
/// The caller should publish the destination only after the stream succeeds.
pub fn write_artifact(
    marker: impl Read,
    mut trained_weights: impl Read,
    mut classes: impl Read,
    fit_sha256: [u8; 32],
    mut out: impl Write,
) -> Result<()> {
    let mut bytes = [0; STATE_BYTES];
    bytes[..STATE_MAGIC.len()].copy_from_slice(STATE_MAGIC);
    hex::decode_to_slice(
        SPEC_SHA256,
        &mut bytes[STATE_MAGIC.len()..STATE_PREFIX_BYTES],
    )
    .map_err(|_| invalid("invalid native occurrence feature spec digest"))?;
    trained_weights
        .read_exact(&mut bytes[STATE_PREFIX_BYTES..STATE_PREFIX_BYTES + WEIGHT_BYTES])?;
    classes.read_exact(&mut bytes[STATE_PREFIX_BYTES + WEIGHT_BYTES..])?;
    if trained_weights.read(&mut [0])? != 0 || classes.read(&mut [0])? != 0 {
        return Err(invalid("native occurrence state input has trailing bytes"));
    }
    let envelope = Envelope {
        schema: SCHEMA.to_owned(),
        marker_bytes: MARKER_BYTES,
        marker_sha256: MARKER_SHA256.to_owned(),
        state_bytes: STATE_BYTES,
        state_sha256: hex::encode(Sha256::digest(bytes)),
        fit_sha256: hex::encode(fit_sha256),
    };
    let header = serde_json::to_vec(&envelope)?;
    State::new(envelope, &bytes)?;
    out.write_all(MAGIC)?;
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(&header)?;
    let mut marker = HashingReader::new(marker);
    if std::io::copy(&mut (&mut marker).take(MARKER_BYTES), &mut out)? != MARKER_BYTES
        || marker.read(&mut [0])? != 0
        || marker.finalize() != MARKER_SHA256
    {
        return Err(invalid(
            "native occurrence frozen marker digest or length mismatch",
        ));
    }
    out.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winner_requires_a_unique_strict_improvement_over_no_read() {
        let first = Candidate {
            position: 4,
            token: 10,
            features: [0; ACTIVE],
        };
        let next = Candidate {
            position: 5,
            ..first
        };
        let mut best = Best::default();
        assert_eq!(best.read(0), None);
        best.consider(first, -1);
        assert_eq!(best.read(-1), None);
        assert_eq!(best.read(-2), Some(first));
        best.consider(next, -1);
        assert_eq!(best.read(-2), None);
        best.consider(next, 1);
        assert_eq!(best.read(0), Some(next));
        assert_eq!(best.read(1), None);
    }
}
