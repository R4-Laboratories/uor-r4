//! Sparse interval ranking over one original protocol-2 user turn.

use std::{
    cmp::Reverse,
    io::{Read, Write},
    ops::Range,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::{dialogue::EncodedUserOpening, ByteBpeTokenizer};

use super::{
    cl06_action, cl06_bfs, native_whole_turn, native_word_ending, orbit_root_coords, orbit_root_id,
    HashingReader, IntegerDiscreteModel, MAGIC, ROLE_COPY_STOPS,
};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.lut-discrete-native-interval/1";
pub const BANKS: usize = 16;
const SLOTS: usize = 256;
pub const WORDS: usize = BANKS * SLOTS;
pub const WEIGHT_BYTES: usize = WORDS * 2;
pub const MAX_SPAN: usize = 8;
const NO_READ_SLOTS: u8 = 224;
const AMBIGUOUS_QUERY_TOKENS: usize = 32;

type Keys = [u16; BANKS];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Geometry,
    Count,
}

impl Arm {
    fn tag(self) -> u8 {
        match self {
            Self::Geometry => 1,
            Self::Count => 2,
        }
    }

    fn bound_digest(self, bytes: &[u8; WEIGHT_BYTES]) -> String {
        let mut sha = Sha256::new();
        sha.update(SCHEMA.as_bytes());
        sha.update([self.tag()]);
        sha.update(bytes);
        hex::encode(sha.finalize())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Span {
    pub start: u16,
    pub end: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Candidate {
    #[serde(flatten)]
    pub span: Span,
    pub keys: Keys,
}

#[derive(Debug, Serialize)]
pub struct Features {
    pub arm: Arm,
    pub user: [u16; 2],
    pub fact: [u16; 2],
    pub question: [u16; 2],
    pub candidates: Vec<Candidate>,
    pub no_read_keys: Keys,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Selection {
    pub span: Option<Span>,
    pub best_candidate_score: Option<i32>,
    pub no_read_score: i32,
}

#[derive(Clone, Debug)]
struct Weights([i16; WORDS]);

impl Weights {
    fn parse(bytes: &[u8; WEIGHT_BYTES]) -> Self {
        Self(std::array::from_fn(|index| {
            i16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])
        }))
    }

    fn bytes(&self) -> [u8; WEIGHT_BYTES] {
        let mut bytes = [0; WEIGHT_BYTES];
        for (word, pair) in self.0.iter().zip(bytes.chunks_exact_mut(2)) {
            pair.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn score(&self, keys: &Keys) -> i32 {
        keys.iter()
            .map(|&key| i32::from(self.0[key as usize]))
            .sum()
    }

    fn select(&self, features: &Features) -> Selection {
        let no_read_score = self.score(&features.no_read_keys);
        let mut span = None;
        let mut best_candidate_score = None;
        for candidate in &features.candidates {
            let score = self.score(&candidate.keys);
            if best_candidate_score.is_none_or(|best| score > best) {
                best_candidate_score = Some(score);
                span = (score > no_read_score).then_some(candidate.span);
            }
        }
        Selection {
            span,
            best_candidate_score,
            no_read_score,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    base_sha256: String,
    base_bytes: u64,
    arm: u8,
    state_sha256: String,
    arm_state_sha256: String,
    state_bytes: usize,
    fit_sha256: String,
}

#[derive(Clone, Debug)]
pub(super) struct State {
    weights: Weights,
    arm: Arm,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct Identity<'a> {
    pub base_sha256: &'static str,
    pub arm: Arm,
    pub state_sha256: &'a str,
    pub state_bytes: usize,
    pub fit_sha256: &'a str,
}

pub fn identity(model: &IntegerDiscreteModel) -> Option<Identity<'_>> {
    model.interval.as_ref().map(|state| Identity {
        base_sha256: native_whole_turn::BASE_SHA256,
        arm: state.arm,
        state_sha256: &state.state_sha256,
        state_bytes: WEIGHT_BYTES,
        fit_sha256: &state.fit_sha256,
    })
}

pub(super) fn read_from(
    reader: &mut dyn Read,
    header: &[u8],
    expected_body: Option<u64>,
) -> Result<IntegerDiscreteModel> {
    let base_bytes = native_whole_turn::BASE_BYTES;
    if header.len() > 1024 {
        return Err(invalid("interval header exceeds size bound"));
    }
    let envelope: Envelope = serde_json::from_slice(header)?;
    let arm = match envelope.arm {
        1 => Arm::Geometry,
        2 => Arm::Count,
        _ => return Err(invalid("invalid interval arm tag")),
    };
    if envelope.schema != SCHEMA
        || envelope.base_sha256 != native_whole_turn::BASE_SHA256
        || envelope.base_bytes != base_bytes
        || envelope.state_bytes != WEIGHT_BYTES
        || expected_body.is_some_and(|len| len != base_bytes + WEIGHT_BYTES as u64)
        || !native_word_ending::canonical_digest(&envelope.state_sha256)
        || !native_word_ending::canonical_digest(&envelope.fit_sha256)
    {
        return Err(invalid("invalid interval artifact bindings"));
    }
    let mut bytes = [0; WEIGHT_BYTES];
    reader.read_exact(&mut bytes)?;
    if hex::encode(Sha256::digest(bytes)) != envelope.state_sha256 {
        return Err(invalid("interval state digest mismatch"));
    }
    if arm.bound_digest(&bytes) != envelope.arm_state_sha256 {
        return Err(invalid("interval arm/state binding mismatch"));
    }
    let mut model = native_whole_turn::read_balanced_base(reader)?;
    model.interval = Some(State {
        weights: Weights::parse(&bytes),
        arm,
        state_sha256: envelope.state_sha256,
        fit_sha256: envelope.fit_sha256,
    });
    Ok(model)
}

/// Package the original native signed words and frozen writer without conversion.
pub fn write_artifact(
    base: impl Read,
    mut weights: impl Read,
    arm: Arm,
    fit_sha256: [u8; 32],
    mut out: impl Write,
) -> Result<()> {
    let mut bytes = [0; WEIGHT_BYTES];
    weights.read_exact(&mut bytes)?;
    if weights.read(&mut [0])? != 0 {
        return Err(invalid("interval state has trailing bytes"));
    }
    let base_bytes = native_whole_turn::BASE_BYTES;
    let header = serde_json::to_vec(&Envelope {
        schema: SCHEMA.into(),
        base_sha256: native_whole_turn::BASE_SHA256.into(),
        base_bytes,
        arm: arm.tag(),
        state_sha256: hex::encode(Sha256::digest(bytes)),
        arm_state_sha256: arm.bound_digest(&bytes),
        state_bytes: WEIGHT_BYTES,
        fit_sha256: hex::encode(fit_sha256),
    })?;
    out.write_all(MAGIC)?;
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(&header)?;
    out.write_all(&bytes)?;
    let mut base = HashingReader::new(base);
    if std::io::copy(&mut (&mut base).take(base_bytes), &mut out)? != base_bytes
        || base.read(&mut [0])? != 0
        || base.finalize() != native_whole_turn::BASE_SHA256
    {
        return Err(invalid(
            "interval balanced artifact digest or length mismatch",
        ));
    }
    Ok(())
}

/// Read back the persisted little-endian words after a cold artifact load.
pub fn persisted_words(model: &IntegerDiscreteModel) -> Result<[u8; WEIGHT_BYTES]> {
    Ok(state(model)?.weights.bytes())
}

fn state(model: &IntegerDiscreteModel) -> Result<&State> {
    model
        .interval
        .as_ref()
        .ok_or_else(|| invalid("artifact has no interval state"))
}

pub fn select(
    model: &IntegerDiscreteModel,
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
) -> Result<Selection> {
    let state = state(model)?;
    Ok(state
        .weights
        .select(&features(model, opening, tokenizer, state.arm)?))
}

#[derive(Debug, Serialize)]
pub struct Scores {
    pub candidates: Vec<i32>,
    pub selection: Selection,
}

/// Exact candidate-order replay for a reopened artifact's own arm.
pub fn scores(model: &IntegerDiscreteModel, features: &Features) -> Option<Scores> {
    let state = model
        .interval
        .as_ref()
        .filter(|state| state.arm == features.arm)?;
    Some(Scores {
        candidates: features
            .candidates
            .iter()
            .map(|candidate| state.weights.score(&candidate.keys))
            .collect(),
        selection: state.weights.select(features),
    })
}

/// Training inventory and serving share candidate generation and key addressing.
pub fn features(
    model: &IntegerDiscreteModel,
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
    arm: Arm,
) -> Result<Features> {
    if model.interval.is_none() && model.artifact_sha256() != native_whole_turn::BASE_SHA256 {
        return Err(invalid(
            "interval features require the pinned balanced artifact",
        ));
    }
    if opening.tokenizer_cid() != native_whole_turn::TOKENIZER_CID
        || tokenizer.address() != native_whole_turn::TOKENIZER_CID
    {
        return Err(invalid("interval features require the pinned tokenizer"));
    }
    model.validate_prefix(opening.tokens())?;
    let user = opening.user_range();
    if user.end > u16::MAX as usize || model.codebook_roots.len() != tokenizer.vocab_size() {
        return Err(invalid("invalid interval address width or codebook"));
    }
    let original_ids = &opening.tokens()[user.clone()];
    let decoded: Vec<_> = original_ids
        .iter()
        .map(|&id| tokenizer.decode_bytes(&[id]))
        .collect();
    let mut prompt = Prompt::new(original_ids, &decoded, &model.codebook_roots);
    prompt.prepare();
    Ok(prompt.collect(arm, user.start))
}

fn keyed(slots: [u8; BANKS]) -> Keys {
    std::array::from_fn(|bank| (bank * SLOTS + usize::from(slots[bank])) as u16)
}

fn terminal(byte: u8) -> bool {
    matches!(byte, b'.' | b'?' | b'!' | b'\n')
}

fn token_class(bytes: &[u8]) -> u8 {
    match bytes
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
    {
        Some(b'a'..=b'z') => 1,
        Some(b'A'..=b'Z') => 2,
        Some(b'0'..=b'9') => 3,
        Some(byte) if !byte.is_ascii() => 4,
        Some(byte) if terminal(byte) => 5,
        Some(_) => 6,
        None => 0,
    }
}

#[derive(Clone, Debug)]
struct Word {
    spelling: String,
    start: usize,
    end: usize,
    capitalized: bool,
}

#[derive(Clone, Copy, Debug)]
struct Anchor {
    fact: usize,
    query: usize,
    quality: u8,
}

struct Prompt<'a> {
    ids: &'a [u32],
    bytes: &'a [Vec<u8>],
    roots: &'a [[u8; 4]],
    fact: Range<usize>,
    query: Range<usize>,
    left_edges: Vec<usize>,
    right_edges: Vec<usize>,
    anchors: Vec<Anchor>,
    starts: Vec<bool>,
    ends: Vec<bool>,
}

impl<'a> Prompt<'a> {
    fn new(ids: &'a [u32], bytes: &'a [Vec<u8>], roots: &'a [[u8; 4]]) -> Self {
        let mut byte_ids = Vec::new();
        let mut unsegmented = Vec::new();
        for (i, token) in bytes.iter().enumerate() {
            byte_ids.extend(std::iter::repeat_n(i, token.len()));
            unsegmented.extend(token);
        }
        let (fact_end, query) = if let Some(question) =
            unsegmented.iter().rposition(|&byte| byte == b'?')
        {
            let visible_prefix = &unsegmented[..question];
            let query_end = byte_ids[question] + 1;
            let separator = visible_prefix
                .iter()
                .rposition(|&byte| terminal(byte))
                .or_else(|| visible_prefix.iter().rposition(|&byte| byte == b':'));
            separator.map_or_else(
                || {
                    // An unseparated question leaves passage and query ambiguous.
                    // Retain every copyable site through the visible question
                    // and take its bounded suffix only for question features.
                    (
                        query_end,
                        query_end.saturating_sub(AMBIGUOUS_QUERY_TOKENS)..query_end,
                    )
                },
                |separator| (byte_ids[separator], byte_ids[separator] + 1..query_end),
            )
        } else if let Some(separator) = unsegmented.windows(2).rposition(|pair| pair == b"\n\n") {
            (byte_ids[separator], byte_ids[separator + 1] + 1..ids.len())
        } else {
            (0, 0..ids.len())
        };
        Self {
            ids,
            bytes,
            roots,
            fact: 0..fact_end,
            query,
            left_edges: vec![0; fact_end],
            right_edges: vec![fact_end; fact_end],
            anchors: Vec::new(),
            starts: vec![false; fact_end],
            ends: vec![false; fact_end],
        }
    }

    fn words(&self, range: Range<usize>) -> Vec<Word> {
        let mut words: Vec<Word> = Vec::new();
        let mut bytes = Vec::new();
        let mut positions = Vec::new();
        for position in range {
            let token = &self.bytes[position];
            bytes.extend(token);
            positions.extend(std::iter::repeat_n(position, token.len()));
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return words;
        };
        let mut inside = false;
        for (byte, character) in text.char_indices() {
            let is_word = character.is_alphanumeric();
            if is_word {
                let start = positions[byte];
                let end = positions[byte + character.len_utf8() - 1] + 1;
                if !inside {
                    words.push(Word {
                        spelling: String::new(),
                        start,
                        end,
                        capitalized: character.is_uppercase(),
                    });
                }
                let word = words.last_mut().expect("lexical character starts a word");
                word.spelling.extend(character.to_lowercase());
                word.end = end;
            }
            inside = is_word;
        }
        words
    }

    fn prepare(&mut self) {
        let mut left = self.fact.start;
        for i in self.fact.clone() {
            self.left_edges[i] = left;
            if self.bytes[i].iter().copied().any(terminal) {
                left = i + 1;
            }
        }
        let mut right = self.fact.end;
        for i in self.fact.clone().rev() {
            if self.bytes[i].iter().copied().any(terminal) {
                right = i;
            }
            self.right_edges[i] = right;
        }
        let facts = self.words(self.fact.clone());
        let question = self.words(self.query.clone());
        for fact in &facts {
            self.starts[fact.start] = true;
            self.ends[fact.end - 1] = true;
            if fact.spelling.len() < 3 {
                continue;
            }
            let Some(query) = question.iter().find(|word| word.spelling == fact.spelling) else {
                continue;
            };
            let repeated = facts
                .iter()
                .filter(|word| word.spelling == fact.spelling)
                .count()
                > 2;
            self.anchors.push(Anchor {
                fact: fact.start,
                query: query.start,
                quality: 1
                    + 2 * u8::from(!repeated)
                    + 4 * u8::from(fact.capitalized && query.capitalized),
            });
        }
    }

    fn collect(&self, arm: Arm, offset: usize) -> Features {
        let mut candidates = Vec::new();
        for start in self.fact.clone() {
            if !self.starts[start] || !self.copyable(start) {
                continue;
            }
            let last = (start + MAX_SPAN).min(self.right_edges[start]);
            for end in start + 1..=last {
                if !self.copyable(end - 1) {
                    break;
                }
                if self.ends[end - 1] {
                    candidates.push(Candidate {
                        span: Span {
                            start: (offset + start) as u16,
                            end: (offset + end) as u16,
                        },
                        keys: self.candidate_keys(start, end, arm),
                    });
                }
            }
        }
        let bounds =
            |range: &Range<usize>| [(offset + range.start) as u16, (offset + range.end) as u16];
        Features {
            arm,
            user: [offset as u16, (offset + self.ids.len()) as u16],
            fact: bounds(&self.fact),
            question: bounds(&self.query),
            candidates,
            no_read_keys: self.no_read_keys(arm),
        }
    }

    fn copyable(&self, position: usize) -> bool {
        !ROLE_COPY_STOPS.contains(&self.ids[position])
    }

    fn class(&self, position: Option<usize>) -> u8 {
        position.map_or(0, |i| token_class(&self.bytes[i]))
    }

    fn candidate_keys(&self, start: usize, end: usize, arm: Arm) -> Keys {
        let last = end - 1;
        let left = start.checked_sub(1);
        let right = (end < self.ids.len()).then_some(end);
        let clause = self.left_edges[start]..self.right_edges[last];
        let mut local = self.anchors.iter().filter(|a| clause.contains(&a.fact));
        let anchor = local
            .clone()
            .min_by_key(|a| (Reverse(a.quality), start.abs_diff(a.fact), a.fact));
        let quality = anchor.map_or(0, |a| a.quality);
        let source_anchor = anchor.map(|a| a.fact);
        let query_anchor = anchor.map(|a| a.query);
        let relative = |position| {
            source_anchor.map_or(31, |source| {
                ((position as isize - source as isize).clamp(-15, 15) + 15) as u8
            })
        };
        let class_start = self.class(Some(start));
        let class_last = self.class(Some(last));
        let mut slots = [0; BANKS];
        slots[..8].copy_from_slice(&[
            (end - start) as u8,
            class_start * 8 + class_last,
            quality * 16 + local.by_ref().count().min(15) as u8,
            relative(start),
            relative(end),
            ((start - clause.start).min(13) * 16 + (clause.end - end).min(15)) as u8,
            self.class(left) * 8 + self.class(right),
            quality * 16
                + self
                    .anchors
                    .iter()
                    .filter(|a| (start..end).contains(&a.fact))
                    .count()
                    .min(15) as u8,
        ]);
        let query_first = (!self.query.is_empty()).then_some(self.query.start);
        let query_last = (!self.query.is_empty()).then(|| self.query.end - 1);
        let site_pairs = [
            (query_anchor, source_anchor, Some(start)),
            (query_anchor, source_anchor, Some(last)),
            (query_anchor, source_anchor, left),
            (query_anchor, source_anchor, right),
            (query_first, Some(start), Some(last)),
            (query_last, Some(last), Some(start)),
            (Some(start), Some(last), left),
            (Some(last), Some(start), right),
        ];
        for (i, (query, source, context)) in site_pairs.into_iter().enumerate() {
            slots[8 + i] = self.route(arm, query, source, context, i % 4) % NO_READ_SLOTS;
        }
        keyed(slots)
    }

    fn no_read_keys(&self, arm: Arm) -> Keys {
        let first = (!self.query.is_empty()).then_some(self.query.start);
        let last = (!self.query.is_empty()).then(|| self.query.end - 1);
        let first_fact = (!self.fact.is_empty()).then_some(self.fact.start);
        let last_fact = (!self.fact.is_empty()).then(|| self.fact.end - 1);
        let strongest = self.anchors.iter().max_by_key(|anchor| anchor.quality);
        let fact_anchor = strongest.map(|a| a.fact);
        let query_anchor = strongest.map(|a| a.query);
        let mut slots = [0; BANKS];
        slots[..8].copy_from_slice(&[
            0,
            self.class(first),
            self.class(last),
            self.anchors.len().min(31) as u8,
            strongest.map_or(0, |anchor| anchor.quality),
            self.query.len().min(31) as u8,
            u8::from(!self.fact.is_empty()),
            self.anchors
                .iter()
                .filter(|a| a.quality >= 5)
                .count()
                .min(31) as u8,
        ]);
        let triples = [
            (first, fact_anchor, query_anchor),
            (last, fact_anchor, query_anchor),
            (query_anchor, fact_anchor, first_fact),
            (query_anchor, fact_anchor, last_fact),
            (first, last, query_anchor),
            (last, first, query_anchor),
            (first, first_fact, last_fact),
            (last, last_fact, first_fact),
        ];
        for (i, (query, source, context)) in triples.into_iter().enumerate() {
            slots[8 + i] = self.route(arm, query, source, context, i % 4) % 32;
        }
        keyed(slots.map(|slot| NO_READ_SLOTS + slot))
    }

    fn route(
        &self,
        arm: Arm,
        query: Option<usize>,
        source: Option<usize>,
        context: Option<usize>,
        block: usize,
    ) -> u8 {
        let [query, source, context] = [query, source, context].map(|p| p.map(|p| self.ids[p]));
        let [Some(query), Some(source), Some(context)] = [query, source, context] else {
            return 0;
        };
        match arm {
            Arm::Geometry => {
                let [q, s, c] =
                    [query, source, context].map(|id| self.roots[id as usize][block] as usize);
                let action = cl06_bfs(q, s);
                let transported = cl06_action(action, &orbit_root_coords(c));
                // Two signed axes identify an orbit root. Pair that transported
                // root with the original source root to retain direction.
                let root = orbit_root_id(&transported)
                    .expect("D11 Clifford action preserves the native orbit");
                ((s * 16 + root) % (usize::from(NO_READ_SLOTS) - 1) + 1) as u8
            }
            Arm::Count => {
                // Ordered token-context address with the same fixed number of
                // slots, update opportunities, and bank roles as Geometry.
                let address = ((query * 31 + source) * 31 + context) % u32::from(NO_READ_SLOTS - 1);
                (address + 1) as u8
            }
        }
    }
}

/// A detached lexical opening can follow the writer's native protocol space.
/// An untouched source ID that already carries space can begin immediately.
pub fn leading_space(
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
    span: Span,
) -> bool {
    opening
        .tokens()
        .get(span.start as usize)
        .is_some_and(|&id| {
            tokenizer
                .decode_bytes(&[id])
                .first()
                .is_some_and(|byte| !byte.is_ascii_whitespace())
        })
}
