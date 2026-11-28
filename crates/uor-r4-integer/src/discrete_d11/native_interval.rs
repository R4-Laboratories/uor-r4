//! Sparse interval ranking over one original protocol-2 user turn.

use std::{
    cmp::Reverse,
    io::{Read, Write},
    ops::Range,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::{ByteBpeTokenizer, dialogue::EncodedUserOpening};

use super::{
    HashingReader, IntegerDiscreteModel, MAGIC, ROLE_COPY_STOPS, cl06_action, cl06_bfs,
    cl06_compose, cl06_inverse, native_whole_turn, native_word_ending, orbit_root_coords,
    orbit_root_id,
};
use crate::{Result, invalid};

pub const SCHEMA: &str = "uor-r4.lut-discrete-native-interval/1";
pub const DIRECTED_SCHEMA: &str = "uor-r4.lut-discrete-native-interval/2";
pub const BANKS: usize = 16;
const SLOTS: usize = 256;
pub const WORDS: usize = BANKS * SLOTS;
pub const WEIGHT_BYTES: usize = WORDS * 2;
pub const MAX_SPAN: usize = 8;
/// Capacity of the read-only source census. Serving checks its own artifact.
pub const SOURCE_CENSUS_CONTEXT: usize = 512;
const NO_READ_SLOTS: u8 = 224;
const AMBIGUOUS_QUERY_TOKENS: usize = 32;
const RELATION_BANK: usize = 10;
const RELATION_BLOCK: usize = (RELATION_BANK - 8) % 4;
const RELATION_ACTIONS: u32 = 64;

type Keys = [u16; BANKS];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Geometry,
    Count,
    DirectedGeometry,
    DirectedCount,
    SwappedGeometry,
}

impl Arm {
    fn schema(self) -> &'static str {
        if self.is_directed() {
            DIRECTED_SCHEMA
        } else {
            SCHEMA
        }
    }

    pub fn is_directed(self) -> bool {
        matches!(
            self,
            Self::DirectedGeometry | Self::DirectedCount | Self::SwappedGeometry
        )
    }

    fn tag(self) -> u8 {
        match self {
            Self::Geometry => 1,
            Self::Count => 2,
            Self::DirectedGeometry => 3,
            Self::DirectedCount => 4,
            Self::SwappedGeometry => 5,
        }
    }

    fn bound_digest(self, bytes: &[u8; WEIGHT_BYTES]) -> String {
        let mut sha = Sha256::new();
        sha.update(self.schema().as_bytes());
        sha.update([self.tag()]);
        sha.update(bytes);
        hex::encode(sha.finalize())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
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

/// Original token positions for one bank-10 capture, in unreversed order.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct RelationEndpoints {
    pub query: [Option<u16>; 2],
    pub source: [Option<u16>; 2],
    pub query_anchor: Option<[u16; 2]>,
    pub source_anchor: Option<u16>,
    pub clause: [u16; 2],
}

#[derive(Debug, Serialize)]
pub struct RelationToken {
    pub id: u32,
    pub bytes: Vec<u8>,
    pub root: u8,
}

#[derive(Debug, Default, Serialize)]
pub struct RelationCapture {
    /// Parallel to `Features::candidates`.
    pub relation_endpoints: Vec<RelationEndpoints>,
    /// Absolute positions in the complete protocol-2 opening.
    pub relation_tokens: Vec<RelationToken>,
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

impl State {
    pub(super) fn schema(&self) -> &'static str {
        self.arm.schema()
    }
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
        3 => Arm::DirectedGeometry,
        4 => Arm::DirectedCount,
        5 => Arm::SwappedGeometry,
        _ => return Err(invalid("invalid interval arm tag")),
    };
    if envelope.schema != arm.schema()
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
        schema: arm.schema().into(),
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
    collect_features(model, opening, tokenizer, arm, None)
}

/// Collect one directed arm with its original protocol-2 positions and roots.
pub fn relation_features(
    model: &IntegerDiscreteModel,
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
    arm: Arm,
) -> Result<(Features, RelationCapture)> {
    if !arm.is_directed() {
        return Err(invalid("relation capture requires a directed interval arm"));
    }
    let mut capture = RelationCapture::default();
    let features = collect_features(model, opening, tokenizer, arm, Some(&mut capture))?;
    Ok((features, capture))
}

fn collect_features(
    model: &IntegerDiscreteModel,
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
    arm: Arm,
    capture: Option<&mut RelationCapture>,
) -> Result<Features> {
    if model.interval.is_none() && model.artifact_sha256() != native_whole_turn::BASE_SHA256 {
        return Err(invalid(
            "interval features require the pinned balanced artifact",
        ));
    }
    validate_source_tokenizer(opening, tokenizer)?;
    model.validate_prefix(opening.tokens())?;
    let user = opening.user_range();
    if user.end > u16::MAX as usize || model.codebook_roots.len() != tokenizer.vocab_size() {
        return Err(invalid("invalid interval address width or codebook"));
    }
    let original_ids = &opening.tokens()[user.clone()];
    let decoded = source_bytes(original_ids, tokenizer);
    let mut prompt = Prompt::new(original_ids, &decoded);
    prompt.prepare(arm.is_directed());
    let endpoints = capture.map(|capture| {
        capture.relation_tokens = opening
            .tokens()
            .iter()
            .map(|&id| RelationToken {
                id,
                bytes: tokenizer.decode_bytes(&[id]),
                root: model.codebook_roots[id as usize][RELATION_BLOCK],
            })
            .collect();
        &mut capture.relation_endpoints
    });
    Ok(prompt.collect(arm, user.start, &model.codebook_roots, endpoints))
}

fn validate_source_tokenizer(
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
) -> Result<()> {
    if opening.tokenizer_cid() != native_whole_turn::TOKENIZER_CID
        || tokenizer.address() != native_whole_turn::TOKENIZER_CID
    {
        return Err(invalid("interval features require the pinned tokenizer"));
    }
    Ok(())
}

fn source_bytes(ids: &[u32], tokenizer: &ByteBpeTokenizer) -> Vec<Vec<u8>> {
    ids.iter()
        .map(|&id| tokenizer.decode_bytes(&[id]))
        .collect()
}

/// Ordered source spans before arm-specific address keys or scoring.
#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SourceCandidates {
    pub user: [u16; 2],
    pub fact: [u16; 2],
    pub question: [u16; 2],
    pub candidates: Vec<Span>,
}

/// Enumerate source spans on the complete opening, without a model artifact.
/// NoRead is available separately from this ordered list in every interval arm.
pub fn census_source_candidates(
    opening: &EncodedUserOpening,
    tokenizer: &ByteBpeTokenizer,
) -> Result<SourceCandidates> {
    if opening.tokens().len() > SOURCE_CENSUS_CONTEXT {
        return Err(invalid("opening exceeds source census capacity"));
    }
    validate_source_tokenizer(opening, tokenizer)?;
    let user = opening.user_range();
    let ids = &opening.tokens()[user.clone()];
    let bytes = source_bytes(ids, tokenizer);
    let mut prompt = Prompt::new(ids, &bytes);
    prompt.prepare_spans();
    let offset = user.start;
    Ok(SourceCandidates {
        user: prompt.absolute_range(0..prompt.ids.len(), offset),
        fact: prompt.absolute_range(prompt.fact.clone(), offset),
        question: prompt.absolute_range(prompt.query.clone(), offset),
        candidates: prompt
            .source_spans()
            .map(|span| prompt.absolute_span(span, offset))
            .collect(),
    })
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
    query_end: usize,
    whole_query: bool,
    quality: u8,
}

#[derive(Clone, Copy)]
struct Relation {
    query: [Option<usize>; 2],
    source: [Option<usize>; 2],
}

impl Relation {
    fn describe(
        self,
        anchor: Option<&Anchor>,
        clause: Range<usize>,
        offset: usize,
    ) -> RelationEndpoints {
        let absolute = |position| (offset + position) as u16;
        let endpoints = |sites: [Option<usize>; 2]| sites.map(|site| site.map(absolute));
        RelationEndpoints {
            query: endpoints(self.query),
            source: endpoints(self.source),
            query_anchor: anchor.map(|a| [absolute(a.query), absolute(a.query_end)]),
            source_anchor: anchor.map(|a| absolute(a.fact)),
            clause: [absolute(clause.start), absolute(clause.end)],
        }
    }
}

#[derive(Clone, Copy)]
struct CliffordDisplacement(usize);

impl CliffordDisplacement {
    fn between([left, right]: [u8; 2]) -> Self {
        Self(cl06_bfs(usize::from(left), usize::from(right)))
    }

    fn relative_to(self, query: Self) -> Self {
        Self(cl06_compose(cl06_inverse(query.0), self.0))
    }
}

fn ordered_count_slot(ids: [u32; 4]) -> u8 {
    let mut digest = 2_166_136_261u32;
    for byte in ids.into_iter().flat_map(u32::to_le_bytes) {
        digest = (digest ^ u32::from(byte)).wrapping_mul(16_777_619);
    }
    (digest % RELATION_ACTIONS + 1) as u8
}

/// The nearest complete lexical word on either side, within the original
/// token window. Use the facing subtoken for a word spanning several tokens.
fn lexical_neighbors(
    words: &[Word],
    focus: Range<usize>,
    bounds: Range<usize>,
) -> [Option<usize>; 2] {
    let left = words[..words.partition_point(|word| word.end <= focus.start)]
        .last()
        .filter(|word| bounds.start <= word.start)
        .map(|word| word.end - 1);
    let right = words
        .get(words.partition_point(|word| word.start < focus.end))
        .filter(|word| word.end <= bounds.end)
        .map(|word| word.start);
    [left, right]
}

struct Prompt<'a> {
    ids: &'a [u32],
    bytes: &'a [Vec<u8>],
    fact: Range<usize>,
    query: Range<usize>,
    left_edges: Vec<usize>,
    right_edges: Vec<usize>,
    anchors: Vec<Anchor>,
    lexical_words: Option<Vec<Word>>,
    starts: Vec<bool>,
    ends: Vec<bool>,
}

impl<'a> Prompt<'a> {
    fn new(ids: &'a [u32], bytes: &'a [Vec<u8>]) -> Self {
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
            fact: 0..fact_end,
            query,
            left_edges: vec![0; fact_end],
            right_edges: vec![fact_end; fact_end],
            anchors: Vec::new(),
            lexical_words: None,
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

    fn prepare_spans(&mut self) -> Vec<Word> {
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
        for fact in &facts {
            self.starts[fact.start] = true;
            self.ends[fact.end - 1] = true;
        }
        facts
    }

    fn prepare(&mut self, directed: bool) {
        let facts = self.prepare_spans();
        let question = self.words(self.query.clone());
        // The bounded suffix of an unseparated query can begin inside a word
        // or a UTF-8 character. Keep the original words for directed endpoints.
        self.lexical_words = directed.then(|| self.words(0..self.ids.len()));
        for fact in &facts {
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
                query_end: query.end,
                whole_query: self.lexical_words.as_ref().is_some_and(|words| {
                    let first = words.partition_point(|word| word.start < query.start);
                    words[first..]
                        .iter()
                        .take_while(|word| word.start == query.start)
                        .any(|word| word.end == query.end && word.spelling == query.spelling)
                }),
                quality: 1
                    + 2 * u8::from(!repeated)
                    + 4 * u8::from(fact.capitalized && query.capitalized),
            });
        }
    }

    fn source_spans(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        self.fact
            .clone()
            .filter(|&start| self.starts[start] && self.copyable(start))
            .flat_map(|start| {
                let last = (start + MAX_SPAN).min(self.right_edges[start]);
                (start + 1..=last)
                    .take_while(|&end| self.copyable(end - 1))
                    .filter(|&end| self.ends[end - 1])
                    .map(move |end| start..end)
            })
    }

    fn absolute_range(&self, range: Range<usize>, offset: usize) -> [u16; 2] {
        [(offset + range.start) as u16, (offset + range.end) as u16]
    }

    fn absolute_span(&self, range: Range<usize>, offset: usize) -> Span {
        let [start, end] = self.absolute_range(range, offset);
        Span { start, end }
    }

    fn collect(
        &self,
        arm: Arm,
        offset: usize,
        roots: &[[u8; 4]],
        mut capture: Option<&mut Vec<RelationEndpoints>>,
    ) -> Features {
        let candidates = self
            .source_spans()
            .map(|span| Candidate {
                span: self.absolute_span(span.clone(), offset),
                keys: self.candidate_keys(span, arm, offset, roots, capture.as_deref_mut()),
            })
            .collect();
        Features {
            arm,
            user: self.absolute_range(0..self.ids.len(), offset),
            fact: self.absolute_range(self.fact.clone(), offset),
            question: self.absolute_range(self.query.clone(), offset),
            candidates,
            no_read_keys: self.no_read_keys(arm, roots),
        }
    }

    fn copyable(&self, position: usize) -> bool {
        !ROLE_COPY_STOPS.contains(&self.ids[position])
    }

    fn class(&self, position: Option<usize>) -> u8 {
        position.map_or(0, |i| token_class(&self.bytes[i]))
    }

    fn candidate_keys(
        &self,
        span: Range<usize>,
        arm: Arm,
        offset: usize,
        roots: &[[u8; 4]],
        capture: Option<&mut Vec<RelationEndpoints>>,
    ) -> Keys {
        let Range { start, end } = span;
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
            slots[8 + i] = self.route(arm, query, source, context, i % 4, roots) % NO_READ_SLOTS;
        }
        if let Some(words) = &self.lexical_words {
            let relation = Relation {
                query: anchor
                    .filter(|a| a.whole_query)
                    .map_or([None; 2], |anchor| {
                        lexical_neighbors(words, anchor.query..anchor.query_end, self.query.clone())
                    }),
                source: lexical_neighbors(words, start..end, clause.clone()),
            };
            slots[RELATION_BANK] = self.relation_slot(relation, arm, roots);
            if let Some(capture) = capture {
                capture.push(relation.describe(anchor, clause, offset));
            }
        }
        keyed(slots)
    }

    fn relation_slot(&self, relation: Relation, arm: Arm, roots: &[[u8; 4]]) -> u8 {
        let [Some(q_left), Some(q_right)] = relation.query else {
            return 0;
        };
        let [Some(s_left), Some(s_right)] = relation.source else {
            return 0;
        };
        let ids = [q_left, q_right, s_left, s_right].map(|position| self.ids[position]);
        if arm == Arm::DirectedCount {
            return ordered_count_slot(ids);
        }
        let [q_left, q_right, s_left, s_right] = ids.map(|id| roots[id as usize][RELATION_BLOCK]);
        let query = CliffordDisplacement::between([q_left, q_right]);
        let source = CliffordDisplacement::between(if arm == Arm::SwappedGeometry {
            [s_right, s_left]
        } else {
            [s_left, s_right]
        });
        (source.relative_to(query).0 + 1) as u8
    }

    fn no_read_keys(&self, arm: Arm, roots: &[[u8; 4]]) -> Keys {
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
            slots[8 + i] = self.route(arm, query, source, context, i % 4, roots) % 32;
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
        roots: &[[u8; 4]],
    ) -> u8 {
        let [query, source, context] = [query, source, context].map(|p| p.map(|p| self.ids[p]));
        let [Some(query), Some(source), Some(context)] = [query, source, context] else {
            return 0;
        };
        match arm {
            Arm::Geometry | Arm::DirectedGeometry | Arm::DirectedCount | Arm::SwappedGeometry => {
                let [q, s, c] =
                    [query, source, context].map(|id| roots[id as usize][block] as usize);
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

#[cfg(test)]
mod tests {
    use super::*;

    const DIRECTED_ARMS: [Arm; 3] = [
        Arm::DirectedGeometry,
        Arm::DirectedCount,
        Arm::SwappedGeometry,
    ];

    #[test]
    fn relative_transport_uses_inverse_query_on_the_left() {
        let query = CliffordDisplacement::between([0, 2]);
        let source = CliffordDisplacement::between([1, 3]);
        let swapped = CliffordDisplacement::between([3, 1]);
        assert_eq!((query.0, source.0, swapped.0), (7, 1, 42));
        assert_eq!(source.relative_to(query).0, 18);
        assert_eq!(cl06_compose(source.0, cl06_inverse(query.0)), 3);
        assert_eq!(swapped.relative_to(query).0, 55);
        assert_eq!(cl06_inverse(source.0), 51);
        assert_ne!(swapped.0, cl06_inverse(source.0));

        let axes = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            cl06_action(source.relative_to(query).0, &axes),
            cl06_action(cl06_inverse(query.0), &cl06_action(source.0, &axes)),
        );
    }

    #[test]
    fn four_site_count_matches_the_reachable_clifford_slots() {
        let transports: Vec<_> = (0..16)
            .flat_map(|left| (0..16).map(move |right| CliffordDisplacement::between([left, right])))
            .collect();
        let mut geometry = [false; RELATION_ACTIONS as usize];
        for &query in &transports {
            for &source in &transports {
                geometry[source.relative_to(query).0] = true;
            }
        }
        assert!(geometry.into_iter().all(|reachable| reachable));
        for position in 0..4 {
            let mut count = [false; RELATION_ACTIONS as usize];
            for id in 0..RELATION_ACTIONS {
                let mut ids = [701, 952, 1024, 1502];
                ids[position] = id;
                count[usize::from(ordered_count_slot(ids) - 1)] = true;
            }
            assert_eq!(count, geometry);
        }
        assert_eq!(ordered_count_slot([11, 22, 33, 44]), 22);
        assert_eq!(ordered_count_slot([11, 22, 44, 33]), 6);
    }

    #[test]
    fn ambiguous_query_suffix_keeps_original_word_boundaries() {
        for (left_half, suffix, anchor_word, first_word, expected_query) in [
            (
                &b" cy"[..],
                &b"an"[..],
                " Keeper",
                " Keeper",
                [None, Some(5)],
            ),
            (
                &b" star"[..],
                &b"Anvil"[..],
                " left",
                " Anvil",
                [None, None],
            ),
            (
                &b" caf\xc3"[..],
                &b"\xa9"[..],
                " Keeper",
                " Keeper",
                [None, None],
            ),
        ] {
            let mut bytes = vec![
                b"Amber".to_vec(),
                first_word.as_bytes().to_vec(),
                left_half.to_vec(),
                suffix.to_vec(),
                anchor_word.as_bytes().to_vec(),
                b" bird".to_vec(),
            ];
            bytes.extend(std::iter::repeat_n(b" x".to_vec(), 28));
            bytes.push(b"?".to_vec());
            let ids: Vec<u32> = (20..20 + bytes.len() as u32).collect();
            let roots = [[1; 4]; 64];
            let mut prompt = Prompt::new(&ids, &bytes);
            assert_eq!(prompt.query.start, 3);
            prompt.prepare(true);
            for arm in DIRECTED_ARMS {
                let mut trace = Vec::new();
                let features = prompt.collect(arm, 0, &roots, Some(&mut trace));
                let index = features
                    .candidates
                    .iter()
                    .position(|candidate| candidate.span == (Span { start: 1, end: 2 }))
                    .expect("complete first source word");
                assert_eq!(trace[index].query, expected_query);
                assert_eq!(
                    features.candidates[index].keys[RELATION_BANK],
                    (RELATION_BANK * SLOTS) as u16
                );
            }
        }
    }

    #[test]
    fn directed_capture_changes_only_bank_ten_at_original_lexical_sites() {
        let ids: Vec<_> = (20..35).collect();
        let bytes = [
            "First", " Bind", "ed", " am", "ber", " fox", " we", "st", " ridge", ".", " north",
            " Bind", "ed", " clear", "?",
        ]
        .map(|part| part.as_bytes().to_vec());
        let mut roots = [[13; 4]; 35];
        for (position, root) in [(10, 0), (13, 2), (4, 1), (6, 3)] {
            roots[ids[position] as usize][RELATION_BLOCK] = root;
        }

        let offset = 3;
        let mut original = Prompt::new(&ids, &bytes);
        original.prepare(false);
        let legacy = original.collect(Arm::Geometry, offset, &roots, None);
        let mut directed = Prompt::new(&ids, &bytes);
        directed.prepare(true);
        let relation_bank = (RELATION_BANK * SLOTS) as u16;
        for (arm, slot) in DIRECTED_ARMS.into_iter().zip([19, 41, 56]) {
            let mut trace = Vec::new();
            let captured = directed.collect(arm, offset, &roots, Some(&mut trace));
            let serving = directed.collect(arm, offset, &roots, None);
            assert_eq!(captured.candidates, serving.candidates);
            assert_eq!(captured.no_read_keys, serving.no_read_keys);
            assert_eq!(captured.no_read_keys, legacy.no_read_keys);
            assert_eq!(
                (captured.user, captured.fact, captured.question),
                (legacy.user, legacy.fact, legacy.question)
            );
            assert_eq!(captured.candidates.len(), legacy.candidates.len());
            assert_eq!(captured.candidates.len(), trace.len());
            for ((a, b), endpoints) in captured
                .candidates
                .iter()
                .zip(&legacy.candidates)
                .zip(&trace)
            {
                assert_eq!(a.span, b.span);
                for bank in (0..BANKS).filter(|&bank| bank != RELATION_BANK) {
                    assert_eq!(a.keys[bank], b.keys[bank]);
                }
                if endpoints.source.contains(&None) {
                    assert_eq!(a.keys[RELATION_BANK], relation_bank);
                }
            }
            let i = captured
                .candidates
                .iter()
                .position(|candidate| candidate.span == (Span { start: 8, end: 9 }))
                .expect("fox candidate");
            assert_eq!(
                trace[i],
                RelationEndpoints {
                    query: [Some(13), Some(16)],
                    source: [Some(7), Some(9)],
                    query_anchor: Some([14, 16]),
                    source_anchor: Some(4),
                    clause: [3, 12],
                }
            );
            assert_eq!(
                captured.candidates[i].keys[RELATION_BANK],
                relation_bank + slot
            );

            for missing in 0..4 {
                let mut sites = [Some(10), Some(13), Some(4), Some(6)];
                sites[missing] = None;
                let [q_left, q_right, s_left, s_right] = sites;
                let relation = Relation {
                    query: [q_left, q_right],
                    source: [s_left, s_right],
                };
                assert_eq!(directed.relation_slot(relation, arm, &roots), 0);
            }
        }
    }
}
