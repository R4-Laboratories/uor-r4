//! Protocol-2 occurrence admission over the serializer's fresh user turn.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uor_r4_tokenizer::dialogue::EncodedUserOpening;

use super::{
    native_occurrence::{Candidate, OpeningAssessment, OpeningProbe, WeightWords, WEIGHT_BYTES},
    native_source_choice, native_word_ending, HashingReader, IntegerDiscreteModel, MAGIC,
};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.lut-discrete-native-whole-turn/1";
pub const BASE_SHA256: &str = "5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa";
const BASE_BYTES: u64 = 1_202_177_144;
const BASE_HEADER_BYTES: usize = 358;
const BASE_HEADER_SHA256: &str = "66b9959d08c86252e1aef4583f9c04f22d9697ac40dc84c4757d3017a3ee419a";
pub(super) const TOKENIZER_CID: &str =
    "blake3:3338e1a777cf20c8f85df9971d73e99cc6ad1ab25434cb797a792471a33a3b27";

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    base_sha256: String,
    base_bytes: u64,
    state_sha256: String,
    state_bytes: usize,
    fit_sha256: String,
}

#[derive(Clone, Debug)]
pub(super) struct State {
    weights: WeightWords,
    state_sha256: String,
    fit_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct Identity<'a> {
    pub base_sha256: &'static str,
    pub state_sha256: &'a str,
    pub state_bytes: usize,
    pub fit_sha256: &'a str,
}

pub fn identity(model: &IntegerDiscreteModel) -> Option<Identity<'_>> {
    model.whole_turn.as_ref().map(|state| Identity {
        base_sha256: BASE_SHA256,
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
    if header.len() > 1024 {
        return Err(invalid("whole-turn occurrence header exceeds size bound"));
    }
    let envelope: Envelope = serde_json::from_slice(header)?;
    if envelope.schema != SCHEMA
        || envelope.base_sha256 != BASE_SHA256
        || envelope.base_bytes != BASE_BYTES
        || envelope.state_bytes != WEIGHT_BYTES
        || expected_body.is_some_and(|len| len != BASE_BYTES + WEIGHT_BYTES as u64)
        || !native_word_ending::canonical_digest(&envelope.state_sha256)
        || !native_word_ending::canonical_digest(&envelope.fit_sha256)
    {
        return Err(invalid("invalid whole-turn occurrence bindings"));
    }
    let mut bytes = [0; WEIGHT_BYTES];
    reader.read_exact(&mut bytes)?;
    if hex::encode(Sha256::digest(bytes)) != envelope.state_sha256 {
        return Err(invalid("whole-turn occurrence state digest mismatch"));
    }
    let weights = WeightWords::parse(&bytes)?;
    let mut base = HashingReader::new((&mut *reader).take(BASE_BYTES));
    let mut preamble = [0; BASE_HEADER_BYTES];
    base.read_exact(&mut preamble)?;
    if hex::encode(Sha256::digest(preamble)) != BASE_HEADER_SHA256 {
        return Err(invalid(
            "whole-turn occurrence requires the pinned balanced header",
        ));
    }
    let mut model = native_source_choice::read_from(
        &mut base,
        &preamble[16..],
        Some(BASE_BYTES - BASE_HEADER_BYTES as u64),
    )?;
    if base.inner.limit() != 0 || base.finalize() != BASE_SHA256 || reader.read(&mut [0])? != 0 {
        return Err(invalid(
            "whole-turn occurrence base digest or length mismatch",
        ));
    }
    model.whole_turn = Some(State {
        weights,
        state_sha256: envelope.state_sha256,
        fit_sha256: envelope.fit_sha256,
    });
    Ok(model)
}

/// Transport all signed words byte-for-byte, followed by the complete balanced
/// artifact. The caller publishes only a completed, independently loaded file.
pub fn write_artifact(
    base: impl Read,
    mut weights: impl Read,
    fit_sha256: [u8; 32],
    mut out: impl Write,
) -> Result<()> {
    let mut bytes = [0; WEIGHT_BYTES];
    weights.read_exact(&mut bytes)?;
    if weights.read(&mut [0])? != 0 {
        return Err(invalid("whole-turn occurrence words have trailing bytes"));
    }
    WeightWords::parse(&bytes)?;
    let header = serde_json::to_vec(&Envelope {
        schema: SCHEMA.into(),
        base_sha256: BASE_SHA256.into(),
        base_bytes: BASE_BYTES,
        state_sha256: hex::encode(Sha256::digest(bytes)),
        state_bytes: WEIGHT_BYTES,
        fit_sha256: hex::encode(fit_sha256),
    })?;
    out.write_all(MAGIC)?;
    out.write_all(&(header.len() as u64).to_le_bytes())?;
    out.write_all(&header)?;
    out.write_all(&bytes)?;
    let mut base = HashingReader::new(base);
    if std::io::copy(&mut (&mut base).take(BASE_BYTES), &mut out)? != BASE_BYTES
        || base.read(&mut [0])? != 0
        || base.finalize() != BASE_SHA256
    {
        return Err(invalid(
            "whole-turn occurrence balanced artifact digest or length mismatch",
        ));
    }
    Ok(())
}

/// Original reader words from the exact balanced comparison artifact.
pub fn balanced_words(model: &IntegerDiscreteModel) -> Result<[u8; WEIGHT_BYTES]> {
    if model.artifact_sha256() != BASE_SHA256 {
        return Err(invalid(
            "whole-turn comparison requires the pinned balanced artifact",
        ));
    }
    Ok(occurrence(model)?.weight_bytes())
}

/// New envelope words, in their persisted little-endian order.
pub fn expanded_words(model: &IntegerDiscreteModel) -> Result<[u8; WEIGHT_BYTES]> {
    Ok(state(model)?.weights.bytes())
}

/// Source-only differential view. The artifact decides whether to use the
/// framed or explicitly serialized user-turn scope.
pub fn assess(
    model: &IntegerDiscreteModel,
    opening: &EncodedUserOpening,
) -> Result<OpeningAssessment> {
    validate_opening(model, opening)?;
    if model.whole_turn.is_none() && model.artifact_sha256() != BASE_SHA256 {
        return Err(invalid(
            "the framed arm requires the pinned balanced artifact",
        ));
    }
    occurrence(model)?.assess_opening(
        opening,
        &model.codebook_roots,
        model.whole_turn.as_ref().map(|state| &state.weights),
    )
}

pub fn probe(
    model: &IntegerDiscreteModel,
    opening: Option<&EncodedUserOpening>,
) -> Result<OpeningProbe> {
    if let Some(opening) = opening {
        validate_opening(model, opening)?;
    }
    Ok(occurrence(model)?.probe_user(opening, &model.codebook_roots, &state(model)?.weights))
}

pub(super) fn choose(
    model: &IntegerDiscreteModel,
    opening: &EncodedUserOpening,
) -> Result<Option<Candidate>> {
    validate_opening(model, opening)?;
    occurrence(model)?.choose_user(opening, &model.codebook_roots, &state(model)?.weights)
}

fn validate_opening(model: &IntegerDiscreteModel, opening: &EncodedUserOpening) -> Result<()> {
    if opening.tokenizer_cid() != TOKENIZER_CID {
        return Err(invalid(
            "whole-turn occurrence requires the pinned tokenizer",
        ));
    }
    model.validate_prefix(opening.tokens())
}

fn state(model: &IntegerDiscreteModel) -> Result<&State> {
    model
        .whole_turn
        .as_ref()
        .ok_or_else(|| invalid("artifact has no whole-turn occurrence state"))
}

fn occurrence(model: &IntegerDiscreteModel) -> Result<&super::native_occurrence::State> {
    model
        .occurrence
        .as_ref()
        .ok_or_else(|| invalid("artifact has no framed occurrence state"))
}
