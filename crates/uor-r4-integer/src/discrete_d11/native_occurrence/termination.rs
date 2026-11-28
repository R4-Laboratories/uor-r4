//! Optional learned EOS promotion on the ordinary native writer transition.

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    cl06_action, cl06_bfs, native_copy, native_interval, native_whole_turn, native_word_ending,
    orbit_root_coords, orbit_root_id, HashingReader, IntegerDiscreteModel, MAGIC,
};
use crate::{invalid, Result};

pub const SCHEMA: &str = "uor-r4.lut-discrete-ordinary-termination/1";
pub const BANKS: usize = 16;
pub const SLOTS: usize = 256;
pub const WORDS: usize = BANKS * SLOTS;
pub const WEIGHT_BYTES: usize = WORDS * 2;
pub type Keys = [u16; BANKS];

const INTERVAL_SHA256: &str = "6b7cb196d036b0ec223390ff10dbbaecc46c120468f863f9092818125a689443";
const INTERVAL_BYTES: u64 = 1_202_185_782;
const MAX_HEADER: usize = 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Geometry,
    Ordered,
}

impl Arm {
    fn tag(self) -> u8 {
        match self {
            Self::Geometry => 1,
            Self::Ordered => 2,
        }
    }

    fn bound_digest(self, bytes: &[u8]) -> String {
        let mut digest = Sha256::new();
        digest.update(SCHEMA.as_bytes());
        digest.update([self.tag()]);
        digest.update(bytes);
        hex::encode(digest.finalize())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Off,
    Generic,
    VerifiedEnd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Site {
    Ordinary,
    VerifiedEnd,
    Authorized,
}

#[derive(Clone, Debug)]
pub struct Weights(Box<[i16; WORDS]>);

impl Weights {
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != WEIGHT_BYTES {
            return Err(invalid(
                "ordinary termination weights have the wrong byte length",
            ));
        }
        let mut words = Box::new([0; WORDS]);
        for (word, bytes) in words.iter_mut().zip(bytes.chunks_exact(2)) {
            *word = i16::from_le_bytes([bytes[0], bytes[1]]);
        }
        Ok(Self(words))
    }

    pub fn to_le_bytes(&self) -> [u8; WEIGHT_BYTES] {
        let mut bytes = [0; WEIGHT_BYTES];
        for (out, word) in bytes.chunks_exact_mut(2).zip(self.0.iter()) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    pub fn score(&self, keys: &Keys) -> Result<i32> {
        let mut score = 0;
        for (bank, &key) in keys.iter().enumerate() {
            let key = usize::from(key);
            if key / SLOTS != bank {
                return Err(invalid("ordinary termination key lies outside its bank"));
            }
            score += i32::from(self.0[key]);
        }
        Ok(score)
    }
}

#[derive(Debug, Serialize)]
pub struct Features {
    pub geometry_keys: Keys,
    pub ordered_keys: Keys,
}

impl Features {
    fn keys(&self, arm: Arm) -> &Keys {
        match arm {
            Arm::Geometry => &self.geometry_keys,
            Arm::Ordered => &self.ordered_keys,
        }
    }
}

fn key(bank: usize, slot: usize) -> u16 {
    (bank * SLOTS + slot) as u16
}

fn root_pair(left: usize, right: usize) -> usize {
    (left << 4) | right
}

fn code_root(code: u16, block: usize) -> usize {
    usize::from((code >> (4 * block)) & 15)
}

/// The same causal operator supplies training keys and reads the saved words.
/// `prefix` contains the real IDs before the decision, without the target ID.
pub fn features(prefix: &[u32], writer: &native_copy::WriterEvent) -> Result<Features> {
    let (blocks, tail) = writer.read_vector.as_chunks::<8>();
    if prefix.is_empty() || blocks.len() != 4 || !tail.is_empty() {
        return Err(invalid(
            "ordinary termination requires four native writer blocks and a prefix",
        ));
    }
    let mut roots = [0; 4];
    for (root, coords) in roots.iter_mut().zip(blocks) {
        *root = orbit_root_id(coords).ok_or_else(|| {
            invalid("ordinary termination writer block is outside the native orbit")
        })?;
    }
    let mut geometry_keys = [0; BANKS];
    for block in 0..4 {
        let read = code_root(writer.read_code, block);
        let recurrence = code_root(writer.recurrence_code, block);
        let post = roots[block];
        let transported = cl06_action(cl06_bfs(recurrence, read), &orbit_root_coords(post));
        let transported = orbit_root_id(&transported)
            .ok_or_else(|| invalid("ordinary termination transport left the native orbit"))?;
        geometry_keys[block] = key(block, root_pair(read, post));
        geometry_keys[4 + block] = key(4 + block, root_pair(recurrence, post));
        geometry_keys[8 + block] = key(8 + block, root_pair(recurrence, read));
        geometry_keys[12 + block] = key(12 + block, root_pair(roots[(block + 1) % 4], transported));
    }
    let causal_id = |back: usize| {
        prefix
            .len()
            .checked_sub(back + 1)
            .map_or(0, |position| prefix[position].wrapping_add(1))
    };
    let ordered_keys = std::array::from_fn(|bank| {
        let distance = bank / 4 + 1;
        let right = bank % 4;
        let pair = causal_id(right + distance)
            .wrapping_mul(31)
            .wrapping_add(causal_id(right));
        key(bank, (pair & 255) as usize)
    });
    Ok(Features {
        geometry_keys,
        ordered_keys,
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    base_sha256: String,
    base_bytes: u64,
    arm: Arm,
    state_sha256: String,
    arm_state_sha256: String,
    state_bytes: usize,
    fit_sha256: String,
}

impl Envelope {
    fn validate(&self, expected_body: Option<u64>) -> Result<Base> {
        let base = Base::from_sha256(&self.base_sha256)?;
        if self.schema != SCHEMA
            || self.base_bytes != base.bytes()
            || self.state_bytes != WEIGHT_BYTES
            || expected_body.is_some_and(|len| len != base.bytes() + WEIGHT_BYTES as u64)
            || !native_word_ending::canonical_digest(&self.state_sha256)
            || !native_word_ending::canonical_digest(&self.arm_state_sha256)
            || !native_word_ending::canonical_digest(&self.fit_sha256)
        {
            return Err(invalid("invalid ordinary termination artifact bindings"));
        }
        Ok(base)
    }

    fn read_weights(&self, reader: &mut dyn Read) -> Result<Weights> {
        let mut bytes = [0; WEIGHT_BYTES];
        reader.read_exact(&mut bytes)?;
        if hex::encode(Sha256::digest(bytes)) != self.state_sha256
            || self.arm.bound_digest(&bytes) != self.arm_state_sha256
        {
            return Err(invalid("ordinary termination arm or state digest mismatch"));
        }
        Weights::from_le_bytes(&bytes)
    }
}

#[derive(Clone, Debug)]
pub(in crate::discrete_d11) struct State {
    weights: Weights,
    base_sha256: &'static str,
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
    model.termination.as_ref().map(|state| Identity {
        base_sha256: state.base_sha256,
        arm: state.arm,
        state_sha256: &state.state_sha256,
        state_bytes: WEIGHT_BYTES,
        fit_sha256: &state.fit_sha256,
    })
}

pub fn persisted_words(model: &IntegerDiscreteModel) -> Result<[u8; WEIGHT_BYTES]> {
    Ok(model
        .termination
        .as_ref()
        .ok_or_else(missing)?
        .weights
        .to_le_bytes())
}

fn missing() -> crate::IntegerError {
    invalid("ordinary termination mode requires a bound learned artifact")
}

#[derive(Debug, Serialize)]
pub struct Decision {
    #[serde(flatten)]
    pub features: Features,
    pub score: Option<i32>,
    pub site: Site,
    pub ordinary_id: u32,
    pub committed_id: u32,
    pub promoted: bool,
}

pub fn observe(
    model: &IntegerDiscreteModel,
    prefix: &[u32],
    writer: &mut native_copy::WriterEvent,
    mode: Mode,
    site: Site,
) -> Result<Decision> {
    let features = features(prefix, writer)?;
    let score = match &model.termination {
        Some(state) => Some(state.weights.score(features.keys(state.arm))?),
        None if mode == Mode::Off => None,
        None => return Err(missing()),
    };
    let ordinary_id = writer.committed_id;
    writer.committed_id = resolve(mode, site, score.unwrap_or(0), ordinary_id);
    Ok(Decision {
        features,
        score,
        site,
        ordinary_id,
        committed_id: writer.committed_id,
        promoted: ordinary_id != writer.committed_id,
    })
}

fn resolve(mode: Mode, site: Site, score: i32, ordinary_id: u32) -> u32 {
    let eligible = matches!(
        (mode, site),
        (Mode::Generic, Site::Ordinary | Site::VerifiedEnd)
            | (Mode::VerifiedEnd, Site::VerifiedEnd)
    );
    if eligible && score > 0 {
        native_copy::EOS
    } else {
        ordinary_id
    }
}

#[derive(Clone, Copy)]
enum Base {
    Balanced,
    Interval,
}

impl Base {
    fn from_sha256(digest: &str) -> Result<Self> {
        match digest {
            native_whole_turn::BASE_SHA256 => Ok(Self::Balanced),
            INTERVAL_SHA256 => Ok(Self::Interval),
            _ => Err(invalid(
                "ordinary termination requires a registered frozen writer",
            )),
        }
    }

    fn sha256(self) -> &'static str {
        match self {
            Self::Balanced => native_whole_turn::BASE_SHA256,
            Self::Interval => INTERVAL_SHA256,
        }
    }

    fn bytes(self) -> u64 {
        match self {
            Self::Balanced => native_whole_turn::BASE_BYTES,
            Self::Interval => INTERVAL_BYTES,
        }
    }

    fn load(self, reader: &mut dyn Read) -> Result<IntegerDiscreteModel> {
        let mut base = HashingReader::new((&mut *reader).take(self.bytes()));
        let model = match self {
            Self::Balanced => native_whole_turn::read_balanced_base(&mut base)?,
            Self::Interval => {
                let mut magic = [0; 8];
                let mut length = [0; 8];
                base.read_exact(&mut magic)?;
                base.read_exact(&mut length)?;
                let length = u64::from_le_bytes(length);
                if magic != *MAGIC || length > MAX_HEADER as u64 {
                    return Err(invalid(
                        "ordinary termination interval base header is invalid",
                    ));
                }
                let mut header = [0; MAX_HEADER];
                let header = &mut header[..length as usize];
                base.read_exact(header)?;
                native_interval::read_from(&mut base, header, Some(self.bytes() - 16 - length))?
            }
        };
        if base.inner.limit() != 0
            || base.finalize() != self.sha256()
            || reader.read(&mut [0])? != 0
        {
            return Err(invalid(
                "ordinary termination base digest or length mismatch",
            ));
        }
        Ok(model)
    }
}

pub fn allows_capture(model: &IntegerDiscreteModel) -> bool {
    identity(model).is_some() || Base::from_sha256(model.artifact_sha256()).is_ok()
}

pub(in crate::discrete_d11) fn read_from(
    reader: &mut dyn Read,
    header: &[u8],
    expected_body: Option<u64>,
) -> Result<IntegerDiscreteModel> {
    if header.len() > MAX_HEADER {
        return Err(invalid("ordinary termination header exceeds size bound"));
    }
    let envelope: Envelope = serde_json::from_slice(header)?;
    let base = envelope.validate(expected_body)?;
    let weights = envelope.read_weights(reader)?;
    let mut model = base.load(reader)?;
    model.termination = Some(State {
        weights,
        base_sha256: base.sha256(),
        arm: envelope.arm,
        state_sha256: envelope.state_sha256,
        fit_sha256: envelope.fit_sha256,
    });
    Ok(model)
}

/// Transport the fitted little-endian words and the frozen writer verbatim.
pub fn write_artifact(
    base: impl Read,
    base_sha256: &str,
    mut weights: impl Read,
    arm: Arm,
    fit_sha256: [u8; 32],
    mut output: impl Write,
) -> Result<()> {
    let frozen = Base::from_sha256(base_sha256)?;
    let mut bytes = [0; WEIGHT_BYTES];
    weights.read_exact(&mut bytes)?;
    if weights.read(&mut [0])? != 0 {
        return Err(invalid("ordinary termination state has trailing bytes"));
    }
    let header = serde_json::to_vec(&Envelope {
        schema: SCHEMA.into(),
        base_sha256: frozen.sha256().into(),
        base_bytes: frozen.bytes(),
        arm,
        state_sha256: hex::encode(Sha256::digest(bytes)),
        arm_state_sha256: arm.bound_digest(&bytes),
        state_bytes: WEIGHT_BYTES,
        fit_sha256: hex::encode(fit_sha256),
    })?;
    output.write_all(MAGIC)?;
    output.write_all(&(header.len() as u64).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(&bytes)?;
    let mut input = HashingReader::new(base);
    if std::io::copy(&mut (&mut input).take(frozen.bytes()), &mut output)? != frozen.bytes()
        || input.read(&mut [0])? != 0
        || input.finalize() != frozen.sha256()
    {
        return Err(invalid(
            "ordinary termination base digest or length mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn writer() -> native_copy::WriterEvent {
        native_copy::WriterEvent {
            read_pos: None,
            claim_pos: None,
            gate_id: 37,
            source_id: None,
            source_distance: None,
            source_admitted: false,
            minimum_distance: 12,
            first_admitted_cell: 0,
            native_proposal: 37,
            committed_id: 37,
            copied: false,
            read_code: 0x4321,
            recurrence_code: 0x1234,
            causal_successor: None,
            read_vector: [5, 6, 7, 8]
                .into_iter()
                .flat_map(orbit_root_coords)
                .collect(),
        }
    }

    #[test]
    fn ordinary_termination_only_promotes_on_eligible_positive_scores() -> Result<()> {
        let writer = writer();
        let features = features(&[0, 15, 99, 131], &writer)?;
        let mut words = [0; WEIGHT_BYTES];
        let zero = Weights::from_le_bytes(&words)?;
        assert_eq!(zero.score(&features.geometry_keys)?, 0);
        assert_eq!(resolve(Mode::Generic, Site::Ordinary, 0, 37), 37);
        let index = usize::from(features.geometry_keys[0]) * 2;
        words[index..index + 2].copy_from_slice(&7i16.to_le_bytes());
        let learned = Weights::from_le_bytes(&words)?;
        let score = learned.score(&features.geometry_keys)?;
        assert_eq!(learned.to_le_bytes(), words);
        assert_eq!(score, 7);
        assert_eq!(
            resolve(Mode::Generic, Site::Ordinary, score, 37),
            native_copy::EOS
        );
        assert_eq!(resolve(Mode::Off, Site::Ordinary, score, 37), 37);
        assert_eq!(resolve(Mode::Generic, Site::Authorized, score, 37), 37);
        assert_eq!(resolve(Mode::VerifiedEnd, Site::Ordinary, score, 37), 37);
        assert_eq!(
            resolve(Mode::VerifiedEnd, Site::VerifiedEnd, score, 37),
            native_copy::EOS
        );
        assert_eq!(resolve(Mode::Generic, Site::Ordinary, -score, 37), 37);
        assert_eq!(
            resolve(Mode::Generic, Site::Ordinary, -score, native_copy::EOS),
            native_copy::EOS
        );
        Ok(())
    }

    #[test]
    fn ordinary_termination_features_use_native_state_and_ordered_control() -> Result<()> {
        let mut writer = writer();
        let a = features(&[0, 15, 99, 131], &writer)?;
        let b = features(&[0, 15, 131, 99], &writer)?;
        assert_eq!(a.geometry_keys, b.geometry_keys);
        assert_ne!(a.ordered_keys, b.ordered_keys);
        assert_eq!(a.geometry_keys[0], 0x15);
        assert_eq!(a.geometry_keys[4], 4 * 256 + 0x45);
        assert_eq!(a.geometry_keys[8], 8 * 256 + 0x41);
        assert_eq!(a.ordered_keys[0], (31 * 100 + 132) & 255);
        writer.recurrence_code ^= 1;
        let changed = features(&[0, 15, 99, 131], &writer)?;
        assert_ne!(changed.geometry_keys, a.geometry_keys);
        assert_eq!(changed.ordered_keys, a.ordered_keys);
        writer.read_vector[1] = 1234;
        assert!(features(&[0], &writer).is_err());
        Ok(())
    }

    #[test]
    fn ordinary_termination_refuses_malformed_words_keys_and_bindings() -> Result<()> {
        let state = [0; WEIGHT_BYTES];
        let mut envelope = Envelope {
            schema: SCHEMA.into(),
            base_sha256: native_whole_turn::BASE_SHA256.into(),
            base_bytes: native_whole_turn::BASE_BYTES,
            arm: Arm::Geometry,
            state_sha256: hex::encode(Sha256::digest(state)),
            arm_state_sha256: Arm::Geometry.bound_digest(&state),
            state_bytes: WEIGHT_BYTES,
            fit_sha256: hex::encode([1; 32]),
        };
        let body_len = native_whole_turn::BASE_BYTES + WEIGHT_BYTES as u64;
        assert!(envelope.validate(Some(body_len)).is_ok());
        assert!(envelope.validate(Some(body_len - 1)).is_err());
        let weights = envelope.read_weights(&mut state.as_slice())?;
        assert!(weights.score(&[0; BANKS]).is_err());
        assert!(Weights::from_le_bytes(&state[1..]).is_err());
        assert!(envelope.read_weights(&mut &state[1..]).is_err());
        let mut corrupt = state;
        corrupt[1] = 1;
        assert!(envelope.read_weights(&mut corrupt.as_slice()).is_err());
        envelope.arm = Arm::Ordered;
        assert!(envelope.read_weights(&mut state.as_slice()).is_err());
        envelope.fit_sha256.make_ascii_uppercase();
        // Invalid digest syntax is rejected before any large nested read.
        envelope.fit_sha256.replace_range(0..1, "F");
        assert!(envelope.validate(None).is_err());
        Ok(())
    }
}
