use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Sha = String;

pub fn require(condition: bool, reason: impl std::fmt::Display) -> Result<()> {
    if !condition {
        return Err(
            std::io::Error::new(std::io::ErrorKind::InvalidData, reason.to_string()).into(),
        );
    }
    Ok(())
}

pub fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn is_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn pinned<T: serde::de::DeserializeOwned>(path: &Path, digest: &str) -> Result<T> {
    let bytes = std::fs::read(path)?;
    require(
        is_sha(digest) && sha(&bytes) == digest,
        format!("SHA-256 mismatch: {}", path.display()),
    )?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    TwoAnswer,
    Answerability,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mixture {
    Original,
    Natural,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Key {
    Count,
    Geometry,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fit {
    Actual,
    Permuted,
    Zero,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub mixture: Mixture,
    pub key: Key,
    pub fit: Fit,
}

pub fn arms() -> impl Iterator<Item = Arm> {
    [Mixture::Original, Mixture::Natural]
        .into_iter()
        .flat_map(|mixture| {
            [Key::Count, Key::Geometry]
                .into_iter()
                .flat_map(move |key| {
                    [Fit::Actual, Fit::Permuted, Fit::Zero]
                        .into_iter()
                        .map(move |fit| Arm { mixture, key, fit })
                })
        })
}

pub const PROPOSED: Arm = Arm {
    mixture: Mixture::Natural,
    key: Key::Geometry,
    fit: Fit::Actual,
};
pub const NATURAL_COUNT: Arm = Arm {
    key: Key::Count,
    ..PROPOSED
};
pub const HISTORICAL_GEOMETRY: Arm = Arm {
    mixture: Mixture::Original,
    ..PROPOSED
};
pub const HISTORICAL_COUNT: Arm = Arm {
    key: Key::Count,
    ..HISTORICAL_GEOMETRY
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub start: u16,
    pub end: u16,
}

impl Span {
    pub fn fits(self, bounds: [u16; 2]) -> bool {
        bounds[0] <= self.start
            && self.start < self.end
            && self.end <= bounds[1]
            && self.end - self.start <= 8
    }
    pub fn tokens(self, ids: &[u32]) -> Option<&[u32]> {
        ids.get(usize::from(self.start)..usize::from(self.end))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Negative {
    Unanswered,
    ContradictedPremise,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Target {
    Source {
        span: Span,
        source_ids: Vec<u32>,
        expected_ids: Vec<u32>,
        response_lexical_offset: usize,
    },
    NoRead {
        group: Negative,
    },
}

impl Target {
    pub fn span(&self) -> Option<Span> {
        match self {
            Self::Source { span, .. } => Some(*span),
            Self::NoRead { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub opening_sha256: Sha,
    pub prompt_ids: Vec<u32>,
    pub user: [u16; 2],
    pub question: [u16; 2],
    pub target: Target,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Opening {
    pub fact: [u16; 2],
    pub fact_ids: Vec<u32>,
    pub candidates: Vec<Span>,
    pub cases: Vec<Case>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "stage", rename_all = "snake_case", deny_unknown_fields)]
pub enum Qualification {
    SourceExcluded { reason: String },
    CaptureExcluded { reason: String },
    RebindExcluded { reason: String },
    Qualified { opening: Opening },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Study {
    Original4340,
    Expanded4341,
}

impl Study {
    pub fn additional(self, family: Family, title: &str) -> usize {
        match (self, family, title) {
            (Self::Expanded4341, Family::Answerability, "1973_oil_crisis") => 10,
            (Self::Expanded4341, Family::Answerability, "Harvard_University") => 13,
            (Self::Expanded4341, Family::Answerability, "Islamism") => 23,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub passage_sha256: Sha,
    pub left_id: String,
    pub right_id: String,
    pub qualification: Qualification,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Title {
    pub name: String,
    pub family: Family,
    pub mechanical_candidates: usize,
    pub passages: usize,
    pub selected: Vec<Pair>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ControlSource {
    pub title: String,
    pub passage_sha256: Sha,
    pub openings: Vec<Opening>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Training {
    pub mixture: Mixture,
    pub sources: Vec<ControlSource>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Roster {
    pub schema: String,
    pub study: Study,
    pub registration_sha256: Sha,
    pub selection_sha256: Sha,
    pub judgments_sha256: Sha,
    pub source_auth_sha256: Sha,
    pub tokenizer_sha256: Sha,
    pub protected_reference_sha256: Sha,
    pub titles: Vec<Title>,
    pub training: Vec<Training>,
    pub protected: Vec<ControlSource>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    #[serde(deserialize_with = "Option::deserialize")]
    pub span: Option<Span>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub best_candidate_score: Option<i32>,
    pub no_read_score: i32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scores {
    pub candidates: Vec<i32>,
    pub selection: Selection,
}

// This projection understands the ordinary client's --interval-features JSONL.
// Keys are compared before this projection by #4337's verify-device.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NativeFeature {
    pub case_id: String,
    pub arm: Key,
    pub user: [u16; 2],
    pub fact: [u16; 2],
    pub question: [u16; 2],
    pub candidates: Vec<NativeCandidate>,
    pub scores: Option<Scores>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NativeCandidate {
    pub start: u16,
    pub end: u16,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    pub request_id: String,
    pub opening_sha256: Sha,
    pub features: Vec<NativeFeature>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceRow {
    pub case_id: String,
    pub scores: Scores,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Panel {
    pub capture: Vec<Capture>,
    pub reference: Vec<ReferenceRow>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub arm: Arm,
    pub artifact_sha256: Sha,
    pub state_sha256: Sha,
    pub verified_device_sha256: Sha,
    pub fit_input_sha256: Sha,
    pub fit_replay_state_sha256: Sha,
    pub fit_output_sha256: Sha,
    pub fit_replay_output_sha256: Sha,
    pub validation: Panel,
    pub swaps: Vec<Capture>,
    pub train: Panel,
    pub protected: Panel,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub schema: String,
    pub roster_sha256: Sha,
    pub ordinary_client_sha256: Sha,
    pub base_writer_sha256: Sha,
    pub tokenizer_sha256: Sha,
    pub protected_reference_sha256: Sha,
    pub models: Vec<Model>,
}
