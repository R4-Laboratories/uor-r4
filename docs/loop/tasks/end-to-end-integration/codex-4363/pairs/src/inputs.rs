use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uor_r4_tokenizer::ByteBpeTokenizer;

use crate::{Result, digest, present, require};

const DONOR_BANK: &str = include_str!("../../../../../../../docs/labs/m2-donor-corpus/instrument/src/bank.rs");

const TOKENIZER: (&str, &str) = (
    "20261127-codex-visible-answer/panel/inputs/tokenizer.json",
    "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
);
const DEV: (&str, &str) = (
    "20261127-codex-visible-answer/panel/inputs/coqa-dev-v1.0.json",
    "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a",
);
const D87: (&str, &str) = (
    "20261127-codex-visible-answer/panel/inputs/d87-bank-prompt-digests.txt",
    "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541",
);
const ROSTERS: [(&str, &str, &str, usize); 5] = [
    (
        "4312",
        "20261127-codex-visible-answer/panel/frozen-v1/judgments.jsonl",
        "bea3926ba99a205a2035ae81c8aaa78cc864f6cccf408a071bcccca4553f86aa",
        200,
    ),
    (
        "4313",
        "20261127-codex-question-swap/panel/frozen-v2/keys.jsonl",
        "866a01878f6c2890a4dbe715a45808df171afe47f7bbc466bf74f6cf578815bd",
        5,
    ),
    (
        "4314-train",
        "20261127-codex-marker-question/panel-v2/train/keys.jsonl",
        "91a7f5c743dd5ede11e5ce6876229f5692a3cb82cf770ae7d0882a8634d2165d",
        0,
    ),
    (
        "4314-held",
        "20261127-codex-marker-question/panel-v2/held/keys.jsonl",
        "c7e67e6c96f297bd1e17614354f9525325895872b82d872fea6999974310acc5",
        5,
    ),
    (
        "4315",
        "20261127-codex-shared-source/panel-v1/human/keys.jsonl",
        "fcd87c28afde5a9e1dd9284d31e4630fe28535c8cde475e153b669cfc6f9f231",
        29,
    ),
];

#[derive(Deserialize)]
pub struct Coqa {
    pub data: Vec<Conversation>,
}

#[derive(Clone, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub filename: String,
    pub source: String,
    pub story: String,
    pub questions: Vec<Turn>,
    pub answers: Vec<Turn>,
}

#[derive(Clone, Deserialize)]
pub struct Turn {
    pub input_text: String,
    pub turn_id: usize,
    pub span_start: Option<i64>,
    pub span_end: Option<i64>,
    pub span_text: Option<String>,
}

impl Conversation {
    pub fn valid_turns(&self) -> bool {
        self.questions.len() == self.answers.len()
            && self
                .questions
                .iter()
                .zip(&self.answers)
                .enumerate()
                .all(|(i, (q, a))| q.turn_id == i + 1 && a.turn_id == i + 1)
    }

    pub fn key(&self) -> (String, String) {
        (self.source.clone(), self.filename.clone())
    }
}

pub enum Pool<'a> {
    Dev,
    Train(&'a Path, &'a str),
}

impl Pool<'_> {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Train(..) => "train-as-held",
        }
    }

    pub fn source_salt(&self) -> &'static str {
        match self {
            Self::Dev => "uor-r4/4352/coqa/source/v1",
            Self::Train(..) => "uor-r4/4352/coqa-train-as-held/source/v1",
        }
    }

    pub fn id_salt(&self) -> &'static str {
        match self {
            Self::Dev => "uor-r4/4352/coqa/blind/v1",
            Self::Train(..) => "uor-r4/4352/coqa-train-as-held/blind/v1",
        }
    }
}

#[derive(Default)]
pub struct Exclusions {
    source_ids: HashSet<String>,
    source_files: HashSet<(String, String)>,
    passages: HashSet<[u8; 32]>,
    panel_questions: HashSet<String>,
    d87_prompts: HashSet<[u8; 8]>,
    d87_templates: HashSet<String>,
}

impl Exclusions {
    fn add_source(&mut self, doc: &Conversation) {
        self.source_ids.insert(doc.id.clone());
        self.source_files.insert(doc.key());
        self.passages.insert(passage_digest(&doc.story));
    }

    pub fn source_hit(&self, doc: &Conversation, passage: [u8; 32]) -> Option<&'static str> {
        if self.source_ids.contains(&doc.id) {
            Some("prior_conversation_id")
        } else if self.source_files.contains(&doc.key()) {
            Some("prior_source_filename")
        } else if self.passages.contains(&passage) {
            Some("prior_complete_passage")
        } else {
            None
        }
    }

    pub fn prompt_hit(
        &self,
        first_user: &str,
        questions: impl Iterator<Item = impl AsRef<str>>,
    ) -> Option<&'static str> {
        if std::iter::once(normalize(first_user))
            .chain(questions.map(|text| normalize(text.as_ref())))
            .any(|text| {
                self.d87_prompts.contains(&short_digest(&text))
                    || self.d87_templates.contains(&text)
            })
        {
            return Some("d87_exact_prompt_or_template");
        }
        None
    }

    pub fn prior_question_hit(&self, mut questions: impl Iterator<Item = impl AsRef<str>>) -> bool {
        questions.any(|text| self.panel_questions.contains(&normalize(text.as_ref())))
    }
}

#[derive(Serialize)]
pub struct InputReceipts {
    pub files: BTreeMap<String, String>,
    pub known_coqa_roster_counts: BTreeMap<&'static str, usize>,
    pub historical_union_4312_through_4314: usize,
    pub known_scored_source_union: usize,
    pub preview_rosters: BTreeMap<String, PreviewReceipt>,
    pub preview_source_union: usize,
    pub preview_only_source_ids: usize,
    pub excluded_source_ids: usize,
    pub excluded_source_filenames: usize,
    pub excluded_complete_passage_digests: usize,
    pub known_panel_question_digests: usize,
    pub d87_prompt_digests: usize,
    pub d87_normalized_followup_templates: usize,
}

#[derive(Serialize)]
pub struct PreviewReceipt {
    sha256: String,
    rows: usize,
    unique_sources: usize,
}

pub struct Inputs {
    pub data: Vec<Conversation>,
    pub tokenizer: ByteBpeTokenizer,
    pub exclusions: Exclusions,
    pub receipts: InputReceipts,
}

pub fn normalize(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn passage_digest(text: &str) -> [u8; 32] {
    digest(normalize(text).as_bytes())
}

fn short_digest(normalized: &str) -> [u8; 8] {
    digest(normalized.as_bytes())[..8]
        .try_into()
        .expect("digest width")
}

fn pinned(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path)?;
    require(hex::encode(digest(&bytes)) == sha, "input SHA-256 mismatch")?;
    Ok(bytes)
}

fn followups() -> Result<HashSet<String>> {
    let (lines, _) = present(
        DONOR_BANK
            .split_once("pub const FOLLOW_UPS: [&str; 40] = [\n")
            .and_then(|(_, rest)| rest.split_once("\n];")),
        "original D87 follow-up source changed",
    )?;
    let templates = lines
        .lines()
        .map(|line| {
            present(
                line.trim()
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix("\",")),
                "malformed D87 template",
            )
            .map(normalize)
        })
        .collect::<Result<HashSet<_>>>()?;
    require(templates.len() == 40, "original D87 template count changed")?;
    Ok(templates)
}

fn load_previews(
    path: &Path,
    source_by_id: &BTreeMap<&str, &Conversation>,
    exclusions: &mut Exclusions,
) -> Result<(BTreeMap<String, PreviewReceipt>, HashSet<String>)> {
    let mut files = BTreeMap::new();
    let mut union = HashSet::new();
    for (name, bounds) in [
        ("4313-reproduced.jsonl", Some((14, 12))),
        ("4314-reproduced.jsonl", Some((11, 8))),
        ("4315-reproduced.jsonl", None),
        ("4313-retained.jsonl", None),
        ("4314-retained.jsonl", None),
        ("4315-retained.jsonl", None),
    ] {
        let file = path.join(name);
        if name.contains("-retained") && !file.exists() {
            continue;
        }
        let bytes = fs::read(file)?;
        let mut ids = HashSet::new();
        let mut count = 0usize;
        for line in std::str::from_utf8(&bytes)?.lines() {
            let row: Value = serde_json::from_str(line)?;
            let id = present(
                row["source"]["source_conversation_id"]
                    .as_str()
                    .or_else(|| row["id"].as_str()),
                "preview source ID missing",
            )?;
            let questions = if let Some(items) = row["questions"].as_array() {
                items.iter().map(Value::as_str).collect::<Option<Vec<_>>>()
            } else {
                row["question"].as_str().map(|text| vec![text])
            };
            let questions = present(questions, "preview question missing")?;
            require(!questions.is_empty(), "preview question list is empty")?;
            exclusions
                .panel_questions
                .extend(questions.into_iter().map(normalize));
            let doc = present(
                source_by_id.get(id),
                "preview source ID absent from pinned DEV",
            )?;
            ids.insert(id.to_owned());
            exclusions.add_source(doc);
            count += 1;
        }
        require(count > 0, "empty historical preview")?;
        if let Some((rows, sources)) = bounds {
            require(
                count == rows && ids.len() == sources,
                "historical preview count differs",
            )?;
        }
        files.insert(
            name.to_owned(),
            PreviewReceipt {
                sha256: hex::encode(digest(&bytes)),
                rows: count,
                unique_sources: ids.len(),
            },
        );
        union.extend(ids);
    }
    Ok((files, union))
}

pub fn load(runs: &Path, previews: &Path, pool: &Pool<'_>) -> Result<Inputs> {
    let mut pins = BTreeMap::new();
    let mut read = |path: &str, sha: &str| {
        let bytes = pinned(&runs.join(path), sha)?;
        pins.insert(path.to_owned(), sha.to_owned());
        Ok::<_, Box<dyn std::error::Error>>(bytes)
    };

    let dev: Coqa = serde_json::from_slice(&read(DEV.0, DEV.1)?)?;
    let source_by_id: BTreeMap<_, _> = dev.data.iter().map(|doc| (doc.id.as_str(), doc)).collect();
    require(
        source_by_id.len() == dev.data.len(),
        "repeated CoQA DEV source ID",
    )?;
    let tokenizer = present(
        ByteBpeTokenizer::from_tokenizer_json_bytes(&read(TOKENIZER.0, TOKENIZER.1)?),
        "invalid pinned tokenizer",
    )?;
    let bank_bytes = read(D87.0, D87.1)?;
    let mut exclusions = Exclusions::default();
    for line in std::str::from_utf8(&bank_bytes)?.lines() {
        let mut digest = [0; 8];
        hex::decode_to_slice(line, &mut digest)?;
        require(
            exclusions.d87_prompts.insert(digest),
            "repeated D87 prompt digest",
        )?;
    }
    require(!exclusions.d87_prompts.is_empty(), "empty D87 prompt bank")?;
    exclusions.d87_templates = followups()?;

    let mut roster_counts = BTreeMap::new();
    let mut known_union = HashSet::new();
    let mut prior_union = 0;
    for (name, path, sha, expected) in ROSTERS {
        let bytes = read(path, sha)?;
        let mut ids = HashSet::new();
        for line in std::str::from_utf8(&bytes)?.lines() {
            let row: Value = serde_json::from_str(line)?;
            let identity = if name == "4312" {
                &row["custody"]
            } else {
                &row["source"]
            };
            let Some(id) = identity["source_conversation_id"].as_str() else {
                continue;
            };
            require(
                identity["dataset"] == "CoQA dev v1.0",
                "historical source is not pinned CoQA DEV",
            )?;
            let doc = present(source_by_id.get(id), "prior CoQA source ID absent from DEV")?;
            require(
                identity["source"] == doc.source && identity["passage_filename"] == doc.filename,
                "historical CoQA source filename or domain differs",
            )?;
            let question = present(row["question"].as_str(), "historical CoQA question missing")?;
            exclusions.panel_questions.insert(normalize(question));
            ids.insert(id.to_owned());
            exclusions.add_source(doc);
        }
        require(
            ids.len() == expected,
            format!("CoQA roster count differs for {name}"),
        )?;
        known_union.extend(ids);
        roster_counts.insert(name, expected);
        if name == "4314-held" {
            prior_union = known_union.len();
        }
    }
    require(
        prior_union == 210 && known_union.len() == 239,
        "known CoQA union changed",
    )?;
    let (preview_rosters, preview_ids) = load_previews(previews, &source_by_id, &mut exclusions)?;

    let data = match pool {
        Pool::Dev => dev.data,
        Pool::Train(path, sha) => {
            for doc in &dev.data {
                exclusions.add_source(doc);
            }
            let bytes = pinned(path, sha)?;
            pins.insert("supplemental CoQA source".to_owned(), (*sha).to_owned());
            serde_json::from_slice::<Coqa>(&bytes)?.data
        }
    };
    let receipts = InputReceipts {
        files: pins,
        known_coqa_roster_counts: roster_counts,
        historical_union_4312_through_4314: prior_union,
        known_scored_source_union: known_union.len(),
        preview_rosters,
        preview_source_union: preview_ids.len(),
        preview_only_source_ids: preview_ids.difference(&known_union).count(),
        excluded_source_ids: exclusions.source_ids.len(),
        excluded_source_filenames: exclusions.source_files.len(),
        excluded_complete_passage_digests: exclusions.passages.len(),
        known_panel_question_digests: exclusions.panel_questions.len(),
        d87_prompt_digests: exclusions.d87_prompts.len(),
        d87_normalized_followup_templates: exclusions.d87_templates.len(),
    };
    Ok(Inputs {
        data,
        tokenizer,
        exclusions,
        receipts,
    })
}

pub fn planted_exclusions() -> Result<()> {
    let doc = |id: &str, filename: &str, story: &str| Conversation {
        id: id.to_owned(),
        filename: filename.to_owned(),
        source: "cnn".to_owned(),
        story: story.to_owned(),
        questions: Vec::new(),
        answers: Vec::new(),
    };
    let original = doc("enrolled", "same-file", "An enrolled story.");
    let alias = doc("new-id", "other-file", "AN  ENROLLED story");
    let filename = doc("new-id", "same-file", "Different story");
    let mut exclusion = Exclusions::default();
    exclusion.add_source(&original);
    exclusion
        .d87_prompts
        .insert(short_digest(&normalize("Bank overlap?")));
    exclusion
        .d87_templates
        .insert(normalize("Continue the example."));
    exclusion
        .panel_questions
        .insert(normalize("Panel overlap?"));
    for (row, expected) in [
        (&original, "prior_conversation_id"),
        (&alias, "prior_complete_passage"),
        (&filename, "prior_source_filename"),
    ] {
        require(
            exclusion.source_hit(row, passage_digest(&row.story)) == Some(expected),
            "planted prior source overlap escaped",
        )?;
    }
    require(
        exclusion
            .prompt_hit("Clean prompt", ["BANK overlap!"].into_iter())
            .is_some()
            && exclusion
                .prompt_hit("Bank overlap", ["clean"].into_iter())
                .is_some()
            && exclusion
                .prompt_hit("Clean prompt", ["Continue the example"].into_iter())
                .is_some()
            && exclusion.prior_question_hit(["panel overlap"].into_iter())
            && exclusion
                .prompt_hit("Clean prompt", ["unrelated"].into_iter())
                .is_none(),
        "planted question overlap control failed",
    )
}
