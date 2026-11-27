use crate::*;

const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
const COQA: &str = "20261127-codex-visible-answer/panel/inputs/coqa-dev-v1.0.json";
const BANK: &str = "20261127-codex-visible-answer/panel/inputs/d87-bank-prompt-digests.txt";
const PROMPTS: &str = "20261127-codex-visible-answer/panel/frozen-v1/prompts.jsonl";
const JUDGMENTS: &str = "20261127-codex-visible-answer/panel/frozen-v1/judgments.jsonl";
const SWAP_ROWS: &str = "20261127-codex-question-swap/panel/frozen-v2/rows.jsonl";
const SWAP_KEYS: &str = "20261127-codex-question-swap/panel/frozen-v2/keys.jsonl";
const MARKER_TRAIN_ROWS: &str = "20261127-codex-marker-question/panel-v2/train/rows.jsonl";
const MARKER_TRAIN_KEYS: &str = "20261127-codex-marker-question/panel-v2/train/keys.jsonl";
const MARKER_HELD_ROWS: &str = "20261127-codex-marker-question/panel-v2/held/rows.jsonl";
const MARKER_HELD_KEYS: &str = "20261127-codex-marker-question/panel-v2/held/keys.jsonl";
const COLON: &str = "20261127-codex-marker-question/donor-v2/colon-targets.jsonl";
const OPENINGS: &str = "20261127-codex-marker-question/donor-v2/openings.jsonl";
const UORT: &str = "20261127-codex-marker-question/donor-v2/answer-first.tokens.u16";

const PINS: [(&str, &str); 14] = [
    (
        TOKENIZER,
        "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
    ),
    (
        COQA,
        "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a",
    ),
    (
        BANK,
        "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541",
    ),
    (
        PROMPTS,
        "021f4c1fe8e10884ecaa3cb3b67ed290356fb520595cc87a5c5f7b601e70d9f9",
    ),
    (
        JUDGMENTS,
        "bea3926ba99a205a2035ae81c8aaa78cc864f6cccf408a071bcccca4553f86aa",
    ),
    (
        SWAP_ROWS,
        "cdd8af93baac3d5cd3a953b0d1d407b96b3f6c2538cae8f23489d331ccf4f571",
    ),
    (
        SWAP_KEYS,
        "866a01878f6c2890a4dbe715a45808df171afe47f7bbc466bf74f6cf578815bd",
    ),
    (
        MARKER_TRAIN_ROWS,
        "c4a7b2f0bb811c08079076bb87879fe040d5e2165ccc4cded73fe54f5f1f30e6",
    ),
    (
        MARKER_TRAIN_KEYS,
        "91a7f5c743dd5ede11e5ce6876229f5692a3cb82cf770ae7d0882a8634d2165d",
    ),
    (
        MARKER_HELD_ROWS,
        "5e6647eadd73a03cb236093270f0096cc02d2611cb2b7abb13dd99bdbf31c0d4",
    ),
    (
        MARKER_HELD_KEYS,
        "c7e67e6c96f297bd1e17614354f9525325895872b82d872fea6999974310acc5",
    ),
    (
        COLON,
        "f490a70c2296e8cc05910f152f36725f2ee132d79a6178e96c826d08de4cdf78",
    ),
    (
        OPENINGS,
        "d9bc9e45c4fc068ac30841ddf8077a46289bc287714eea7181f710a5d11575ee",
    ),
    (
        UORT,
        "3d55850df31c135698a035a731b8f8fce0110f53d31cc75eeee60d583dece485",
    ),
];

pub(crate) struct Inputs {
    root: std::path::PathBuf,
    pub tokenizer: std::path::PathBuf,
    pub coqa: std::path::PathBuf,
    pub pins: BTreeMap<&'static str, &'static str>,
}

#[derive(Default)]
pub(crate) struct Prior {
    pub prefixes: HashSet<Vec<u32>>,
    pub questions: HashSet<String>,
    pub users: HashSet<String>,
    pub coqa_ids: HashSet<String>,
    pub subject_pairs: HashSet<[String; 2]>,
    pub fallback_contrast: HashSet<u32>,
    pub bank: HashSet<String>,
}

impl Prior {
    pub fn allows_text(&self, question: &str, user: &str) -> bool {
        !self.questions.contains(&normalized(question))
            && !self.users.contains(&normalized(user))
            && !self.bank.contains(&prompt_digest(question))
            && !self.bank.contains(&prompt_digest(user))
    }

    pub fn allows_case(&self, case: &Case) -> bool {
        self.allows_text(&case.question, &case.user) && !self.prefixes.contains(&case.prefix)
    }
}

#[derive(Default)]
pub(crate) struct Donor {
    pub raw_vocab: HashSet<u32>,
    pub colon_targets: HashSet<u32>,
    pub immediate: HashSet<u32>,
    pub lexical: HashSet<u32>,
    pub first_words: HashSet<String>,
    pub prefix_digests: HashSet<[u8; 32]>,
    pub colon_rows: usize,
    pub openings: usize,
}

pub(crate) fn prefix_digest(prefix: &[u32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(prefix.len() * 2);
    for &id in prefix {
        bytes.extend_from_slice(&(id as u16).to_le_bytes());
    }
    Sha256::digest(bytes).into()
}

#[derive(Deserialize)]
struct Prefix {
    prefix: Vec<u32>,
}

#[derive(Deserialize)]
struct HistoricalContrast {
    target: u32,
    rival: u32,
    raw_target: u32,
    raw_rival: u32,
}

#[derive(Deserialize)]
struct HistoricalKey {
    user: String,
    question: String,
    source: serde_json::Value,
}

#[derive(Deserialize)]
struct Judgment {
    question: String,
    custody: JudgmentCustody,
}

#[derive(Deserialize)]
struct JudgmentCustody {
    source_conversation_id: Option<String>,
}

#[derive(Deserialize)]
struct Opening {
    answer_start: usize,
    answer_end: usize,
    immediate_id: u32,
    lexical_id: u32,
    lexical_offset: usize,
}

#[derive(Deserialize)]
struct ColonRow {
    prefix: Vec<u32>,
    target: u32,
    offset: usize,
    genuine_assistant: bool,
}

impl Inputs {
    pub fn load(root: &Path) -> Result<Self> {
        for (name, expected) in PINS {
            require(
                sha(&root.join(name))? == expected,
                format!("pinned input changed: {name}"),
            )?;
        }
        Ok(Self {
            root: root.to_owned(),
            tokenizer: root.join(TOKENIZER),
            coqa: root.join(COQA),
            pins: PINS.into_iter().collect(),
        })
    }

    pub fn load_prior(&self) -> Result<Prior> {
        let mut result = Prior::default();
        for row in read_jsonl::<HistoricalContrast>(&self.root.join(MARKER_TRAIN_ROWS))? {
            result
                .fallback_contrast
                .extend([row.target, row.rival, row.raw_target, row.raw_rival]);
        }
        for name in [PROMPTS, SWAP_ROWS, MARKER_TRAIN_ROWS, MARKER_HELD_ROWS] {
            for row in read_jsonl::<Prefix>(&self.root.join(name))? {
                result.prefixes.insert(row.prefix);
            }
        }
        for name in [SWAP_KEYS, MARKER_TRAIN_KEYS, MARKER_HELD_KEYS] {
            for key in read_jsonl::<HistoricalKey>(&self.root.join(name))? {
                result.questions.insert(normalized(&key.question));
                result.users.insert(normalized(&key.user));
                if let Some(id) = key.source["source_conversation_id"].as_str() {
                    result.coqa_ids.insert(id.to_owned());
                }
                if let Some(subjects) = key.source.get("subjects") {
                    let mut pair: [String; 2] = serde_json::from_value(subjects.clone())?;
                    pair.sort_unstable();
                    result.subject_pairs.insert(pair);
                }
            }
        }
        for judgment in read_jsonl::<Judgment>(&self.root.join(JUDGMENTS))? {
            result.questions.insert(normalized(&judgment.question));
            if let Some(id) = judgment.custody.source_conversation_id {
                result.coqa_ids.insert(id);
            }
        }
        require(
            result.coqa_ids.len() == 210,
            "historical CoQA exclusion cardinality changed",
        )?;
        result.bank = BufReader::new(File::open(self.root.join(BANK))?)
            .lines()
            .collect::<std::io::Result<_>>()?;
        require(
            !result.bank.is_empty()
                && result
                    .bank
                    .iter()
                    .all(|s| s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit())),
            "malformed donor prompt digest bank",
        )?;
        Ok(result)
    }

    pub fn load_donor(&self, tokenizer: &ByteBpeTokenizer) -> Result<Donor> {
        let raw = fs::read(self.root.join(UORT))?;
        require(
            raw.starts_with(b"UORT\x01\x00\x00\x00") && raw.len() >= 64 && raw.len() % 2 == 0,
            "invalid pinned donor UORT header",
        )?;
        let count = u64::from_le_bytes(raw[8..16].try_into()?) as usize;
        require((raw.len() - 64) / 2 == count, "wrong donor UORT length")?;
        let tokens: Vec<_> = raw[64..]
            .chunks_exact(2)
            .map(|bytes| u32::from(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect();
        let mut donor = Donor {
            raw_vocab: tokens.iter().copied().collect(),
            ..Donor::default()
        };
        let mut opening_offsets = HashSet::new();
        for opening in read_jsonl::<Opening>(&self.root.join(OPENINGS))? {
            require(
                opening.answer_start <= opening.lexical_offset
                    && opening.lexical_offset < opening.answer_end
                    && tokens.get(opening.answer_start) == Some(&opening.immediate_id)
                    && tokens.get(opening.lexical_offset) == Some(&opening.lexical_id)
                    && opening_offsets.insert(opening.answer_start),
                "opening metadata differs from canonical UORT",
            )?;
            let answer = tokens
                .get(opening.answer_start..opening.answer_end)
                .ok_or("canonical answer span outside UORT")?;
            let answer = tokenizer.decode(answer);
            if let Some(first) = words(&answer).next() {
                donor.first_words.insert(first.to_lowercase());
            }
            donor.immediate.insert(opening.immediate_id);
            donor.lexical.insert(opening.lexical_id);
            donor.openings += 1;
        }
        for line in BufReader::new(File::open(self.root.join(COLON))?).lines() {
            let row: ColonRow = serde_json::from_str(&line?)?;
            require(
                tokens.get(row.offset) == Some(&row.target)
                    && row.prefix.len() <= LIMIT
                    && row.prefix.iter().all(|&id| id < 4096),
                "canonical colon-to-UORT target identity changed",
            )?;
            if row.genuine_assistant {
                require(
                    opening_offsets.remove(&row.offset),
                    "unmatched genuine assistant opening",
                )?;
            }
            donor.colon_targets.insert(row.target);
            donor.prefix_digests.insert(prefix_digest(&row.prefix));
            donor.colon_rows += 1;
        }
        require(
            donor.colon_rows == 118_915 && donor.openings == 4_345 && opening_offsets.is_empty(),
            "canonical donor population changed",
        )?;
        Ok(donor)
    }
}
