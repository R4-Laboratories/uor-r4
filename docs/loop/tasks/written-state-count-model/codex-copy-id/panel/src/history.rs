use super::*;

pub(crate) const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
pub(crate) const TOKENIZER_SHA256: &str =
    "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const BANK: &str = "20261127-codex-visible-answer/panel/inputs/d87-bank-prompt-digests.txt";
const OLD_ROWS: &str = "20261127-codex-visible-answer/panel/frozen-v1/prompts.jsonl";
const OLD_QUESTIONS: &str = "20261127-codex-visible-answer/panel/frozen-v1/judgments.jsonl";
const SWAP: &str = "20261127-codex-question-swap/panel/frozen-v2";
const MARKER_TRAIN: &str = "20261127-codex-marker-question/panel-v2/train";
const MARKER_HELD: &str = "20261127-codex-marker-question/panel-v2/held";
const OPENINGS: &str = "20261127-codex-marker-question/donor-v2/openings.jsonl";
const COLONS: &str = "20261127-codex-marker-question/donor-v2/colon-targets.jsonl";
const UORT: &str = "20261127-codex-marker-question/donor-v2/answer-first.tokens.u16";
const PRIOR_ROOT: &str = "20261127-codex-shared-source/panel-v1";
const PRIOR_SPLITS: [&str; 5] = ["train", "held", "human", "absent", "train-absent"];
const PINS: [(&str, &str); 8] = [
    (TOKENIZER, TOKENIZER_SHA256),
    (
        BANK,
        "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541",
    ),
    (
        OLD_ROWS,
        "021f4c1fe8e10884ecaa3cb3b67ed290356fb520595cc87a5c5f7b601e70d9f9",
    ),
    (
        OLD_QUESTIONS,
        "bea3926ba99a205a2035ae81c8aaa78cc864f6cccf408a071bcccca4553f86aa",
    ),
    (
        OPENINGS,
        "d9bc9e45c4fc068ac30841ddf8077a46289bc287714eea7181f710a5d11575ee",
    ),
    (
        COLONS,
        "f490a70c2296e8cc05910f152f36725f2ee132d79a6178e96c826d08de4cdf78",
    ),
    (
        UORT,
        "3d55850df31c135698a035a731b8f8fce0110f53d31cc75eeee60d583dece485",
    ),
    (
        "20261127-codex-shared-source/panel-v1/SHA256SUMS",
        "5a405952795ed13cff8d7737ee2027f021664252bfeed1d7e7f52f4213602dd8",
    ),
];
const PREVIOUS_PANELS: [(&str, &str, &str); 3] = [
    (
        SWAP,
        "cdd8af93baac3d5cd3a953b0d1d407b96b3f6c2538cae8f23489d331ccf4f571",
        "866a01878f6c2890a4dbe715a45808df171afe47f7bbc466bf74f6cf578815bd",
    ),
    (
        MARKER_TRAIN,
        "c4a7b2f0bb811c08079076bb87879fe040d5e2165ccc4cded73fe54f5f1f30e6",
        "91a7f5c743dd5ede11e5ce6876229f5692a3cb82cf770ae7d0882a8634d2165d",
    ),
    (
        MARKER_HELD,
        "5e6647eadd73a03cb236093270f0096cc02d2611cb2b7abb13dd99bdbf31c0d4",
        "c7e67e6c96f297bd1e17614354f9525325895872b82d872fea6999974310acc5",
    ),
];

pub(crate) struct History {
    pub pins: BTreeMap<String, String>,
}

#[derive(Default)]
pub(crate) struct Prior {
    pub prefixes: BTreeSet<Vec<u32>>,
    pub questions: BTreeSet<String>,
    pub users: BTreeSet<String>,
    pub subject_pairs: BTreeSet<[String; 2]>,
    pub train_ids: BTreeSet<u32>,
    pub held_ids: BTreeSet<u32>,
    pub marker_train_ids: BTreeSet<u32>,
    pub donor_opening_ids: BTreeSet<u32>,
    pub donor_opening_words: BTreeSet<String>,
    pub donor_raw_vocabulary: BTreeSet<u32>,
    pub donor_colon_ids: BTreeSet<u32>,
    pub donor_prefixes: BTreeSet<[u8; 32]>,
    pub bank: BTreeSet<String>,
}

impl Prior {
    pub fn fresh_label(&self, id: u32, word: &str) -> bool {
        !COLLISION.contains(&id)
            && !self.train_ids.contains(&id)
            && !self.held_ids.contains(&id)
            && !self.marker_train_ids.contains(&id)
            && !self.donor_opening_ids.contains(&id)
            && !self.donor_opening_words.contains(word)
    }

    pub fn fresh_prompt(&self, question: &str, user: &str, prefix: &[u32]) -> bool {
        let question = normalize(question);
        let user = normalize(user);
        !self.questions.contains(&question)
            && !self.users.contains(&user)
            && !self.bank.contains(&bank_digest(&question))
            && !self.bank.contains(&bank_digest(&user))
            && !self.prefixes.contains(prefix)
            && !self.donor_prefixes.contains(&prefix_hash(prefix))
    }
}

#[derive(Deserialize)]
struct HistoricalRow {
    prefix: Vec<u32>,
    target: Option<u32>,
    rival: Option<u32>,
    raw_target: Option<u32>,
    raw_rival: Option<u32>,
}

#[derive(Deserialize)]
struct HistoricalKey {
    user: Option<String>,
    question: String,
    source: Option<serde_json::Value>,
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
struct Colon {
    prefix: Vec<u32>,
    target: u32,
    offset: usize,
    genuine_assistant: bool,
}

fn bank_digest(normalized: &str) -> String {
    hex::encode(&Sha256::digest(normalized)[..8])
}

impl History {
    pub fn load(root: &Path) -> Result<Self> {
        let mut pins = BTreeMap::new();
        for (path, expected) in PINS {
            pins.insert(path.to_owned(), expected.to_owned());
        }
        for (base, rows, keys) in PREVIOUS_PANELS {
            pins.insert(format!("{base}/rows.jsonl"), rows.to_owned());
            pins.insert(format!("{base}/keys.jsonl"), keys.to_owned());
        }
        for (path, digest) in &pins {
            require(
                hash(&root.join(path))? == *digest,
                format!("input digest changed: {path}"),
            )?;
        }
        let old_manifest = fs::read_to_string(root.join(PRIOR_ROOT).join("SHA256SUMS"))?;
        for line in old_manifest.lines() {
            let (digest, file) = line.split_once("  ").ok_or("malformed #4315 manifest")?;
            let path = format!("{PRIOR_ROOT}/{file}");
            require(
                hash(&root.join(&path))? == digest,
                format!("input digest changed: {path}"),
            )?;
            pins.insert(path, digest.to_owned());
        }
        Ok(Self { pins })
    }

    pub fn screens(&self, root: &Path, tokenizer: &ByteBpeTokenizer) -> Result<Prior> {
        let mut prior = Prior {
            bank: BufReader::new(File::open(root.join(BANK))?)
                .lines()
                .collect::<io::Result<_>>()?,
            ..Prior::default()
        };
        for row in read_jsonl::<HistoricalRow>(&root.join(OLD_ROWS))? {
            prior.prefixes.insert(row.prefix);
        }
        for key in read_jsonl::<HistoricalKey>(&root.join(OLD_QUESTIONS))? {
            prior.questions.insert(normalize(&key.question));
        }
        for (base, _, _) in PREVIOUS_PANELS {
            Self::add_panel(root, base, &mut prior, base == MARKER_TRAIN, false, false)?;
        }
        for split in PRIOR_SPLITS {
            Self::add_panel(
                root,
                &format!("{PRIOR_ROOT}/{split}"),
                &mut prior,
                false,
                split == "train",
                split == "held" || split == "human",
            )?;
        }
        let raw = fs::read(root.join(UORT))?;
        require(
            raw.len() >= 64 && raw.len() % 2 == 0 && raw.starts_with(b"UORT\x01\x00\x00\x00"),
            "invalid donor token header",
        )?;
        let tokens = raw[64..]
            .chunks_exact(2)
            .map(|pair| u32::from(u16::from_le_bytes([pair[0], pair[1]])))
            .collect::<Vec<_>>();
        require(
            u64::from_le_bytes(raw[8..16].try_into()?) == tokens.len() as u64,
            "donor token count differs",
        )?;
        prior.donor_raw_vocabulary.extend(tokens.iter().copied());
        let mut openings = BTreeSet::new();
        for row in read_jsonl::<Opening>(&root.join(OPENINGS))? {
            require(
                row.answer_start <= row.lexical_offset
                    && row.lexical_offset < row.answer_end
                    && tokens.get(row.answer_start) == Some(&row.immediate_id)
                    && tokens.get(row.lexical_offset) == Some(&row.lexical_id)
                    && openings.insert(row.answer_start),
                "donor opening span differs",
            )?;
            let answer = tokenizer.decode(
                tokens
                    .get(row.answer_start..row.answer_end)
                    .ok_or("donor answer out of bounds")?,
            );
            if let Some(word) = answer
                .split(|ch: char| !ch.is_alphanumeric())
                .find(|part| !part.is_empty())
            {
                prior.donor_opening_words.insert(word.to_lowercase());
            }
            prior
                .donor_opening_ids
                .extend([row.immediate_id, row.lexical_id]);
        }
        let mut colon_count = 0;
        for line in BufReader::new(File::open(root.join(COLONS))?).lines() {
            let row: Colon = serde_json::from_str(&line?)?;
            require(
                row.prefix.len() <= LIMIT && tokens.get(row.offset) == Some(&row.target),
                "donor colon target differs",
            )?;
            if row.genuine_assistant {
                require(
                    openings.remove(&row.offset),
                    "donor colon missing its opening",
                )?;
            }
            prior.donor_colon_ids.insert(row.target);
            prior.donor_prefixes.insert(prefix_hash(&row.prefix));
            colon_count += 1;
        }
        require(
            colon_count == 118_915 && openings.is_empty(),
            "donor population differs",
        )?;
        Ok(prior)
    }

    fn add_panel(
        root: &Path,
        base: &str,
        prior: &mut Prior,
        marker_train: bool,
        train: bool,
        held: bool,
    ) -> Result<()> {
        for row in read_jsonl::<HistoricalRow>(&root.join(base).join("rows.jsonl"))? {
            prior.prefixes.insert(row.prefix);
            let ids = [row.target, row.rival, row.raw_target, row.raw_rival]
                .into_iter()
                .flatten();
            if marker_train {
                prior.marker_train_ids.extend(ids.clone());
            }
            if train {
                prior.train_ids.extend(ids.clone());
            }
            if held {
                prior.held_ids.extend(ids);
            }
        }
        for key in read_jsonl::<HistoricalKey>(&root.join(base).join("keys.jsonl"))? {
            prior.questions.insert(normalize(&key.question));
            if let Some(user) = key.user {
                prior.users.insert(normalize(&user));
            }
            if let Some(source) = key.source {
                if let Some(subjects) = source.get("subjects") {
                    let mut pair: [String; 2] = serde_json::from_value(subjects.clone())?;
                    pair.sort_unstable();
                    prior.subject_pairs.insert(pair);
                }
            }
        }
        Ok(())
    }
}
