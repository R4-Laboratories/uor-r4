use super::*;

#[allow(dead_code)]
#[path = "../../../codex-4323/panel/src/design.rs"]
mod scope_design;

const ARCHIVES: [(&str, &str, &[&str]); 4] = [
    (
        "20261127-codex-postperiod",
        "56091ad40d7e5da6f3d6742e02f04e37cdb01171fc3fddb4acba9bcf4f11b876",
        &["finite", "held"],
    ),
    (
        "20261127-codex-source-choice",
        "2b6abf25514e9a744a447659388003994ea5a0d1e92c50dd7934337da0968b7c",
        &["train", "finite", "held-primary", "held-secondary"],
    ),
    (
        "20261127-codex-period-margin",
        "2d10cc6d64a4197385ca83482f4e93b45a888d3ed10dcf7e18f892d7091c240c",
        &["dev", "held", "protect"],
    ),
    (
        "20261127-codex-scope-trace",
        "54781b20590b75f9ef29a7dea59ad1cb0b8f973fbc3fe1b0836c4927e0eccf58",
        &["train", "dev", "held"],
    ),
];

#[derive(Default)]
pub struct Archive {
    pub labels: BTreeSet<u32>,
    pub pair_ids: BTreeSet<String>,
    pub prefixes: BTreeSet<Vec<u32>>,
    pub prompts: BTreeSet<String>,
    pub contexts: BTreeSet<String>,
    pub templates: BTreeSet<String>,
    pub directive_words: BTreeMap<String, BTreeSet<String>>,
    pub roots: BTreeMap<&'static str, &'static str>,
    pub rows: usize,
}

#[derive(Deserialize)]
struct PriorRow {
    pair_id: String,
    side: u8,
    prefix: Vec<u32>,
    family: String,
    ids: Option<[u32; 4]>,
    target: Option<u32>,
    next_id: Option<u32>,
}

#[derive(Deserialize)]
struct PriorKey {
    pair_id: String,
    side: u8,
    context: String,
    question: String,
    directive: String,
    user: String,
    queried_person: Option<String>,
    rival_person: Option<String>,
    opening_word: Option<String>,
    next_word: Option<String>,
    rival_opening: Option<String>,
    rival_next: Option<String>,
}

pub fn normalized(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
}

pub fn template(text: &str) -> String {
    words(text).collect::<Vec<_>>().join(" ")
}

fn masked(text: &str, names: &BTreeSet<String>, labels: &BTreeSet<String>) -> String {
    words(text)
        .map(|word| {
            if word.bytes().any(|byte| byte.is_ascii_digit()) || names.contains(&word) {
                "name".to_owned()
            } else if labels.contains(&word) {
                "word".to_owned()
            } else if [
                "first", "second", "third", "fourth", "one", "two", "three", "four",
            ]
            .contains(&word.as_str())
            {
                "role".to_owned()
            } else {
                word
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl Archive {
    pub fn read(runs: &Path, vocab: &ByteBpeTokenizer) -> Result<Self> {
        let mut archive = Self::default();
        for (run, digest, splits) in ARCHIVES {
            let root = runs.join(run).join("panel-v1");
            require(
                hash(&root.join("SHA256SUMS"))? == digest,
                format!("{run} manifest drift"),
            )?;
            verify_manifest(&root)?;
            archive.roots.insert(run, digest);
            let mut directive_words = BTreeSet::new();
            for split in splits {
                let dir = root.join(split);
                verify_manifest(&dir)?;
                let rows: Vec<PriorRow> = read_jsonl(&dir.join("rows.jsonl"))?;
                let keys: Vec<PriorKey> = read_jsonl(&dir.join("keys.jsonl"))?;
                require(
                    rows.len() == keys.len(),
                    format!("{run}/{split} key count drift"),
                )?;
                archive.rows += rows.len();
                for (row, key) in rows.into_iter().zip(keys) {
                    require(
                        row.pair_id == key.pair_id && row.side == key.side,
                        format!("{run}/{split} key join drift"),
                    )?;
                    let labels = [
                        key.opening_word,
                        key.next_word,
                        key.rival_opening,
                        key.rival_next,
                    ];
                    let mut label_ids: BTreeSet<u32> = row
                        .ids
                        .into_iter()
                        .flatten()
                        .chain(row.target)
                        .chain(row.next_id)
                        .collect();
                    let mut label_words = BTreeSet::new();
                    for word in labels.iter().flatten() {
                        label_words.extend(words(word));
                        label_ids.extend(vocab.encode(&format!(" {word}")));
                    }
                    for &id in &label_ids {
                        label_words
                            .extend(words(&String::from_utf8_lossy(&vocab.decode_bytes(&[id]))));
                    }
                    let names = [key.queried_person, key.rival_person]
                        .iter()
                        .flatten()
                        .flat_map(|name| words(name))
                        .collect();
                    archive.labels.extend(label_ids);
                    archive.pair_ids.insert(row.pair_id);
                    archive.prefixes.insert(row.prefix);
                    archive.prompts.insert(normalized(&key.user));
                    archive.contexts.insert(normalized(&key.context));
                    archive.templates.extend([
                        template(&key.directive),
                        template(&row.family),
                        masked(&key.question, &names, &BTreeSet::new()),
                    ]);
                    for fact in key
                        .context
                        .split(['.', ';'])
                        .filter(|fact| !fact.trim().is_empty())
                    {
                        archive.templates.insert(masked(fact, &names, &label_words));
                    }
                    directive_words.extend(words(&key.directive));
                }
            }
            archive
                .directive_words
                .insert(run.to_owned(), directive_words);
        }
        for split in scope_design::SPLITS {
            for family in split.families {
                archive
                    .templates
                    .extend(family.counts.into_iter().map(template));
                if let Some(selectors) = family.selectors {
                    archive
                        .templates
                        .extend(selectors.into_iter().map(template));
                }
            }
        }
        Ok(archive)
    }

    pub fn check_template(&self, text: &str) -> Result<()> {
        require(
            !self.templates.contains(&template(text)),
            format!("overlap with archived authoring template: {text}"),
        )
    }
}
