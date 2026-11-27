use super::design::{self, EndingFamily, FactFamily};
use super::*;
use serde_json::{json, Value};
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};

const OLD_ROOT: &str = "20261127-codex-copy-id/panel-v1";
const OLD_MANIFEST_SHA: &str = "a6b8962b01e0a9bbb866496ad4f1d95a00770c5e419ca98025039c43ed83d8b0";
const SPLITS: [Split; 3] = [
    Split {
        name: "train",
        families: design::TRAIN,
        facts: design::TRAIN_FACTS,
        labels: design::TRAIN_LABELS,
        count: 32,
        person_start: 0,
    },
    Split {
        name: "held",
        families: design::HELD,
        facts: design::HELD_FACTS,
        labels: design::HELD_LABELS,
        count: 50,
        person_start: 384,
    },
    Split {
        name: "finite",
        families: design::FINITE,
        facts: design::FINITE_FACTS,
        labels: design::FINITE_LABELS,
        count: 8,
        person_start: 784,
    },
];

struct Split {
    name: &'static str,
    families: &'static [EndingFamily],
    facts: &'static [FactFamily],
    labels: &'static str,
    count: usize,
    person_start: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Key {
    pub pair_id: String,
    pub side: u8,
    pub user: String,
    pub context: String,
    pub question: String,
    pub instruction: String,
    pub reply: String,
    pub target_word: String,
    pub distractor_word: String,
    pub target_person: String,
    pub distractor_person: String,
    pub target_fact: String,
    pub fact_family: String,
    pub fact_first: bool,
    pub target_after_subject: bool,
    pub instruction_variant: usize,
    pub stratum: String,
}

#[derive(Default)]
struct OldPanel {
    ids: BTreeSet<String>,
    contexts: BTreeSet<String>,
    bare_questions: BTreeSet<String>,
    query_templates: BTreeSet<String>,
    instruction_templates: BTreeSet<String>,
}

#[derive(Deserialize)]
struct OldKey {
    pair_id: String,
    context: String,
    question: String,
    user: String,
    requested_subject: String,
}
#[derive(Deserialize)]
struct OldRow {
    prefix: Vec<u32>,
}

impl OldPanel {
    fn load(root: &Path, historical: &mut history::Prior) -> Result<Self> {
        let path = root.join(OLD_ROOT);
        require(
            hash(&path.join("SHA256SUMS"))? == OLD_MANIFEST_SHA,
            "#4318 manifest changed",
        )?;
        for line in fs::read_to_string(path.join("SHA256SUMS"))?.lines() {
            let (digest, relative) = line.split_once("  ").ok_or("malformed #4318 manifest")?;
            require(
                hash(&path.join(relative))? == digest,
                format!("changed #4318 file {relative}"),
            )?;
        }
        let mut result = Self::default();
        for split in ["held", "absent", "finite"] {
            let path = path.join(split);
            for row in read_jsonl::<OldRow>(&path.join("rows.jsonl"))? {
                historical.prefixes.insert(row.prefix);
            }
            for key in read_jsonl::<OldKey>(&path.join("keys.jsonl"))? {
                let (query, instruction) = key
                    .question
                    .rsplit_once(" Write ")
                    .ok_or("unrecognized #4318 instruction")?;
                let template = query.replace(&key.requested_subject, "{person}");
                result.ids.insert(key.pair_id);
                result.contexts.insert(normalize(&key.context));
                result.bare_questions.insert(normalize(query));
                result.query_templates.insert(normalize(&template));
                result
                    .instruction_templates
                    .insert(normalize(&format!("Write {instruction}")));
                historical.questions.insert(normalize(&key.question));
                historical.users.insert(normalize(&key.user));
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct Fresh {
    ids: BTreeSet<String>,
    prefixes: BTreeSet<Vec<u32>>,
    questions: BTreeSet<String>,
    contexts: BTreeSet<String>,
    subjects: BTreeSet<String>,
    templates: BTreeSet<String>,
    names: BTreeSet<&'static str>,
    label_ids: BTreeSet<u32>,
}

struct Builder<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    user_head: Vec<u32>,
}

impl<'a> Builder<'a> {
    fn new(tokenizer: &'a ByteBpeTokenizer, protocol: &DialogueProtocol) -> Result<Self> {
        let encoded_marker = [
            tokenizer.encode("\n"),
            tokenizer.encode(protocol.assistant_marker()),
        ]
        .concat();
        require(
            encoded_marker == MARKER,
            "protocol 2 assistant marker changed",
        )?;
        Ok(Self {
            tokenizer,
            encoder: protocol.bind(tokenizer)?,
            user_head: [vec![0], tokenizer.encode("User:")].concat(),
        })
    }

    fn row(&self, key: &Key, target: u32, fact_start: usize, fact_end: usize) -> Result<Row> {
        let user = Message {
            role: "user",
            content: &key.user,
        };
        let open = self.encoder.encode_assistant_prefix(&[user]);
        let closed = self.encoder.encode_document(&[
            Message {
                role: "user",
                content: &key.user,
            },
            Message {
                role: "assistant",
                content: &key.reply,
            },
        ]);
        let expected = if key.side == 0 {
            vec![target, EOS]
        } else {
            vec![target, PERIOD, EOS]
        };
        require(
            open.special_token_occurrences == 0
                && closed.special_token_occurrences == 0
                && open.emitted_turns == 1
                && closed.emitted_turns == 2
                && open.tokens.len() <= LIMIT
                && open.tokens.starts_with(&self.user_head)
                && open.tokens.ends_with(&MARKER)
                && closed.tokens.strip_prefix(open.tokens.as_slice()) == Some(expected.as_slice())
                && closed.response_mask.get(open.tokens.len()..)
                    == Some(vec![1; expected.len()].as_slice())
                && self.tokenizer.decode_bytes(&expected[..expected.len() - 1])
                    == format!(" {}", key.reply).as_bytes(),
            format!("noncanonical open/full response: {}", key.user),
        )?;

        let body_end = open.tokens.len() - MARKER.len();
        let start = self.user_head.len();
        let body = &open.tokens[start..body_end];
        let delimiters = body
            .windows(2)
            .enumerate()
            .filter(|(_, pair)| *pair == [NEWLINE, NEWLINE])
            .map(|(pos, _)| pos)
            .collect::<Vec<_>>();
        let [delimiter] = delimiters.as_slice() else {
            return Err(format!("fact delimiter drift: {}", key.user).into());
        };
        let questions = occurrences(body, QUESTION);
        let [question] = questions.as_slice() else {
            return Err(format!("question delimiter drift: {}", key.user).into());
        };
        require(
            *question > delimiter + 1,
            "question sits outside query scope",
        )?;
        require(
            self.tokenizer.decode_bytes(&body[..*delimiter])
                == format!(" {}", key.context).as_bytes()
                && self
                    .tokenizer
                    .decode_bytes(&body[delimiter + 2..=*question])
                    == key.question.as_bytes()
                && self.tokenizer.decode_bytes(&body[question + 1..])
                    == format!(" {}", key.instruction).as_bytes(),
            format!("wire text changed: {}", key.user),
        )?;

        let source = occurrences(&open.tokens, target);
        let [position] = source.as_slice() else {
            return Err(format!("target occurrence is not unique: {}", key.user).into());
        };
        require(
            (start..start + delimiter).contains(position),
            "target sits outside factual scope",
        )?;
        let offset = self
            .tokenizer
            .decode_bytes(&open.tokens[start..*position])
            .len();
        let piece = self.tokenizer.decode_bytes(&[target]);
        require(
            offset >= fact_start
                && offset + piece.len() <= fact_end + 1
                && piece == format!(" {}", key.target_word).as_bytes(),
            format!("source occurrence belongs to the wrong fact: {}", key.user),
        )?;
        Ok(Row {
            pair_id: key.pair_id.clone(),
            side: key.side,
            prefix: open.tokens,
            gold_pos: (*position).try_into()?,
            target,
            expected,
            family: String::new(),
        })
    }
}

fn occurrences(ids: &[u32], needle: u32) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter_map(|(pos, &id)| (id == needle).then_some(pos))
        .collect()
}

fn render_fact(template: &str, person: &str, word: &str) -> String {
    template.replace("{person}", person).replace("{word}", word)
}

fn labels<'a>(
    list: &'a str,
    tokenizer: &ByteBpeTokenizer,
    fresh: &mut Fresh,
) -> Result<Vec<(&'a str, u32)>> {
    list.split_whitespace()
        .map(|word| {
            let expected = format!(" {word}");
            let ids = tokenizer.encode(&expected);
            let [id] = ids.as_slice() else {
                return Err(format!("non-whole-word label {word}").into());
            };
            require(
                tokenizer.decode_bytes(&[*id]) == expected.as_bytes(),
                "label bytes differ",
            )?;
            require(
                fresh.label_ids.insert(*id),
                format!("label ID {id} reused across splits"),
            )?;
            Ok((word, *id))
        })
        .collect()
}

fn register_templates(split: &Split, old: &OldPanel, fresh: &mut Fresh) -> Result<()> {
    for fact in split.facts {
        require(fresh.names.insert(fact.name), "reused fact family name")?;
        let query = normalize(fact.question);
        require(
            !old.query_templates.contains(&query) && fresh.templates.insert(query),
            "query template overlap",
        )?;
        for template in fact.facts {
            require(
                fresh.templates.insert(normalize(template)),
                "fact template overlap",
            )?;
        }
    }
    for family in split.families {
        require(
            fresh.names.insert(family.name),
            "reused instruction family name",
        )?;
        for sides in family.variants {
            for side in sides {
                let template = normalize(side);
                require(
                    !old.instruction_templates.contains(&template)
                        && fresh.templates.insert(template),
                    format!("instruction template overlap: {side}"),
                )?;
            }
        }
    }
    Ok(())
}

fn build_split(
    split: &Split,
    builder: &Builder<'_>,
    historical: &history::Prior,
    old: &OldPanel,
    fresh: &mut Fresh,
    output: &Path,
) -> Result<Value> {
    register_templates(split, old, fresh)?;
    let labels = labels(split.labels, builder.tokenizer, fresh)?;
    let directory = output.join(split.name);
    fs::create_dir(&directory)?;
    let mut rows = fresh_writer(&directory.join("rows.jsonl"))?;
    let mut keys = fresh_writer(&directory.join("keys.jsonl"))?;
    let mut census = BTreeMap::new();
    let mut token_set = BTreeSet::new();
    let mut vocabulary_by_side = [BTreeSet::new(), BTreeSet::new()];
    let mut minimum = usize::MAX;
    let mut maximum = 0;

    for (family_index, family) in split.families.iter().enumerate() {
        let mut by_label = BTreeMap::new();
        let mut by_fact = BTreeMap::new();
        let mut variants = vec![0; family.variants.len()];
        let mut first = 0;
        let mut before = 0;
        for index in 0..split.count {
            let number = family_index * split.count + index;
            let variant_rotation = if split.name == "held" {
                (index % 10) / 5 + index / 10 + family_index
            } else {
                index / 2 + index / labels.len()
            };
            let variant = variant_rotation % family.variants.len();
            let instructions = family.variants[variant];
            let topic = &split.facts[(index / 2 + family_index * 3) % split.facts.len()];
            let label_index = (index + family_index) % labels.len();
            let (word, target) = labels[label_index];
            let rival_index = if COLLISION.contains(&target) {
                1 - label_index
            } else {
                (label_index + 1 + index / labels.len() % (labels.len() - 1)) % labels.len()
            };
            let (rival_word, _) = labels[rival_index];
            let person = design::person(split.person_start + 2 * number);
            let rival = design::person(split.person_start + 2 * number + 1);
            let fact_first = (index + index / labels.len() + family_index) % 2 == 0;
            let value_first = (index + index / 4 + index / labels.len() + family_index) % 2;
            let target_fact = render_fact(topic.facts[value_first], &person, word);
            let rival_fact = render_fact(
                topic.facts[(index / 4 + family_index) % 2],
                &rival,
                rival_word,
            );
            let (first_fact, second_fact) = if fact_first {
                (&target_fact, &rival_fact)
            } else {
                (&rival_fact, &target_fact)
            };
            let context = format!("{} {first_fact} {second_fact}", topic.background);
            let fact_start =
                topic.background.len() + 1 + if fact_first { 0 } else { first_fact.len() + 1 };
            let fact_end = fact_start + target_fact.len();
            let question = topic.question.replace("{person}", &person);
            let pair_id = format!(
                "{:x}",
                Sha256::digest(format!(
                    "codex-4319-ending-v1\n{}\n{}\n{context}\n{question}",
                    split.name, family.name
                ))
            )[..24]
                .to_owned();
            let stratum = if COLLISION.contains(&target) {
                "collision_1808_3984"
            } else {
                "other"
            };
            require(
                fresh.ids.insert(pair_id.clone())
                    && !old.ids.contains(&pair_id)
                    && fresh.questions.insert(normalize(&question))
                    && !old.bare_questions.contains(&normalize(&question))
                    && fresh.contexts.insert(normalize(&context))
                    && !old.contexts.contains(&normalize(&context))
                    && fresh.subjects.insert(person.clone())
                    && fresh.subjects.insert(rival.clone())
                    && historical.fresh_prompt(&question, &context, &[]),
                format!("duplicate or historical pair {pair_id}"),
            )?;
            let mut pair = Vec::new();
            for (side, instruction) in instructions.iter().enumerate() {
                let user = format!("{context}\n\n{question} {instruction}");
                let reply = if side == 0 {
                    word.to_owned()
                } else {
                    format!("{word}.")
                };
                let key = Key {
                    pair_id: pair_id.clone(),
                    side: side.try_into()?,
                    user,
                    context: context.clone(),
                    question: question.clone(),
                    instruction: (*instruction).to_owned(),
                    reply,
                    target_word: word.to_owned(),
                    distractor_word: rival_word.to_owned(),
                    target_person: person.clone(),
                    distractor_person: rival.clone(),
                    target_fact: target_fact.clone(),
                    fact_family: topic.name.to_owned(),
                    fact_first,
                    target_after_subject: value_first == 0,
                    instruction_variant: variant,
                    stratum: stratum.to_owned(),
                };
                let mut row = builder.row(&key, target, fact_start, fact_end)?;
                row.family = family.name.to_owned();
                require(
                    historical.fresh_prompt(
                        &format!("{question} {instruction}"),
                        &key.user,
                        &row.prefix,
                    ) && fresh.prefixes.insert(row.prefix.clone()),
                    format!("duplicate or historical row {}", key.user),
                )?;
                let instruction_ids = builder.tokenizer.encode(&format!(" {instruction}"));
                token_set.extend(instruction_ids.iter().copied());
                vocabulary_by_side[side].extend(instruction_ids);
                minimum = minimum.min(row.prefix.len());
                maximum = maximum.max(row.prefix.len());
                serde_json::to_writer(&mut rows, &row)?;
                writeln!(rows)?;
                serde_json::to_writer(&mut keys, &key)?;
                writeln!(keys)?;
                pair.push(row);
            }
            let question_position = occurrences(&pair[0].prefix, QUESTION)[0];
            require(
                pair[0].prefix[..=question_position] == pair[1].prefix[..=question_position]
                    && pair[0].prefix != pair[1].prefix
                    && pair[0].gold_pos == pair[1].gold_pos
                    && pair[0].target == pair[1].target,
                format!("pair crosses a non-instruction field: {pair_id}"),
            )?;
            *by_label.entry(format!("{word}:{target}")).or_insert(0) += 1;
            *by_fact.entry(topic.name).or_insert(0) += 1;
            variants[variant] += 1;
            first += usize::from(fact_first);
            before += usize::from(value_first != 0);
        }
        census.insert(
            family.name,
            json!({
                "pairs": split.count, "rows_per_side": split.count, "target_first_fact": first,
                "target_second_fact": split.count-first, "label_before_person": before,
                "label_after_person": split.count-before, "instruction_variants": variants,
                "label_counts": by_label, "fact_family_counts": by_fact,
                "instruction_templates": family.variants,
            }),
        );
    }
    rows.flush()?;
    keys.flush()?;
    Ok(json!({
        "pairs": split.count * split.families.len(), "rows": 2 * split.count * split.families.len(),
        "families": census, "prefix_min": minimum, "prefix_max": maximum,
        "instruction_token_ids": token_set,
        "instruction_token_ids_by_side": vocabulary_by_side,
    }))
}

pub(super) fn run(root: &Path, output: &Path) -> Result<()> {
    let tokenizer = tokenizer(&root.join(history::TOKENIZER))?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let builder = Builder::new(&tokenizer, &protocol)?;
    let history = history::History::load(root)?;
    let mut historical = history.screens(root, &tokenizer)?;
    let old = OldPanel::load(root, &mut historical)?;
    fs::create_dir(output)?;
    let mut fresh = Fresh::default();
    let mut coverage = BTreeMap::new();
    for split in &SPLITS {
        coverage.insert(
            split.name,
            build_split(split, &builder, &historical, &old, &mut fresh, output)?,
        );
    }
    let train_tokens: BTreeSet<u32> =
        serde_json::from_value(coverage["train"]["instruction_token_ids"].clone())?;
    let held_tokens: BTreeSet<u32> =
        serde_json::from_value(coverage["held"]["instruction_token_ids"].clone())?;
    let shared = held_tokens
        .intersection(&train_tokens)
        .copied()
        .collect::<Vec<_>>();
    let exclusive = held_tokens
        .difference(&train_tokens)
        .map(|&id| (id, tokenizer.decode(&[id])))
        .collect::<BTreeMap<_, _>>();
    let receipt = json!({
        "schema": "codex-4319.word-ending-panel/1",
        "vocab_sha256": history::TOKENIZER_SHA256,
        "question_id": QUESTION, "fact_delimiter": [NEWLINE, NEWLINE],
        "assistant_marker": MARKER, "eos_id": EOS, "period_id": PERIOD,
        "collision_target_ids": COLLISION, "coverage": coverage,
        "old_4318_manifest": OLD_MANIFEST_SHA, "old_4318_pair_keys": old.ids.len(),
        "old_4318_templates": {"interrogatives": old.query_templates, "instructions": old.instruction_templates},
        "previous_screen_input_sha256": history.pins,
        "total_unique": {"pair_ids": fresh.ids.len(), "prefixes": fresh.prefixes.len(), "bare_questions": fresh.questions.len(), "contexts": fresh.contexts.len(), "subjects": fresh.subjects.len(), "templates": fresh.templates.len(), "label_ids": fresh.label_ids.len()},
        "shared_train_held_instruction_ids": shared, "held_only_instruction_pieces": exclusive,
    });
    let mut file = fresh_writer(&output.join("receipt.json"))?;
    serde_json::to_writer_pretty(&mut file, &receipt)?;
    writeln!(file)?;
    file.flush()?;
    let mut manifest = fresh_writer(&output.join("SHA256SUMS"))?;
    writeln!(
        manifest,
        "{}  receipt.json",
        hash(&output.join("receipt.json"))?
    )?;
    for split in &SPLITS {
        for name in ["keys.jsonl", "rows.jsonl"] {
            let relative = format!("{}/{name}", split.name);
            writeln!(manifest, "{}  {relative}", hash(&output.join(&relative))?)?;
        }
    }
    manifest.flush()?;
    Ok(())
}
