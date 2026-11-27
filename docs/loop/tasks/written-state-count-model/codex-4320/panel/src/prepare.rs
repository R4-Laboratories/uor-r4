use super::design::{self, Family, Topic};
use super::*;
use serde_json::{json, Value};
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};

const PARENT: &str = "20261127-codex-word-ending/panel-v1";
const COPY: &str = "20261127-codex-copy-id/panel-v1";
const EARLIER: [(&str, &str, &[&str]); 2] = [
    (
        PARENT,
        "7df4af3f38ef7775caaf02326cee8dd8fa14de9ba043f23cee038cdd61f25fc0",
        &["train", "held", "finite"],
    ),
    (
        COPY,
        "a6b8962b01e0a9bbb866496ad4f1d95a00770c5e419ca98025039c43ed83d8b0",
        &["held", "absent", "finite"],
    ),
];

#[derive(Serialize, Deserialize)]
pub(super) struct Key {
    pub pair_id: String,
    pub side: u8,
    pub user: String,
    pub question: String,
    pub directive: String,
    pub context: String,
    pub reply: String,
    pub queried_person: String,
    pub rival_person: String,
    pub authorized_fact: String,
    pub rival_fact: String,
    pub opening_word: String,
    pub next_word: String,
    pub rival_opening: String,
    pub rival_next: String,
    pub fact_topic: String,
    pub fact_first: bool,
    pub opening_before_next: bool,
    pub rival_positions: [u32; 2],
    pub instruction_variant: usize,
    pub instruction_final_period: bool,
    pub opening_collision: bool,
    pub next_collision: bool,
}

#[derive(Default)]
struct PreviouslySeen {
    ids: BTreeSet<String>,
    contexts: BTreeSet<String>,
    facts: BTreeSet<String>,
    questions: BTreeSet<String>,
    templates: BTreeSet<String>,
    subjects: BTreeSet<String>,
    label_ids: BTreeSet<u32>,
    source_pins: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct OldRow {
    prefix: Vec<u32>,
    target: Option<u32>,
    rival: Option<u32>,
    family: Option<String>,
}

#[derive(Deserialize)]
struct OldKey {
    pair_id: String,
    context: String,
    user: String,
    question: String,
    target_person: Option<String>,
    distractor_person: Option<String>,
    requested_subject: Option<String>,
    subjects: Option<Vec<String>>,
    target_word: Option<String>,
    distractor_word: Option<String>,
    values_by_subject: Option<Vec<String>>,
}

impl PreviouslySeen {
    fn load(root: &Path, history: &mut history::Prior, vocab: &ByteBpeTokenizer) -> Result<Self> {
        let mut earlier = Self::default();
        for (name, expected, splits) in EARLIER {
            let path = root.join(name);
            require(
                hash(&path.join("SHA256SUMS"))? == expected,
                "prior manifest changed",
            )?;
            earlier
                .source_pins
                .insert(format!("{name}/SHA256SUMS"), expected.to_owned());
            verify_manifest(&path, &path.join("SHA256SUMS"))?;
            for split in splits {
                let dir = path.join(split);
                for row in read_jsonl::<OldRow>(&dir.join("rows.jsonl"))? {
                    history.prefixes.insert(row.prefix);
                    earlier
                        .label_ids
                        .extend([row.target, row.rival].into_iter().flatten());
                    if let Some(family) = row.family {
                        earlier.templates.insert(normalize(&family));
                    }
                }
                for key in read_jsonl::<OldKey>(&dir.join("keys.jsonl"))? {
                    let query_directive = if name == COPY {
                        key.question.as_str()
                    } else {
                        key.user
                            .split_once("\n\n")
                            .ok_or("old prompt lacks fact delimiter")?
                            .1
                    };
                    let (query, directive) = if name == COPY {
                        let (query, directive) = query_directive
                            .rsplit_once(" Write ")
                            .ok_or("copy panel prompt lacks directive")?;
                        (query.to_owned(), format!("Write {directive}"))
                    } else {
                        let (query, directive) = query_directive
                            .split_once('?')
                            .ok_or("word-ending panel prompt lacks question")?;
                        (format!("{query}?"), directive.to_owned())
                    };
                    let requested = key
                        .target_person
                        .as_ref()
                        .or(key.requested_subject.as_ref())
                        .ok_or("old prompt lacks requested person")?;
                    let subjects = key.subjects.unwrap_or_else(|| {
                        [key.target_person.clone(), key.distractor_person]
                            .into_iter()
                            .flatten()
                            .collect()
                    });
                    let values = key.values_by_subject.unwrap_or_else(|| {
                        [key.target_word, key.distractor_word]
                            .into_iter()
                            .flatten()
                            .collect()
                    });
                    earlier.ids.insert(key.pair_id);
                    earlier.contexts.insert(normalize(&key.context));
                    earlier.questions.insert(normalize(&query));
                    earlier.templates.insert(normalize(&directive));
                    earlier
                        .templates
                        .insert(normalize(&query.replace(requested, "{person}")));
                    for sentence in key.context.split_inclusive('.') {
                        earlier.facts.insert(normalize(sentence));
                        if values.len() == 2 && subjects.len() == 2 {
                            let mut template = sentence.to_owned();
                            for subject in &subjects {
                                template = template.replace(subject, "{person}");
                            }
                            for value in &values {
                                template = template.replace(&format!(" {value}"), " {word}");
                            }
                            earlier.templates.insert(normalize(&template));
                        }
                    }
                    for word in values {
                        if let [id] = vocab.encode(&format!(" {word}")).as_slice() {
                            earlier.label_ids.insert(*id);
                        }
                    }
                    earlier.subjects.extend(subjects);
                    history.questions.insert(normalize(query_directive));
                    history.users.insert(normalize(&key.user));
                }
            }
        }
        for index in &history.subject_pairs {
            earlier.subjects.extend(index.iter().cloned());
        }
        Ok(earlier)
    }
}

#[derive(Default)]
struct Fresh {
    ids: BTreeSet<String>,
    prefixes: BTreeSet<Vec<u32>>,
    questions: BTreeSet<String>,
    contexts: BTreeSet<String>,
    facts: BTreeSet<String>,
    people: BTreeSet<String>,
    templates: BTreeSet<String>,
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

    fn row(
        &self,
        key: &mut Key,
        family: &str,
        labels: [Label<'_>; 4],
        offsets: [usize; 4],
    ) -> Result<Row> {
        let open = self.encoder.encode_assistant_prefix(&[Message {
            role: "user",
            content: &key.user,
        }]);
        let complete = self.encoder.encode_document(&[
            Message {
                role: "user",
                content: &key.user,
            },
            Message {
                role: "assistant",
                content: &key.reply,
            },
        ]);
        let mut expected = vec![labels[0].id, PERIOD];
        if key.side == 1 {
            expected.push(labels[1].id);
        }
        expected.push(EOS);
        require(
            open.special_token_occurrences == 0
                && complete.special_token_occurrences == 0
                && open.emitted_turns == 1
                && complete.emitted_turns == 2
                && open.tokens.len() + 4 <= LIMIT
                && !open.tokens.contains(&EOS)
                && open.tokens.starts_with(&self.user_head)
                && open.tokens.ends_with(&MARKER)
                && complete.tokens.strip_prefix(open.tokens.as_slice())
                    == Some(expected.as_slice())
                && complete.response_mask.get(open.tokens.len()..)
                    == Some(vec![1; expected.len()].as_slice())
                && self.tokenizer.decode_bytes(&expected[..expected.len() - 1])
                    == format!(" {}", key.reply).as_bytes(),
            format!(
                "unrecognized open/complete dialogue for pair {}",
                key.pair_id
            ),
        )?;

        let start = self.user_head.len();
        let end = open.tokens.len() - MARKER.len();
        let body = &open.tokens[start..end];
        let delimiters = body
            .windows(2)
            .enumerate()
            .filter_map(|(pos, pair)| (pair == [NEWLINE, NEWLINE]).then_some(pos))
            .collect::<Vec<_>>();
        let [delimiter] = delimiters.as_slice() else {
            return Err(format!("fact delimiter drift: {}", key.pair_id).into());
        };
        let questions = occurrences(body, QUESTION);
        let [question] = questions.as_slice() else {
            return Err(format!("question delimiter drift: {}", key.pair_id).into());
        };
        require(*question > delimiter + 1, "question outside query scope")?;
        let user_bytes = format!(" {}", key.user);
        let (ordinary_ids, fallback_count) = self.tokenizer.encode_lossy(&user_bytes);
        require(
            fallback_count == 0
                && ordinary_ids == body
                && self.tokenizer.decode_bytes(body) == user_bytes.as_bytes()
                && self.tokenizer.decode_bytes(&body[..*delimiter])
                    == format!(" {}", key.context).as_bytes()
                && self
                    .tokenizer
                    .decode_bytes(&body[delimiter + 2..=*question])
                    == key.question.as_bytes()
                && self.tokenizer.decode_bytes(&body[question + 1..])
                    == format!(" {}", key.directive).as_bytes()
                && (body.last() == Some(&PERIOD)) == key.instruction_final_period,
            format!("wire text drift or fallback: {}", key.pair_id),
        )?;
        let mut positions = [0; 4];
        for (slot, (label, offset)) in labels.into_iter().zip(offsets).enumerate() {
            let found = occurrences(&open.tokens, label.id);
            let [position] = found.as_slice() else {
                return Err(format!(
                    "source token {} occurs {} times in pair {}: {}",
                    label.word,
                    found.len(),
                    key.pair_id,
                    key.user
                )
                .into());
            };
            require(
                (start..start + delimiter).contains(position)
                    && self
                        .tokenizer
                        .decode_bytes(&open.tokens[start..*position])
                        .len()
                        == offset
                    && self.tokenizer.decode_bytes(&[label.id])
                        == format!(" {}", label.word).as_bytes(),
                format!(
                    "source address belongs to the wrong fact for pair {}",
                    key.pair_id
                ),
            )?;
            positions[slot] = (*position).try_into()?;
        }
        key.rival_positions = [positions[2], positions[3]];
        Ok(Row {
            pair_id: key.pair_id.clone(),
            side: key.side,
            family: family.to_owned(),
            prefix: open.tokens,
            gold_pos: positions[0],
            target: labels[0].id,
            next_pos: positions[1],
            next_id: labels[1].id,
            expected,
        })
    }
}

#[derive(Clone, Copy)]
struct Label<'a> {
    word: &'a str,
    id: u32,
}

fn labels(
    builder: &Builder<'_>,
    prior: &history::Prior,
    old: &PreviouslySeen,
) -> Result<Vec<Label<'static>>> {
    let mut result = Vec::new();
    for word in "lake anchor"
        .split_whitespace()
        .chain(design::WORDS.split_whitespace())
    {
        let piece = format!(" {word}");
        let encoded = builder.tokenizer.encode(&piece);
        let [id] = encoded.as_slice() else {
            continue;
        };
        let id = *id;
        if old.label_ids.contains(&id) || !prior.fresh_label(id, word) {
            if !COLLISION.contains(&id) {
                continue;
            }
        }
        require(
            builder.tokenizer.decode_bytes(&[id]) == piece.as_bytes(),
            "label round trip differs",
        )?;
        result.push(Label { word, id });
    }
    require(
        result.len() >= 18 && result[..2].iter().map(|label| label.id).eq(COLLISION),
        format!("insufficient fresh original whole words: {}", result.len()),
    )?;
    Ok(result)
}

fn occurrences(ids: &[u32], needle: u32) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter_map(|(pos, &id)| (id == needle).then_some(pos))
        .collect()
}

fn render_fact(
    template: &str,
    name: &str,
    opening: Label<'_>,
    next: Label<'_>,
) -> Result<(String, [usize; 2])> {
    let fact = template
        .replace("{person}", name)
        .replace("{first}", opening.word)
        .replace("{next}", next.word);
    let locations = |Label { word, .. }: Label<'_>| {
        fact.match_indices(&format!(" {word}"))
            .filter_map(|(index, _)| {
                matches!(
                    fact.as_bytes().get(index + 1 + word.len()),
                    Some(b' ' | b'.')
                )
                .then_some(index)
            })
            .collect::<Vec<_>>()
    };
    let first = locations(opening);
    let next = locations(next);
    let ([first], [next]) = (first.as_slice(), next.as_slice()) else {
        return Err("fact must contain each source as a unique standalone value".into());
    };
    Ok((fact, [*first, *next]))
}

fn register(template: &str, old: &PreviouslySeen, fresh: &mut Fresh) -> Result<()> {
    let normalized = normalize(template);
    require(
        !old.templates.contains(&normalized) && fresh.templates.insert(normalized),
        format!("reused wording or facts template: {template}"),
    )
}

struct Split<'a> {
    name: &'a str,
    families: &'a [Family],
    topics: &'a [Topic],
    pool: &'a [Label<'a>],
    pairs_per_family: usize,
    person_start: usize,
}

#[derive(Default, Serialize)]
struct Census {
    pairs: usize,
    queried_fact_first: usize,
    queried_fact_second: usize,
    opening_before_next: usize,
    next_before_opening: usize,
    variant_pairs: [usize; 2],
    final_period_pairs: usize,
    unpunctuated_pairs: usize,
    roles_by_id: BTreeMap<u32, [usize; 4]>,
    topic_pairs: BTreeMap<String, usize>,
    gold_positions: BTreeMap<u32, usize>,
    next_positions: BTreeMap<u32, usize>,
    opening_collisions: usize,
    next_collisions: usize,
}

fn build_split(
    split: Split<'_>,
    builder: &Builder<'_>,
    history: &history::Prior,
    old: &PreviouslySeen,
    fresh: &mut Fresh,
    output: &Path,
) -> Result<Value> {
    for topic in split.topics {
        register(topic.name, old, fresh)?;
        register(topic.question, old, fresh)?;
        for fact in topic.facts {
            register(fact, old, fresh)?;
        }
    }
    for label in split.pool {
        require(
            fresh.label_ids.insert(label.id),
            "finite and held share a raw value",
        )?;
    }
    let directory = output.join(split.name);
    fs::create_dir(&directory)?;
    let mut rows = fresh_writer(&directory.join("rows.jsonl"))?;
    let mut keys = fresh_writer(&directory.join("keys.jsonl"))?;
    let mut coverage = BTreeMap::new();
    let mut min_prefix = usize::MAX;
    let mut max_prefix = 0;
    for (family_no, family) in split.families.iter().enumerate() {
        register(family.name, old, fresh)?;
        for wordings in family.variants {
            for wording in wordings {
                register(wording, old, fresh)?;
            }
        }

        let mut census = Census::default();
        let width = split.pool.len();
        for index in 0..split.pairs_per_family {
            let pair_no = family_no * split.pairs_per_family + index;
            let person = design::name(split.person_start + 2 * pair_no);
            let rival = design::name(split.person_start + 2 * pair_no + 1);
            let label = (index + family_no * 2) % width;
            let shift = 2 + (index / width + family_no) % (width / 2 - 1) * 2;
            let indices = [
                label,
                (label + shift) % width,
                label ^ 1,
                ((label ^ 1) + shift) % width,
            ];
            let labels = indices.map(|i| split.pool[i]);
            let [opening, next, rival_opening, rival_next] = labels;
            let fact_first = (index + index / width + family_no) % 2 == 0;
            let variant = (index / (width / 2) + index / width + family_no) % 2;
            let instruction_final_period =
                (index / 2 + index % 2 + index / width + family_no) % 2 == 0;
            let topic = &split.topics[(index + index / 4 + family_no) % split.topics.len()];
            let (authorized_fact, offsets) = render_fact(
                topic.facts[(index + index / 2 + family_no) % 2],
                &person,
                opening,
                next,
            )?;
            let (rival_fact, rival_offsets) = render_fact(
                topic.facts[(index / 2 + family_no) % 2],
                &rival,
                rival_opening,
                rival_next,
            )?;
            let (context, asked_start, rival_start) = if fact_first {
                (
                    format!("{authorized_fact} {rival_fact}"),
                    1,
                    authorized_fact.len() + 2,
                )
            } else {
                (
                    format!("{rival_fact} {authorized_fact}"),
                    rival_fact.len() + 2,
                    1,
                )
            };
            let absolute_offsets = [
                offsets[0] + asked_start,
                offsets[1] + asked_start,
                rival_offsets[0] + rival_start,
                rival_offsets[1] + rival_start,
            ];
            let question = topic.question.replace("{person}", &person);
            let pair_id = format!(
                "{:x}",
                Sha256::digest(format!(
                    "codex-4320-postperiod-v1\n{}\n{}\n{context}\n{question}",
                    split.name, family.name
                ))
            )[..24]
                .to_owned();
            require(
                fresh.ids.insert(pair_id.clone())
                    && !old.ids.contains(&pair_id)
                    && fresh.contexts.insert(normalize(&context))
                    && !old.contexts.contains(&normalize(&context))
                    && fresh.questions.insert(normalize(&question))
                    && !old.questions.contains(&normalize(&question))
                    && history.fresh_prompt(&question, &context, &[]),
                format!("overlapping pair: {pair_id}"),
            )?;
            for name in [&person, &rival] {
                require(
                    fresh.people.insert(name.clone()) && !old.subjects.contains(name.as_str()),
                    format!("reused person: {name}"),
                )?;
            }
            for fact in [&authorized_fact, &rival_fact] {
                let norm = normalize(fact);
                require(
                    !old.facts.contains(&norm) && fresh.facts.insert(norm),
                    format!("reused fact: {fact}"),
                )?;
            }
            let mut pair = Vec::new();
            for (side, directive) in family.variants[variant].into_iter().enumerate() {
                let directive = if instruction_final_period {
                    directive
                } else {
                    directive
                        .strip_suffix('.')
                        .ok_or("base directive must have a final period")?
                };
                let mut key = Key {
                    pair_id: pair_id.clone(),
                    side: side.try_into()?,
                    user: format!("{context}\n\n{question} {directive}"),
                    question: question.clone(),
                    directive: directive.to_owned(),
                    context: context.clone(),
                    reply: if side == 0 {
                        format!("{}.", opening.word)
                    } else {
                        format!("{}. {}", opening.word, next.word)
                    },
                    queried_person: person.clone(),
                    rival_person: rival.clone(),
                    authorized_fact: authorized_fact.clone(),
                    rival_fact: rival_fact.clone(),
                    opening_word: opening.word.to_owned(),
                    next_word: next.word.to_owned(),
                    rival_opening: rival_opening.word.to_owned(),
                    rival_next: rival_next.word.to_owned(),
                    fact_topic: topic.name.to_owned(),
                    fact_first,
                    opening_before_next: offsets[0] < offsets[1],
                    rival_positions: [0; 2],
                    instruction_variant: variant,
                    instruction_final_period,
                    opening_collision: COLLISION.contains(&opening.id),
                    next_collision: COLLISION.contains(&next.id),
                };
                let row = builder.row(&mut key, family.name, labels, absolute_offsets)?;
                require(
                    history.fresh_prompt(
                        &format!("{question} {directive}"),
                        &key.user,
                        &row.prefix,
                    ) && fresh.prefixes.insert(row.prefix.clone()),
                    format!("overlapping full prompt: {pair_id}"),
                )?;
                min_prefix = min_prefix.min(row.prefix.len());
                max_prefix = max_prefix.max(row.prefix.len());
                serde_json::to_writer(&mut rows, &row)?;
                writeln!(rows)?;
                serde_json::to_writer(&mut keys, &key)?;
                writeln!(keys)?;
                pair.push(row);
            }
            let [stop, cont] = pair.as_slice() else {
                return Err("missing matched side".into());
            };
            let question_pos = stop
                .prefix
                .iter()
                .position(|id| *id == QUESTION)
                .ok_or("question vanished")?;
            require(
                stop.prefix[..=question_pos] == cont.prefix[..=question_pos]
                    && stop.prefix != cont.prefix
                    && stop.gold_pos == cont.gold_pos
                    && stop.next_pos == cont.next_pos
                    && stop.target == cont.target
                    && stop.next_id == cont.next_id,
                format!("only the final directive may vary: {pair_id}"),
            )?;
            census.pairs += 1;
            census.queried_fact_first += usize::from(fact_first);
            census.queried_fact_second += usize::from(!fact_first);
            census.opening_before_next += usize::from(stop.gold_pos < stop.next_pos);
            census.next_before_opening += usize::from(stop.next_pos < stop.gold_pos);
            census.variant_pairs[variant] += 1;
            census.final_period_pairs += usize::from(instruction_final_period);
            census.unpunctuated_pairs += usize::from(!instruction_final_period);
            census.opening_collisions += usize::from(COLLISION.contains(&stop.target));
            census.next_collisions += usize::from(COLLISION.contains(&stop.next_id));
            for (slot, label) in labels.iter().enumerate() {
                census.roles_by_id.entry(label.id).or_default()[slot] += 1;
            }
            *census.topic_pairs.entry(topic.name.to_owned()).or_default() += 1;
            *census.gold_positions.entry(stop.gold_pos).or_default() += 1;
            *census.next_positions.entry(stop.next_pos).or_default() += 1;
        }
        require(
            census.queried_fact_first == census.queried_fact_second
                && census.variant_pairs[0] == census.variant_pairs[1]
                && census.final_period_pairs == census.unpunctuated_pairs
                && census.opening_before_next == census.next_before_opening
                && census
                    .roles_by_id
                    .values()
                    .all(|counts| *counts == [split.pairs_per_family / width; 4]),
            format!("imbalanced matched panel family {}", family.name),
        )?;
        coverage.insert(family.name, census);
    }
    rows.flush()?;
    keys.flush()?;
    Ok(json!({
        "rows": split.pairs_per_family * split.families.len() * 2,
        "pairs": split.pairs_per_family * split.families.len(),
        "prefix_min": min_prefix, "prefix_max": max_prefix,
        "values": split.pool.iter().map(|label| (label.id, label.word)).collect::<BTreeMap<_, _>>(),
        "coverage": coverage,
    }))
}

pub(super) fn run(root: &Path, output: &Path) -> Result<()> {
    let vocab = tokenizer(&root.join(history::TOKENIZER))?;
    let protocol = DialogueProtocol::literal_roles_v2(&vocab)?;
    let builder = Builder::new(&vocab, &protocol)?;
    let history = history::History::load(root)?;
    let mut prior = history.screens(root, &vocab)?;
    let older = PreviouslySeen::load(root, &mut prior, &vocab)?;
    let words = labels(&builder, &prior, &older)?;
    fs::create_dir(output)?;
    let mut fresh = Fresh::default();
    for split in [
        Split {
            name: "finite",
            families: design::FINITE,
            topics: &[design::FINITE_TOPIC],
            pool: &words[10..18],
            pairs_per_family: 16,
            person_start: 0,
        },
        Split {
            name: "held",
            families: design::HELD,
            topics: design::TOPICS,
            pool: &words[..10],
            pairs_per_family: 50,
            person_start: 128,
        },
    ] {
        let name = split.name;
        let census = build_split(split, &builder, &prior, &older, &mut fresh, output)?;
        let directory = output.join(name);
        let receipt = json!({
            "schema": "codex-4320.postperiod-panel/1",
            "split": name, "tokenizer_sha256": history::TOKENIZER_SHA256,
            "question_id": QUESTION, "fact_delimiter": [NEWLINE, NEWLINE], "assistant_marker": MARKER,
            "period_id": PERIOD, "eos_id": EOS,
            "collision_ids": COLLISION, "prior_prompt_and_donor_sha256": history.pins,
            "immediate_panel_pins": older.source_pins, "census": census,
            "prior_matching_full_prefixes": 0, "prior_matching_facts_or_bare_questions_in_immediate_panels": 0,
            "role_order_for_counts": ["opening", "next", "rival_opening", "rival_next"],
        });
        let mut file = fresh_writer(&directory.join("receipt.json"))?;
        serde_json::to_writer_pretty(&mut file, &receipt)?;
        writeln!(file)?;
        file.flush()?;
        write_manifest(&directory, &["rows.jsonl", "keys.jsonl", "receipt.json"])?;
    }
    write_manifest(output, &["finite/SHA256SUMS", "held/SHA256SUMS"])
}
