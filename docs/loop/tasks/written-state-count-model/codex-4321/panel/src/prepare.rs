use super::{design::*, *};
use serde_json::json;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};

pub const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
pub const TOKENIZER_SHA256: &str =
    "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const PRIOR: &str = "20261127-codex-postperiod/panel-v1";
const PRIOR_SHA256: &str = "56091ad40d7e5da6f3d6742e02f04e37cdb01171fc3fddb4acba9bcf4f11b876";

#[derive(Default)]
struct Seen {
    prefixes: BTreeSet<Vec<u32>>,
    texts: BTreeSet<String>,
    contexts: BTreeSet<String>,
    templates: BTreeSet<String>,
    labels: BTreeSet<u32>,
}

#[derive(Deserialize)]
struct PriorRow {
    prefix: Vec<u32>,
    target: u32,
    next_id: u32,
    family: String,
}
#[derive(Deserialize)]
struct PriorKey {
    user: String,
    question: String,
    directive: String,
    context: String,
    queried_person: String,
    rival_person: String,
    authorized_fact: String,
    rival_fact: String,
    opening_word: String,
    next_word: String,
    rival_opening: String,
    rival_next: String,
}

fn read_prior(root: &Path, vocab: &ByteBpeTokenizer) -> Result<Seen> {
    let root = root.join(PRIOR);
    require(
        hash(&root.join("SHA256SUMS"))? == PRIOR_SHA256,
        "prior manifest drift",
    )?;
    verify_manifest(&root, &root.join("SHA256SUMS"))?;
    let mut seen = Seen::default();
    for split in ["finite", "held"] {
        let dir = root.join(split);
        verify_manifest(&dir, &dir.join("SHA256SUMS"))?;
        for row in read_jsonl::<PriorRow>(&dir.join("rows.jsonl"))? {
            seen.prefixes.insert(row.prefix);
            seen.labels.extend([row.target, row.next_id]);
            seen.templates.insert(normalize(&row.family));
        }
        for key in read_jsonl::<PriorKey>(&dir.join("keys.jsonl"))? {
            seen.texts.insert(normalize(&key.user));
            seen.contexts.insert(normalize(&key.context));
            seen.templates
                .insert(normalize(key.directive.trim_end_matches('.')));
            seen.templates.insert(normalize(
                &key.question.replace(&key.queried_person, "{name}"),
            ));
            for (fact, name, first, next) in [
                (
                    key.authorized_fact,
                    key.queried_person,
                    key.opening_word,
                    key.next_word,
                ),
                (
                    key.rival_fact,
                    key.rival_person,
                    key.rival_opening,
                    key.rival_next,
                ),
            ] {
                for word in [&first, &next] {
                    seen.labels.extend(vocab.encode(&format!(" {word}")));
                }
                seen.templates.insert(normalize(
                    &fact
                        .replace(&name, "{name}")
                        .replace(&format!(" {first}"), " {word}")
                        .replace(&format!(" {next}"), " {word}"),
                ));
            }
        }
    }
    Ok(seen)
}

fn names(split: usize, family: usize, contrast: u8, index: usize) -> String {
    format!("Case {split}{family}{contrast}{index:03}")
}

const INTRO: &str =
    " Start with the first word and a period; put one space between any later words. ";
struct Wire<'a> {
    vocab: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
}

impl Wire<'_> {
    fn encode(
        &self,
        user: &str,
        context: &str,
        reply: &str,
        labels: &[Label; 4],
        offsets: [usize; 4],
        expected: &[u32],
    ) -> Result<(Vec<u32>, [u32; 4])> {
        let user_turn = Message {
            role: "user",
            content: user,
        };
        let open = self.encoder.encode_assistant_prefix(&[user_turn]);
        let complete = self.encoder.encode_document(&[
            Message {
                role: "user",
                content: user,
            },
            Message {
                role: "assistant",
                content: reply,
            },
        ]);
        let head = [vec![0], self.vocab.encode("User:")].concat();
        let body = open
            .tokens
            .strip_prefix(head.as_slice())
            .and_then(|ids| ids.strip_suffix(MARKER.as_slice()))
            .ok_or("protocol marker drift")?;
        let delimiters = body
            .windows(2)
            .enumerate()
            .filter_map(|(i, ids)| (ids == [NEWLINE, NEWLINE]).then_some(i))
            .collect::<Vec<_>>();
        let [delimiter] = delimiters.as_slice() else {
            return Err("fact boundary drift".into());
        };
        let questions = body
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == QUESTION).then_some(i))
            .collect::<Vec<_>>();
        let [question] = questions.as_slice() else {
            return Err("question boundary drift".into());
        };
        let user_wire = format!(" {user}");
        let (ordinary, fallbacks) = self.vocab.encode_lossy(&user_wire);
        require(
            *question > delimiter + 1
                && open.tokens.len() + 5 < LIMIT
                && open.special_token_occurrences == 0
                && complete.special_token_occurrences == 0
                && open.emitted_turns == 1
                && complete.emitted_turns == 2
                && !open.tokens.contains(&EOS)
                && fallbacks == 0
                && ordinary == body
                && self.vocab.decode_bytes(body) == user_wire.as_bytes()
                && self.vocab.decode_bytes(&body[..*delimiter]) == format!(" {context}").as_bytes()
                && complete.tokens.strip_prefix(open.tokens.as_slice()) == Some(expected)
                && complete.response_mask.get(open.tokens.len()..)
                    == Some(vec![1; expected.len()].as_slice())
                && self.vocab.decode_bytes(&expected[..expected.len() - 1])
                    == format!(" {reply}").as_bytes(),
            format!("canonical text or reference drift for {user}"),
        )?;
        let mut positions = [0; 4];
        for (slot, (label, offset)) in labels.iter().zip(offsets).enumerate() {
            let hits = open
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(i, &id)| (id == label.id).then_some(i))
                .collect::<Vec<_>>();
            let [position] = hits.as_slice() else {
                return Err(format!("{} occurs {} times in {user}", label.word, hits.len()).into());
            };
            require(
                (head.len()..head.len() + delimiter).contains(position)
                    && self
                        .vocab
                        .decode_bytes(&open.tokens[head.len()..*position])
                        .len()
                        == offset
                    && self.vocab.decode_bytes(&[label.id])
                        == format!(" {}", label.word).as_bytes(),
                format!("wrong original occurrence for {}", label.word),
            )?;
            positions[slot] = (*position).try_into()?;
        }
        Ok((open.tokens, positions))
    }
}

#[derive(Clone)]
struct Label {
    id: u32,
    word: String,
}

fn label_pool(vocab: &ByteBpeTokenizer, prior: &Seen) -> Result<Vec<Label>> {
    let mut fixed_ids = prior.labels.clone();
    for template in FACTS
        .into_iter()
        .chain(QUESTIONS)
        .chain(ROLE)
        .chain([INTRO, " Case "])
    {
        fixed_ids.extend(vocab.encode(template));
        fixed_ids.extend(vocab.encode(&format!(" {template}")));
    }
    for family in TRAIN.iter().chain(PRIMARY).chain(SECONDARY).chain(FINITE) {
        for directive in family.wording.iter().flatten().flatten() {
            fixed_ids.extend(vocab.encode(&format!(" {directive}")));
        }
    }
    let words = "apple berry river stone garden cloud bird forest field water fire sand         valley tree ocean star moon gold silver green red blue white black violet orange         sugar salt bread milk soup honey chicken paper wood metal book grass mountain hill         road house door glass floor crown king queen fish horse dog cat bear wolf rock         rain snow wind ice spring summer autumn winter night dawn sun rose leaf fruit         school music sound light bridge beach flower boat ship train wheel clock ring         circle square mouse lion tiger table chair town wall park hall farm gate         milk bread horse mole duck lamb ribbon pocket bottle flame frame shore bank         brown brown cheese cream butter grain rice wheat seed crown bone pearl jewel";
    let mut pool = Vec::new();
    for word in words.split_whitespace() {
        let text = format!(" {word}");
        if let [id] = vocab.encode(&text).as_slice() {
            if !fixed_ids.insert(*id) {
                continue;
            }
            require(
                vocab.decode_bytes(&[*id]) == text.as_bytes(),
                "label decode drift",
            )?;
            pool.push(Label {
                id: *id,
                word: word.to_owned(),
            });
        }
    }
    require(
        pool.len() >= 32,
        format!("only {} fresh whole-word values", pool.len()),
    )?;
    Ok(pool)
}

struct Split {
    index: usize,
    name: &'static str,
    families: &'static [Family],
    contrasts: &'static [u8],
    pairs: usize,
}

fn fact_permutation(mut index: usize) -> [usize; 4] {
    let mut remaining = vec![0, 1, 2, 3];
    std::array::from_fn(|_| {
        let choice = index % remaining.len();
        index /= remaining.len();
        remaining.remove(choice)
    })
}

fn register_templates(split: &Split, seen: &mut Seen) -> Result<()> {
    let mut this_split = BTreeSet::new();
    this_split.insert(normalize(FACTS[split.index]));
    this_split.insert(normalize(QUESTIONS[split.index]));
    for family in split.families {
        this_split.insert(normalize(family.name));
        for &contrast in split.contrasts {
            for pair in family.wording[usize::from(contrast)] {
                for directive in pair {
                    require(!directive.is_empty(), "empty directive")?;
                    this_split.insert(normalize(directive));
                }
            }
        }
    }
    for template in this_split {
        require(
            seen.templates.insert(template.clone()),
            format!("reused prior/split template: {template}"),
        )?;
    }
    Ok(())
}

#[derive(Default, Serialize)]
struct Census {
    pairs: usize,
    punctuation: [usize; 2],
    variant: [usize; 2],
    equal_lengths: usize,
    negative_length_delta: usize,
    positive_length_delta: usize,
    role_by_label: BTreeMap<u32, [usize; 4]>,
    role_by_fact_slot: [[usize; 4]; 4],
}

fn generate(
    split: &Split,
    root: &Path,
    wire: &Wire<'_>,
    pool: &[Label],
    seen: &mut Seen,
) -> Result<()> {
    register_templates(split, seen)?;
    let dir = root.join(split.name);
    fs::create_dir(&dir)?;
    let mut rows = fresh_writer(&dir.join("rows.jsonl"))?;
    let mut keys = fresh_writer(&dir.join("keys.jsonl"))?;
    let mut counts = BTreeMap::new();
    let mut min = usize::MAX;
    let mut max = 0;
    for (family_index, family) in split.families.iter().enumerate() {
        for &contrast in split.contrasts {
            let mut census = Census::default();
            for index in 0..split.pairs {
                let name = names(split.index, family_index, contrast, index);
                let question = QUESTIONS[split.index].replace("{name}", &name);
                let labels: [Label; 4] = std::array::from_fn(|role| {
                    pool[(index + family_index + role * (1 + 2 * ((index / 8) % 4))) % pool.len()]
                        .clone()
                });
                let ids = std::array::from_fn(|role| labels[role].id);
                let order = fact_permutation((index / 2 + family_index * 7) % 24);
                let mut context = String::new();
                let mut offsets = [0; 4];
                for (slot, role) in order.into_iter().enumerate() {
                    let (before, after) = FACTS[split.index]
                        .split_once("{word}")
                        .ok_or("missing fact value")?;
                    let expand =
                        |part: &str| part.replace("{name}", &name).replace("{role}", ROLE[role]);
                    if slot != 0 {
                        context.push(' ');
                    }
                    context.push_str(&expand(before));
                    require(context.ends_with(' '), "source token lacks a leading space")?;
                    offsets[role] = context.len();
                    context.push_str(&labels[role].word);
                    context.push_str(&expand(after));
                    census.role_by_fact_slot[role][slot] += 1;
                }
                require(seen.contexts.insert(normalize(&context)), "reused context")?;
                let pair_id = format!(
                    "{:x}",
                    Sha256::digest(format!(
                        "codex-4321\n{}\n{}\n{context}\n{question}",
                        split.name, family.name
                    ))
                )[..24]
                    .to_owned();
                let punctuate = (index / 2 + family_index) % 2;
                let variant = (index + family_index) % 2;
                let mut pair = Vec::new();
                for (side, directive) in family.wording[usize::from(contrast)][variant]
                    .into_iter()
                    .enumerate()
                {
                    let directive = format!("{directive}{}", if punctuate == 1 { "." } else { "" });
                    let user = format!("{context}\n\n{question}{INTRO}{directive}");
                    let followups = usize::from(contrast) + side;
                    let mut reply = format!("{}.", labels[0].word);
                    let mut expected = vec![labels[0].id, PERIOD];
                    for label in labels.iter().skip(1).take(followups) {
                        reply.push(' ');
                        reply.push_str(&label.word);
                        expected.push(label.id);
                    }
                    expected.push(EOS);
                    let (prefix, positions) =
                        wire.encode(&user, &context, &reply, &labels, offsets, &expected)?;
                    require(
                        seen.texts.insert(normalize(&user)) && seen.prefixes.insert(prefix.clone()),
                        "reused prompt",
                    )?;
                    min = min.min(prefix.len());
                    max = max.max(prefix.len());
                    let row = Row {
                        pair_id: pair_id.clone(),
                        family: family.name.to_owned(),
                        side: side.try_into()?,
                        contrast,
                        prefix,
                        positions,
                        ids,
                        expected,
                    };
                    serde_json::to_writer(&mut rows, &row)?;
                    writeln!(rows)?;
                    serde_json::to_writer(
                        &mut keys,
                        &json!({
                            "pair_id": pair_id, "side": side, "contrast": contrast, "user": user,
                            "context": context, "question": question, "directive": directive,
                            "reply": reply, "fact_role_order": order, "variant": variant,
                        }),
                    )?;
                    writeln!(keys)?;
                    pair.push(row);
                }
                let [stop, copy] = pair.as_slice() else {
                    return Err("incomplete pair".into());
                };
                let question_pos = stop
                    .prefix
                    .iter()
                    .position(|&id| id == QUESTION)
                    .ok_or("missing query mark")?;
                require(
                    stop.prefix[..=question_pos] == copy.prefix[..=question_pos]
                        && stop.positions == copy.positions
                        && stop.ids == copy.ids
                        && stop.prefix != copy.prefix,
                    "paired fact or question drift",
                )?;
                census.pairs += 1;
                census.punctuation[punctuate] += 1;
                census.variant[variant] += 1;
                census.equal_lengths += usize::from(stop.prefix.len() == copy.prefix.len());
                census.negative_length_delta += usize::from(stop.prefix.len() > copy.prefix.len());
                census.positive_length_delta += usize::from(stop.prefix.len() < copy.prefix.len());
                for (role, id) in ids.into_iter().enumerate() {
                    census.role_by_label.entry(id).or_default()[role] += 1;
                }
            }
            counts.insert(format!("{}-{contrast}", family.name), census);
        }
    }
    rows.flush()?;
    keys.flush()?;
    let mut receipt = fresh_writer(&dir.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema": "codex-4321.source-choice-panel/1", "split": split.name,
            "tokenizer_sha256": TOKENIZER_SHA256, "prior_manifest_sha256": PRIOR_SHA256,
            "rows": split.families.len() * split.contrasts.len() * split.pairs * 2,
            "prefix_min": min, "prefix_max": max, "families": counts,
            "labels": pool.iter().map(|label| (label.id, &label.word)).collect::<BTreeMap<_, _>>(),
            "role_order": ["A", "B", "C", "D"],
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    write_manifest(&dir, &["rows.jsonl", "keys.jsonl", "receipt.json"])
}

pub(super) fn run(root: &Path, output: &Path) -> Result<()> {
    let vocab = tokenizer(&root.join(TOKENIZER))?;
    let protocol = DialogueProtocol::literal_roles_v2(&vocab)?;
    let wire = Wire {
        vocab: &vocab,
        encoder: protocol.bind(&vocab)?,
    };
    let mut seen = read_prior(root, &vocab)?;
    let pool = label_pool(&vocab, &seen)?;
    fs::create_dir(output)?;
    for split in [
        Split {
            index: 0,
            name: "train",
            families: TRAIN,
            contrasts: &[0, 1],
            pairs: 48,
        },
        Split {
            index: 1,
            name: "finite",
            families: FINITE,
            contrasts: &[0, 1],
            pairs: 8,
        },
        Split {
            index: 2,
            name: "held-primary",
            families: PRIMARY,
            contrasts: &[1],
            pairs: 50,
        },
        Split {
            index: 3,
            name: "held-secondary",
            families: SECONDARY,
            contrasts: &[0],
            pairs: 50,
        },
    ] {
        let start = split.index * 8;
        generate(&split, output, &wire, &pool[start..start + 8], &mut seen)?;
    }
    write_manifest(
        output,
        &[
            "train/SHA256SUMS",
            "finite/SHA256SUMS",
            "held-primary/SHA256SUMS",
            "held-secondary/SHA256SUMS",
        ],
    )
}
