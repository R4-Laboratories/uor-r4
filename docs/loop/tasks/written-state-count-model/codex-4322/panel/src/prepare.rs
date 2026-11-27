use super::{design::*, *};
use serde_json::json;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};

pub const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
pub const TOKENIZER_SHA256: &str =
    "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const SOURCE_CHOICE: &str = "20261127-codex-source-choice/panel-v1";
const SOURCE_CHOICE_SHA256: &str =
    "2b6abf25514e9a744a447659388003994ea5a0d1e92c50dd7934337da0968b7c";
const ORIGINAL_SPLITS: [&str; 4] = ["train", "finite", "held-primary", "held-secondary"];
const PRIOR: &str = "20261127-codex-postperiod/panel-v1";
const PRIOR_SHA256: &str = "56091ad40d7e5da6f3d6742e02f04e37cdb01171fc3fddb4acba9bcf4f11b876";

#[derive(Default)]
struct Seen {
    prefixes: BTreeSet<Vec<u32>>,
    texts: BTreeSet<String>,
    contexts: BTreeSet<String>,
    templates: BTreeSet<String>,
    labels: BTreeSet<u32>,
    pairs: BTreeSet<String>,
    facts: BTreeSet<String>,
    directive_words: BTreeMap<String, BTreeMap<String, usize>>,
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
    format!("PM{split}{family}{contrast}{index:03}")
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
            format!(
                "canonical text or reference drift for {user}; prefix length {}, fallback count {}",
                open.tokens.len(),
                fallbacks
            ),
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
        .chain([INTRO, " PM "])
    {
        fixed_ids.extend(vocab.encode(template));
        fixed_ids.extend(vocab.encode(&format!(" {template}")));
    }
    for family in DEV_FIRST
        .iter()
        .chain(DEV_POST)
        .chain(HELD_FIRST)
        .chain(HELD_POST)
    {
        for directive in family.wording.iter().flatten() {
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
        pool.len() >= 16,
        format!("only {} fresh whole-word values", pool.len()),
    )?;
    Ok(pool)
}

struct Split {
    index: usize,
    name: &'static str,
    families: [&'static [Family]; 2],
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
    for family in split.families.into_iter().flatten() {
        require(
            this_split.insert(normalize(family.name)),
            "duplicate family",
        )?;
        for directive in family.wording.into_iter().flatten() {
            require(!directive.is_empty(), "empty directive")?;
            require(
                this_split.insert(normalize(directive)),
                "duplicate directive template",
            )?;
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
    same_word_variants: [bool; 2],
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
    let mut directive_words = BTreeMap::new();
    let mut min = usize::MAX;
    let mut max = 0;
    for (contrast, families) in split.families.into_iter().enumerate() {
        let pairs = [50, 25][contrast];
        let contrast = u8::try_from(contrast)?;
        for (family_index, family) in families.iter().enumerate() {
            let mut census = Census::default();
            for index in 0..pairs {
                let name = names(split.index, family_index, contrast, index);
                let question = QUESTIONS[split.index].replace("{name}", &name);
                let labels: [Label; 4] = std::array::from_fn(|role| {
                    pool[(index + family_index + role * (1 + 2 * ((index / 8) % 4))) % pool.len()]
                        .clone()
                });
                let ids = std::array::from_fn(|role| labels[role].id);
                let order = fact_permutation((index + family_index * 7) % 24);
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
                        "codex-4322\n{}\n{}\n{context}\n{question}",
                        split.name, family.name
                    ))
                )[..24]
                    .to_owned();
                let punctuate = (index / 2 + family_index) % 2;
                let variant = (index + family_index) % 2;
                require(seen.pairs.insert(pair_id.clone()), "reused pair id")?;
                for fact in context.split_terminator(". ") {
                    require(
                        seen.facts.insert(normalize(fact.trim_end_matches('.'))),
                        "reused fact",
                    )?;
                }
                let mut pair = Vec::new();
                for (side, directive) in family.wording[variant].into_iter().enumerate() {
                    let directive = format!("{directive}{}", if punctuate == 1 { "." } else { "" });
                    add_words(&mut directive_words, &directive);
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
                verify_references(stop, copy)?;
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
            for (variant, [stop, copy]) in family.wording.into_iter().enumerate() {
                census.same_word_variants[variant] = word_multiset(stop) == word_multiset(copy);
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
            "source_choice_manifest_sha256": SOURCE_CHOICE_SHA256,
            "contrast_pairs": [200, 100], "cap": 5,
            "directive_words": overlap_receipt(&directive_words, &seen.directive_words),
            "tokenizer_sha256": TOKENIZER_SHA256, "prior_manifest_sha256": PRIOR_SHA256,
            "rows": 600,
            "prefix_min": min, "prefix_max": max, "families": counts,
            "labels": pool.iter().map(|label| (label.id, &label.word)).collect::<BTreeMap<_, _>>(),
            "role_order": ["A", "B", "C", "D"],
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    seen.directive_words
        .insert(split.name.to_owned(), directive_words);
    write_manifest(&dir, &["rows.jsonl", "keys.jsonl", "receipt.json"])
}

#[derive(Deserialize)]
struct OriginalKey {
    pair_id: String,
    side: u8,
    user: String,
    question: String,
    directive: String,
    context: String,
}

fn word_multiset(text: &str) -> BTreeMap<String, usize> {
    let mut words = BTreeMap::new();
    add_words(&mut words, text);
    words
}

fn add_words(words: &mut BTreeMap<String, usize>, text: &str) {
    for word in text.split(|letter: char| !letter.is_ascii_alphanumeric()) {
        if !word.is_empty() {
            *words.entry(word.to_ascii_lowercase()).or_default() += 1;
        }
    }
}

fn overlap_receipt(
    current: &BTreeMap<String, usize>,
    previous: &BTreeMap<String, BTreeMap<String, usize>>,
) -> serde_json::Value {
    let comparisons: BTreeMap<_, _> = previous.iter().map(|(split, known)| {
        let shared: BTreeMap<_, _> = current.iter().filter(|(word, _)| known.contains_key(*word)).collect();
        (split, json!({
            "shared_types": shared.len(),
            "current_occurrences_of_shared_types": shared.values().map(|&&count| count).sum::<usize>(),
            "shared_words": shared,
        }))
    }).collect();
    let cues = [
        "before", "after", "stop", "end", "finish", "keep", "omit", "one", "two", "three", "first",
        "second", "third", "word", "words",
    ]
    .into_iter()
    .map(|cue| (cue, current.get(cue).copied().unwrap_or_default()))
    .collect::<BTreeMap<_, _>>();
    json!({
        "types": current.len(), "occurrences": current.values().sum::<usize>(),
        "control_cues": cues,
        "comparisons": comparisons,
    })
}

fn register_original(root: &Path, vocab: &ByteBpeTokenizer, seen: &mut Seen) -> Result<()> {
    require(
        hash(&root.join("SHA256SUMS"))? == SOURCE_CHOICE_SHA256,
        "#4321 root manifest drift",
    )?;
    verify_manifest(root, &root.join("SHA256SUMS"))?;
    for split in ORIGINAL_SPLITS {
        let dir = root.join(split);
        verify_manifest(&dir, &dir.join("SHA256SUMS"))?;
        let rows = read_jsonl::<Row>(&dir.join("rows.jsonl"))?;
        let keys = read_jsonl::<OriginalKey>(&dir.join("keys.jsonl"))?;
        require(rows.len() == keys.len(), "original key count mismatch")?;
        let mut words = BTreeMap::new();
        for (row, key) in rows.into_iter().zip(keys) {
            require(
                row.pair_id == key.pair_id && row.side == key.side,
                "original key join mismatch",
            )?;
            seen.prefixes.insert(row.prefix);
            seen.labels.extend(row.ids);
            seen.pairs.insert(row.pair_id);
            seen.contexts.insert(normalize(&key.context));
            seen.texts.insert(normalize(&key.user));
            seen.templates
                .insert(normalize(key.directive.trim_end_matches('.')));
            seen.templates.insert(normalize(&row.family));
            add_words(&mut words, &key.directive);
            for fact in key.context.split_terminator(". ") {
                seen.facts.insert(normalize(fact.trim_end_matches('.')));
            }
            if let Some(case) = key
                .question
                .split_whitespace()
                .collect::<Vec<_>>()
                .windows(2)
                .find(|words| words[0] == "Case")
            {
                let name = format!("{} {}", case[0], case[1].trim_end_matches('?'));
                seen.templates
                    .insert(normalize(&key.question.replace(&name, "{name}")));
                for fact in key.context.split_terminator(". ") {
                    let mut template = fact.replace(&name, "{name}");
                    for ordinal in ROLE {
                        template = template.replace(&format!(" {ordinal} word"), " {role} word");
                    }
                    for id in row.ids {
                        let value = String::from_utf8(vocab.decode_bytes(&[id]))?;
                        template = template.replace(&format!("{value} "), " {word} ");
                        if let Some(head) = template.strip_suffix(&format!("{value}.")) {
                            template = format!("{head} {{word}}.");
                        } else if let Some(head) = template.strip_suffix(&value) {
                            template = format!("{head} {{word}}");
                        }
                    }
                    if !template.ends_with('.') {
                        template.push('.');
                    }
                    require(
                        template.matches("{word}").count() == 1,
                        "wrong original normalized fact",
                    )?;
                    seen.templates.insert(normalize(&template));
                }
            }
        }
        seen.directive_words.insert(split.to_owned(), words);
    }
    Ok(())
}

fn verify_references(stop: &Row, copy: &Row) -> Result<()> {
    require(
        stop.side == 0
            && copy.side == 1
            && stop.contrast == copy.contrast
            && stop.pair_id == copy.pair_id
            && stop.family == copy.family
            && stop.ids == copy.ids
            && stop.positions == copy.positions,
        "reference pair identity drift",
    )?;
    let [a, b, c, d] = stop.ids;
    let (short, long) = match stop.contrast {
        0 => (vec![a, PERIOD, EOS], vec![a, PERIOD, b, EOS]),
        1 => (vec![a, PERIOD, b, EOS], vec![a, PERIOD, b, c, EOS]),
        _ => return Err("unexpected release contrast".into()),
    };
    require(
        stop.expected == short
            && copy.expected == long
            && !stop.expected.contains(&d)
            && !copy.expected.contains(&d),
        "incorrect full oracle target",
    )?;
    let common = short.len() - 1;
    require(
        short[..common] == long[..common] && short[common] == EOS && long[common] != EOS,
        "opposite actions must differ at common prefix",
    )
}

fn protect(original: &Path, output: &Path) -> Result<()> {
    let original = original.join("held-primary");
    let dir = output.join("protect");
    let row_path = original.join("rows.jsonl");
    let key_path = original.join("keys.jsonl");
    fs::create_dir(&dir)?;
    let mut rows = fresh_writer(&dir.join("rows.jsonl"))?;
    let mut keys = fresh_writer(&dir.join("keys.jsonl"))?;
    let mut protected = Vec::new();
    for (row_line, key_line) in BufReader::new(File::open(&row_path)?)
        .lines()
        .zip(BufReader::new(File::open(&key_path)?).lines())
    {
        let (row_line, key_line) = (row_line?, key_line?);
        let row: Row = serde_json::from_str(&row_line)?;
        let key: OriginalKey = serde_json::from_str(&key_line)?;
        require(
            row.pair_id == key.pair_id && row.side == key.side,
            "protect key join drift",
        )?;
        if row.family == "ceiling" {
            require(row.contrast == 1, "protect contrast changed")?;
            writeln!(rows, "{row_line}")?;
            writeln!(keys, "{key_line}")?;
            protected.push(row);
        }
    }
    require(protected.len() == 100, "protect size drift")?;
    for pair in protected.chunks_exact(2) {
        verify_references(&pair[0], &pair[1])?;
    }
    rows.flush()?;
    keys.flush()?;
    let mut receipt = fresh_writer(&dir.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema": "codex-4321.source-choice-panel/1", "split": "protect",
            "rows": 100, "contrast_pairs": [0, 50], "cap": 5,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "source_choice_manifest_sha256": SOURCE_CHOICE_SHA256,
            "source_split": "held-primary", "source_family": "ceiling",
            "source_rows_sha256": hash(&row_path)?, "source_keys_sha256": hash(&key_path)?,
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
    let original = root.join(SOURCE_CHOICE);
    let mut seen = read_prior(root, &vocab)?;
    register_original(&original, &vocab, &mut seen)?;
    let pool = label_pool(&vocab, &seen)?;
    fs::create_dir(output)?;
    for split in [
        Split {
            index: 0,
            name: "dev",
            families: [DEV_FIRST, DEV_POST],
        },
        Split {
            index: 1,
            name: "held",
            families: [HELD_FIRST, HELD_POST],
        },
    ] {
        let start = split.index * 8;
        let labels = &pool[start..start + 8];
        for label in labels {
            require(seen.labels.insert(label.id), "reused source role id")?;
        }
        generate(&split, output, &wire, labels, &mut seen)?;
    }
    protect(&original, output)?;
    write_manifest(
        output,
        &["dev/SHA256SUMS", "held/SHA256SUMS", "protect/SHA256SUMS"],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(contrast: u8, side: u8, expected: Vec<u32>) -> Row {
        Row {
            pair_id: "pair".to_owned(),
            side,
            contrast,
            positions: [1, 2, 3, 4],
            ids: [11, 12, 13, 14],
            prefix: vec![],
            expected,
            family: "fixture".to_owned(),
        }
    }

    #[test]
    fn answers_require_the_opposite_action_at_the_common_prefix() {
        for (contrast, short, long) in [
            (0, vec![11, PERIOD, EOS], vec![11, PERIOD, 12, EOS]),
            (1, vec![11, PERIOD, 12, EOS], vec![11, PERIOD, 12, 13, EOS]),
        ] {
            let stop = example(contrast, 0, short.clone());
            let copy = example(contrast, 1, long.clone());
            assert!(verify_references(&stop, &copy).is_ok());
            assert!(verify_references(&stop, &example(contrast, 1, short)).is_err());
            assert!(verify_references(&example(contrast, 0, long), &copy).is_err());
            let mut open = copy.clone();
            open.expected.pop();
            assert!(verify_references(&stop, &open).is_err());
            let mut decoy = copy;
            let target = decoy.expected.len() - 2;
            decoy.expected[target] = 14;
            assert!(verify_references(&stop, &decoy).is_err());
        }
    }

    #[test]
    fn scope_variants_preserve_the_word_multiset() {
        for groups in [[DEV_FIRST, DEV_POST], [HELD_FIRST, HELD_POST]] {
            for families in groups {
                assert_eq!(families.len(), 4);
                for [stop, copy] in families.last().unwrap().wording {
                    assert_ne!(stop, copy);
                    assert_eq!(word_multiset(stop), word_multiset(copy));
                }
            }
        }
    }

    #[test]
    fn rejects_templates_shared_across_splits() {
        let mut seen = Seen::default();
        let dev = Split {
            index: 0,
            name: "dev",
            families: [DEV_FIRST, DEV_POST],
        };
        let held = Split {
            index: 1,
            name: "held",
            families: [HELD_FIRST, HELD_POST],
        };
        register_templates(&dev, &mut seen).unwrap();
        register_templates(&held, &mut seen).unwrap();
        assert!(register_templates(&dev, &mut seen).is_err());
    }
}
