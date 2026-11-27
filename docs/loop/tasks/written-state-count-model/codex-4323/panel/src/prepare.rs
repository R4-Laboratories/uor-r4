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
    for split in &SPLITS {
        for text in split
            .fact
            .into_iter()
            .chain([split.question, INTRO])
            .chain(ROLE)
        {
            fixed_ids.extend(vocab.encode(text));
            fixed_ids.extend(vocab.encode(&format!(" {text}")));
        }
        for family in &split.families {
            for cross in 0..4 {
                for text in family.directives(cross) {
                    fixed_ids.extend(vocab.encode(&format!(" {text}")));
                }
            }
        }
    }
    let words = "apple berry river stone garden cloud bird forest field water fire sand
      valley tree ocean star moon gold silver green red blue white black violet orange
      sugar salt bread milk soup honey chicken paper wood metal book grass mountain hill
      road house door glass floor crown king queen fish horse dog cat bear wolf rock
      rain snow wind ice spring summer autumn winter night dawn sun rose leaf fruit
      school music sound light bridge beach flower boat ship train wheel clock ring
      circle square mouse lion tiger table chair town wall park hall farm gate
      mole duck lamb ribbon pocket bottle flame frame shore bank brown cheese
      cream butter grain rice wheat seed bone pearl jewel shell nail paint brick earth
      coffee tea clay stone marble chalk cotton linen silk fur plum lemon lime peach pear
      ocean stream pond lake ocean copper brass steel iron tin pot pan fork spoon plate
      purple yellow pink amber gray grey north south east west maple oak pine cedar elm
      fox deer goat sheep bull cow pig hen dove crow crane owl eagle hawk robin sparrow
      guitar drum bell horn flute harp coin card stamp string rope wire circle cube
      mask sock shirt glove scarf skirt jacket coat boot shoe bag box cup bowl kettle
      building ceiling curtain desk carpet fence roof basket chest drawer cabinet
      engine wagon bridge carriage track plane airport cloud shadow frost thunder storm
      silverwood granite slate marble sand gravel concrete mortar canvas thread needle";
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
        pool.len() >= LABELS_PER_SPLIT * SPLITS.len(),
        format!("only {} fresh whole-word values", pool.len()),
    )?;
    Ok(pool)
}

fn register_templates(split: &Split, seen: &mut Seen) -> Result<()> {
    let mut templates: BTreeSet<String> = split
        .fact
        .into_iter()
        .chain([split.question])
        .map(normalize)
        .collect();
    for family in &split.families {
        require(templates.insert(normalize(family.name)), "duplicate family")?;
        let directives: BTreeSet<_> = (0..4)
            .flat_map(|cross| family.directives(cross))
            .map(|text| normalize(&text))
            .collect();
        require(
            directives.len() == if family.selectors.is_some() { 4 } else { 2 },
            format!("selector/count cross has duplicate forms: {}", family.name),
        )?;
        for text in directives {
            require(templates.insert(text), "duplicate directive template")?;
        }
    }
    for template in templates {
        require(
            seen.templates.insert(template.clone()),
            format!("reused prior/split template: {template}"),
        )?;
    }
    Ok(())
}

const FACT_ORDERS: [[usize; 4]; 8] = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
    [0, 2, 1, 3],
    [1, 3, 0, 2],
    [3, 0, 2, 1],
    [2, 1, 3, 0],
];

#[derive(Default, Serialize)]
struct Census {
    pairs: usize,
    fact_punctuation: [usize; 2],
    crosses: [usize; 4],
    same_directive_words: usize,
    equal_prefix_tokens: usize,
    equal_directive_bytes: usize,
    same_prefix_len_stop_larger: usize,
    same_prefix_len_copy_larger: usize,
    a_before_b: [usize; 4],
    b_before_a: [usize; 4],
    role_by_label: BTreeMap<u32, [usize; 4]>,
    role_by_fact_slot: [[usize; 4]; 4],
}

fn facts(
    split: &Split,
    labels: &[Label; 4],
    order: [usize; 4],
    punctuation: usize,
    name: &str,
) -> Result<(String, [usize; 4])> {
    let mut context = String::new();
    let mut offsets = [0; 4];
    let (before, after) = split.fact[punctuation]
        .split_once("{word}")
        .ok_or("missing fact value")?;
    for (slot, role) in order.into_iter().enumerate() {
        let expand = |part: &str| part.replace("{name}", name).replace("{role}", ROLE[role]);
        if slot != 0 {
            context.push(' ');
        }
        context.push_str(&expand(before));
        require(context.ends_with(' '), "source token lacks a leading space")?;
        offsets[role] = context.len();
        context.push_str(&labels[role].word);
        context.push_str(&expand(after));
    }
    Ok((context, offsets))
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
    for (family_index, family) in split.families.iter().enumerate() {
        let mut census = Census::default();
        for index in 0..PAIRS_PER_FAMILY {
            let cross = index % 4;
            let layout = index / 4;
            let punctuation = (layout / 2 + family_index) % 2;
            let directives = family.directives(cross);
            let scoped = family.selectors.is_some();
            let axis = match (scoped, cross >= 2) {
                (true, false) => "clause-order-swap",
                (true, true) => "selector-swap",
                (false, _) => "direct-count-control",
            };
            let name = format!("SC{family_index}{index:02}");
            let question = split.question.replace("{name}", &name);
            let labels: [Label; 4] = std::array::from_fn(|role| {
                pool[(layout + cross * 2 + family_index + role * 3) % pool.len()].clone()
            });
            let ids = std::array::from_fn(|role| labels[role].id);
            let order = FACT_ORDERS[layout];
            let (context, offsets) = facts(split, &labels, order, punctuation, &name)?;
            require(seen.contexts.insert(normalize(&context)), "reused context")?;
            let pair_id = format!(
                "{:x}",
                Sha256::digest(format!(
                    "codex-scope-trace\n{}\n{}\n{context}\n{question}",
                    split.name, family.name
                ))
            )[..24]
                .to_owned();
            require(seen.pairs.insert(pair_id.clone()), "reused pair id")?;
            let mut pair = Vec::new();
            census.same_directive_words +=
                usize::from(word_multiset(&directives[0]) == word_multiset(&directives[1]));
            census.equal_directive_bytes += usize::from(directives[0].len() == directives[1].len());
            for (side, directive) in directives.into_iter().enumerate() {
                add_words(&mut directive_words, &directive);
                let user = format!("{context}\n\n{question}{INTRO}{directive}");
                let reply = if side == 0 {
                    format!("{}.", labels[0].word)
                } else {
                    format!("{}. {}", labels[0].word, labels[1].word)
                };
                let expected = if side == 0 {
                    vec![ids[0], PERIOD, EOS]
                } else {
                    vec![ids[0], PERIOD, ids[1], EOS]
                };
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
                    contrast: 0,
                    prefix,
                    positions,
                    ids,
                    expected,
                };
                serde_json::to_writer(&mut rows, &row)?;
                writeln!(rows)?;
                let operative_clause =
                    scoped.then(|| 1 + ((cross % 2) ^ if cross >= 2 { side } else { 0 }));
                let short_clause =
                    scoped.then(|| 1 + ((cross % 2) ^ if cross < 2 { side } else { 0 }));
                serde_json::to_writer(
                    &mut keys,
                    &json!({
                        "pair_id": pair_id, "side": side, "contrast": 0, "user": user,
                        "context": context, "question": question, "directive": directive,
                        "reply": reply, "fact_role_order": order, "variant": cross,
                        "family": family.name, "pair_axis": axis, "quoted": family.quoted,
                        "duplicate_directive_words": family.duplicate_words,
                        "operative_clause_ordinal": operative_clause,
                        "one_word_clause_ordinal": short_clause,
                        "expected_first_period_action": if side == 0 { "stop" } else { "copy" },
                        "fact_punctuation": if punctuation == 0 { "period" } else { "semicolon" },
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
                    && stop.prefix != copy.prefix,
                "fact, question, or directive pairing drift",
            )?;
            verify_references(stop, copy)?;
            census.pairs += 1;
            census.fact_punctuation[punctuation] += 1;
            census.crosses[cross] += 1;
            census.equal_prefix_tokens += usize::from(stop.prefix.len() == copy.prefix.len());
            census.same_prefix_len_stop_larger +=
                usize::from(stop.prefix.len() > copy.prefix.len());
            census.same_prefix_len_copy_larger +=
                usize::from(stop.prefix.len() < copy.prefix.len());
            if stop.positions[0] < stop.positions[1] {
                census.a_before_b[cross] += 1;
            } else {
                census.b_before_a[cross] += 1;
            }
            for (slot, role) in order.into_iter().enumerate() {
                census.role_by_fact_slot[role][slot] += 1;
            }
            for (role, id) in ids.into_iter().enumerate() {
                census.role_by_label.entry(id).or_default()[role] += 1;
            }
        }
        require(
            census.crosses == [8; 4]
                && census.fact_punctuation == [16; 2]
                && census.a_before_b == [4; 4]
                && census.b_before_a == [4; 4]
                && census.role_by_label.len() == LABELS_PER_SPLIT
                && census
                    .role_by_label
                    .values()
                    .all(|roles| !roles.contains(&0)),
            format!("counterbalance or source coverage drift: {}", family.name),
        )?;
        counts.insert(family.name, census);
    }
    rows.flush()?;
    keys.flush()?;
    let pairs = split.families.len() * PAIRS_PER_FAMILY;
    let mut receipt = fresh_writer(&dir.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema": "codex-4321.source-choice-panel/1", "split": split.name,
            "scope_metadata_schema": "codex-scope-trace/1",
            "source_choice_manifest_sha256": SOURCE_CHOICE_SHA256,
            "prior_manifest_sha256": PRIOR_SHA256, "period_margin_manifest_sha256": PERIOD_MARGIN_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "authorship": "Codex agent authored directive and fact templates; deterministic crossing and instantiation; no external or human-authored chats",
            "contrast_pairs": [pairs, 0], "rows": 2 * pairs, "cap": 5,
            "phase": if split.name == "held" { 2 } else { 1 },
            "prefix_min": min, "prefix_max": max, "families": counts,
            "directive_words": overlap_receipt(&directive_words, &seen.directive_words),
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

const PERIOD_MARGIN: &str = "20261127-codex-period-margin/panel-v1";
const PERIOD_MARGIN_SHA256: &str =
    "2d10cc6d64a4197385ca83482f4e93b45a888d3ed10dcf7e18f892d7091c240c";

fn register_period_margin(root: &Path, seen: &mut Seen) -> Result<()> {
    require(
        hash(&root.join("SHA256SUMS"))? == PERIOD_MARGIN_SHA256,
        "#4322 manifest drift",
    )?;
    verify_manifest(root, &root.join("SHA256SUMS"))?;
    for split in ["dev", "held"] {
        let dir = root.join(split);
        verify_manifest(&dir, &dir.join("SHA256SUMS"))?;
        let rows = read_jsonl::<Row>(&dir.join("rows.jsonl"))?;
        let keys = read_jsonl::<OriginalKey>(&dir.join("keys.jsonl"))?;
        require(rows.len() == keys.len(), "#4322 key count mismatch")?;
        let mut words = BTreeMap::new();
        for (row, key) in rows.into_iter().zip(keys) {
            require(
                row.pair_id == key.pair_id && row.side == key.side,
                "#4322 key join mismatch",
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
        }
        seen.directive_words
            .insert(format!("period-margin-{split}"), words);
    }
    Ok(())
}

pub(super) fn run(root: &Path, output: &Path) -> Result<()> {
    let vocab = tokenizer(&root.join(TOKENIZER))?;
    let protocol = DialogueProtocol::literal_roles_v2(&vocab)?;
    let wire = Wire {
        vocab: &vocab,
        encoder: protocol.bind(&vocab)?,
    };
    let mut seen = read_prior(root, &vocab)?;
    register_original(&root.join(SOURCE_CHOICE), &vocab, &mut seen)?;
    register_period_margin(&root.join(PERIOD_MARGIN), &mut seen)?;
    let pool = label_pool(&vocab, &seen)?;
    fs::create_dir(output)?;
    for (split, labels) in SPLITS.iter().zip(pool.chunks_exact(LABELS_PER_SPLIT)) {
        for label in labels {
            require(seen.labels.insert(label.id), "reused source role id")?;
        }
        generate(split, output, &wire, labels, &mut seen)?;
    }
    write_manifest(
        output,
        &["train/SHA256SUMS", "dev/SHA256SUMS", "held/SHA256SUMS"],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_starts_from_opposed_targets() {
        for split in &SPLITS {
            for family in split
                .families
                .iter()
                .filter(|family| family.selectors.is_some())
            {
                let [stop, copy] = family.counts;
                let frames = family.selectors.unwrap();
                for (cross, expected) in [
                    [
                        frames[0].replace("{a}", stop).replace("{b}", copy),
                        frames[0].replace("{a}", copy).replace("{b}", stop),
                    ],
                    [
                        frames[1].replace("{a}", copy).replace("{b}", stop),
                        frames[1].replace("{a}", stop).replace("{b}", copy),
                    ],
                    [
                        frames[0].replace("{a}", stop).replace("{b}", copy),
                        frames[1].replace("{a}", stop).replace("{b}", copy),
                    ],
                    [
                        frames[1].replace("{a}", copy).replace("{b}", stop),
                        frames[0].replace("{a}", copy).replace("{b}", stop),
                    ],
                ]
                .into_iter()
                .enumerate()
                {
                    assert_eq!(family.directives(cross), expected, "{}", family.name);
                }
            }
        }
    }
}
