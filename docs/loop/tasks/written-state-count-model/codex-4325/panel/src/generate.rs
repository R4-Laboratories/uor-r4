use super::*;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, Message};

const SCOPE_CONTEXTS: usize = 8;
const DIRECT_CONTEXTS: usize = 32;
const LABELS: &str = "chalk toast coal silver copper bronze horse cast iron silver purple
    shoe gate dress foam leaf seed bean beans milk bread grain lemon apple spice mint violet
    ash wool sand clay coal snow fog rain wind wax stone bark wood grass fruit steel
    plant shelf table foot knee nail mouth hand arm mouse dog cat lion tiger sharp heavy
    short tall rail street shop craft flag wheel lead white map pond lake boat fish brush
    sliver scale music song picture book tray cat crate fork glass grain brown black blue
    slate bird sweet light bank town park ground shore pearl bridge peach bear lake bone
    shelf sight meat shape flat size land space net rule plane shell paper pocket brown
    fill bread spice port north corn carrot grass piece hull brick frame salt sea yard work
    shirt tent rock rice rubber cream milk cave tree watch scarf flower gold water amber
    dust garden soap kind dark shoe sand mirror mask earth skin horse iron tea chain wheel
    basket tone copy angle hour plate night wire sky grape paint boat sides school reed
    sweep grass sun soap linen summer winter spring roast pine berry fish cedar oak
    elm pear peach gold tulip sharp flour long fire snow ice space path post ring
    star tale thread cash key comb score band lens coin hand steam wall tree wood bowl
    cup ball gear face camp soap dream sword duck coast soap brush leaf lamp porch desk
    skirt goat wing stool pine fig tart cloth dome tape ring tube mist tint station
    cake farm beer rice steel horse woods meat rose pear print foot wool fruit light
    strong fresh stock short cold plain field door case pipe tool horn wing tooth fur
    plain grain stake shock hour salt coat jet pin pitch charm state camp torch cave grass";

#[derive(Clone)]
struct Label {
    id: u32,
    word: &'static str,
}

pub(super) struct Record {
    pub row: Row,
    pub key: Value,
}

pub(super) struct Wire<'a> {
    pub vocab: &'a ByteBpeTokenizer,
    pub dialogue: DialogueEncoder<'a>,
}

struct Context {
    id: String,
    facts: String,
    question: String,
    labels: [Label; 4],
    order: [usize; 4],
    punctuation: char,
}

struct Cell {
    operative: usize,
    short: usize,
}

#[derive(Default, Serialize)]
pub(super) struct Census {
    scoped_families: BTreeMap<&'static str, FamilyCensus>,
    direct_families: BTreeMap<&'static str, FamilyCensus>,
    rows: usize,
    scoped_quartets: usize,
    direct_one_two_pairs: usize,
    direct_two_three_pairs: usize,
    equal_length_quartets: usize,
    equal_length_direct_pairs: usize,
    equal_length_clause_order_pairs: usize,
    equal_length_selector_pairs: usize,
    prefix_min: usize,
    prefix_max: usize,
    labels: BTreeMap<u32, &'static str>,
}

#[derive(Default, Serialize)]
struct FamilyCensus {
    contexts: usize,
    rows: usize,
    equal_length_contexts: usize,
    equal_length_clause_order_pairs: usize,
    equal_length_selector_pairs: usize,
    word_counts: [usize; 3],
    a_before_b: usize,
    b_before_a: usize,
    role_slot_counts: [[usize; 4]; 4],
    label_role_counts: BTreeMap<u32, [usize; 4]>,
}

impl FamilyCensus {
    fn record(&mut self, records: &[Record], context: &Context) {
        self.contexts += 1;
        self.rows += records.len();
        let row = &records[0].row;
        self.a_before_b += usize::from(row.positions[0] < row.positions[1]);
        self.b_before_a += usize::from(row.positions[1] < row.positions[0]);
        for (role, label) in context.labels.iter().enumerate() {
            self.label_role_counts.entry(label.id).or_default()[role] += 1;
        }
        for (slot, &role) in context.order.iter().enumerate() {
            self.role_slot_counts[role][slot] += 1;
        }
        self.equal_length_contexts += usize::from(
            records
                .iter()
                .all(|other| other.row.prefix.len() == row.prefix.len()),
        );
        for record in records {
            self.word_counts[usize::from(record.row.side + record.row.contrast)] += 1;
        }
        if records.len() == 4 {
            let len = |i: usize| records[i].row.prefix.len();
            self.equal_length_clause_order_pairs +=
                usize::from(len(0) == len(1)) + usize::from(len(2) == len(3));
            self.equal_length_selector_pairs +=
                usize::from(len(0) == len(3)) + usize::from(len(1) == len(2));
        }
    }

    fn verify(&self, name: &str, scoped: bool, contrast: usize) -> Result<()> {
        let contexts = if scoped {
            SCOPE_CONTEXTS
        } else {
            DIRECT_CONTEXTS
        };
        let mut expected = [0; 3];
        expected[contrast] = contexts * if scoped { 2 } else { 1 };
        expected[contrast + 1] = expected[contrast];
        require(
            self.contexts == contexts
                && self.word_counts == expected
                && self.label_role_counts.len() == 8
                && self
                    .label_role_counts
                    .values()
                    .all(|slots| *slots == [contexts / 8; 4])
                && (!scoped || self.role_slot_counts == [[2; 4]; 4])
                && (!scoped || (self.a_before_b, self.b_before_a) == (4, 4)),
            format!("wrong assignment census for {name}"),
        )
    }
}

impl Wire<'_> {
    fn encode(&self, facts: &str, user: &str, reply: &str, row: &mut Row) -> Result<()> {
        let turn = || Message {
            role: "user",
            content: user,
        };
        let open = self.dialogue.encode_assistant_prefix(&[turn()]);
        let complete = self.dialogue.encode_document(&[
            turn(),
            Message {
                role: "assistant",
                content: reply,
            },
        ]);
        let head = [vec![0], self.vocab.encode("User:")].concat();
        let body = open
            .tokens
            .strip_prefix(head.as_slice())
            .and_then(|ids| ids.strip_suffix(&MARKER))
            .ok_or("dialogue role marker drift")?;
        let fences: Vec<_> = body
            .windows(2)
            .enumerate()
            .filter_map(|(i, ids)| (ids == FACT_FENCE).then_some(i))
            .collect();
        let questions: Vec<_> = body
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == QUESTION).then_some(i))
            .collect();
        let ([fence], [question]) = (fences.as_slice(), questions.as_slice()) else {
            return Err(format!("unexpected question or fact fence for {user}").into());
        };
        let user_bytes = format!(" {user}");
        let (ordinary, fallbacks) = self.vocab.encode_lossy(&user_bytes);
        require(
            question > &(fence + FACT_FENCE.len())
                && open.tokens.len() + CAP < WRITER_LIMIT
                && !open.tokens.contains(&EOS)
                && open.special_token_occurrences == 0
                && complete.special_token_occurrences == 0
                && open.emitted_turns == 1
                && complete.emitted_turns == 2
                && fallbacks == 0
                && ordinary == body
                && self.vocab.decode_bytes(body) == user_bytes.as_bytes()
                && self.vocab.decode_bytes(&body[..*fence]) == format!(" {facts}").as_bytes()
                && complete.tokens.strip_prefix(open.tokens.as_slice())
                    == Some(row.expected.as_slice())
                && complete.response_mask.get(open.tokens.len()..)
                    == Some(vec![1; row.expected.len()].as_slice())
                && self
                    .vocab
                    .decode_bytes(&row.expected[..row.expected.len() - 1])
                    == format!(" {reply}").as_bytes(),
            format!(
                "dialogue, budget, or reference mismatch at {} ({} prefix tokens): {user}",
                row.row_id,
                open.tokens.len()
            ),
        )?;
        for (role, &id) in row.ids.iter().enumerate() {
            let hits: Vec<_> = open
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(i, &token)| (token == id).then_some(i))
                .collect();
            let [position] = hits.as_slice() else {
                return Err(
                    format!("label {id} occurs {} times at {}", hits.len(), row.row_id).into(),
                );
            };
            require(
                (head.len()..head.len() + fence).contains(position),
                format!("label {id} lies outside facts at {}", row.row_id),
            )?;
            row.positions[role] = (*position).try_into()?;
        }
        row.prefix = open.tokens;
        Ok(())
    }
}

impl Context {
    fn record(
        &self,
        wire: &Wire<'_>,
        design: &Design,
        family: &str,
        directive: &str,
        contrast: u8,
        side: u8,
        cell: Option<&Cell>,
    ) -> Result<Record> {
        let operative = cell.map(|cell| cell.operative + 1);
        let one_word = cell.map(|cell| cell.short + 1);
        let suffix = usize::from(contrast + side);
        let pair_id = match operative {
            Some(ordinal) => format!("{}-s{ordinal}", self.id),
            None => format!("{}-d", self.id),
        };
        let row_id = format!("{pair_id}:{side}");
        let ids = self.labels.each_ref().map(|label| label.id);
        let expected = [vec![ids[0], PERIOD], ids[1..1 + suffix].to_vec(), vec![EOS]].concat();
        let mut reply = format!("{}.", self.labels[0].word);
        for label in &self.labels[1..1 + suffix] {
            reply.push(' ');
            reply.push_str(label.word);
        }
        let user = format!(
            "{}\n\n{} {} {directive}",
            self.facts, self.question, design.format
        );
        let mut row = Row {
            row_id,
            pair_id,
            side,
            contrast,
            positions: [0; 4],
            ids,
            prefix: Vec::new(),
            expected,
            family: family.to_owned(),
        };
        wire.encode(&self.facts, &user, &reply, &mut row)?;
        require(
            self.order
                .windows(2)
                .all(|roles| row.positions[roles[0]] < row.positions[roles[1]]),
            format!(
                "physical fact order disagrees with references at {}",
                row.row_id
            ),
        )?;
        let key = json!({
            "split": design.split, "cohort": "fresh", "source_issue": 4325,
            "source_row_id": row.row_id, "row_id": row.row_id, "pair_id": row.pair_id,
            "context_id": self.id, "side": side, "contrast": contrast, "family": family,
            "pair_axis": if operative.is_some() { "clause-order-swap" } else { "direct-count-control" },
            "selector_swap_id": one_word.map(|ordinal| format!("{}-o{ordinal}", self.id)),
            "one_word_clause_ordinal": one_word, "operative_clause_ordinal": operative,
            "expected_word_count": suffix + 1,
            "expected_first_period_action": if suffix == 0 { "stop" } else { "copy" },
            "expected_post_b_action": match suffix { 0 => None, 1 => Some("stop"), _ => Some("copy") },
            "context": self.facts, "question": self.question, "directive": directive,
            "user": user, "reply": reply, "quoted": directive.contains("'"),
            "fact_role_order": self.order,
            "fact_punctuation": match self.punctuation { ',' => "comma", ';' => "semicolon", _ => "period" },
        });
        Ok(Record { row, key })
    }
}

fn label_pool(vocab: &ByteBpeTokenizer, reserved: &mut BTreeSet<u32>) -> Result<[Label; 8]> {
    let mut pool = Vec::new();
    for word in LABELS.split_whitespace() {
        let text = format!(" {word}");
        if let [id] = vocab.encode(&text).as_slice() {
            if vocab.decode_bytes(&[*id]) == text.as_bytes() && reserved.insert(*id) {
                pool.push(Label { id: *id, word });
            }
        }
        if pool.len() == 8 {
            break;
        }
    }
    let count = pool.len();
    pool.try_into()
        .map_err(|_| format!("only {count} disjoint whole-token labels").into())
}

pub(super) struct Generator<'a> {
    wire: &'a Wire<'a>,
    archive: &'a Archive,
    contexts: BTreeSet<String>,
    prompts: BTreeSet<String>,
    prefixes: BTreeSet<Vec<u32>>,
    row_ids: BTreeSet<String>,
}

impl<'a> Generator<'a> {
    pub fn new(wire: &'a Wire<'a>, archive: &'a Archive) -> Self {
        Self {
            wire,
            archive,
            contexts: BTreeSet::new(),
            prompts: BTreeSet::new(),
            prefixes: BTreeSet::new(),
            row_ids: BTreeSet::new(),
        }
    }

    fn check(&mut self, context: &Context, records: &mut [Record]) -> Result<()> {
        let facts = normalized(&context.facts);
        require(
            !self.archive.contexts.contains(&facts) && self.contexts.insert(facts),
            format!("reused context {}", context.id),
        )?;
        let first = &records[0].row;
        let through_question = first
            .prefix
            .iter()
            .position(|&token| token == QUESTION)
            .ok_or("missing question token")?;
        require(
            records.iter().all(|r| {
                r.row.ids == first.ids
                    && r.row.positions == first.positions
                    && r.row.family == first.family
                    && r.row.prefix.get(..=through_question)
                        == Some(&first.prefix[..=through_question])
            }),
            format!("fact or question drift within {}", context.id),
        )?;
        let equal_length = records
            .iter()
            .all(|r| r.row.prefix.len() == first.prefix.len());
        let lengths: Vec<_> = records.iter().map(|r| r.row.prefix.len()).collect();
        for (pair_index, pair) in records.chunks_exact_mut(2).enumerate() {
            let [stop, copy] = pair else {
                return Err("incomplete generated pair".into());
            };
            require(
                stop.row.side == 0
                    && copy.row.side == 1
                    && stop.row.pair_id == copy.row.pair_id
                    && stop.row.contrast == copy.row.contrast
                    && stop.row.expected[..stop.row.expected.len() - 1]
                        == copy.row.expected[..stop.row.expected.len() - 1],
                format!("incomplete opposing pair in {}", context.id),
            )?;
            let start = pair_index * 2;
            for (i, r) in [stop, copy].into_iter().enumerate() {
                let j = start + i;
                let prompt = normalized(string(&r.key, "user")?);
                require(
                    !self.archive.prompts.contains(&prompt)
                        && !self.archive.prefixes.contains(&r.row.prefix)
                        && !self.archive.pair_ids.contains(&r.row.pair_id)
                        && self.prompts.insert(prompt)
                        && self.prefixes.insert(r.row.prefix.clone())
                        && self.row_ids.insert(r.row.row_id.clone()),
                    format!(
                        "reused prompt, encoded prefix, or row id at {}",
                        r.row.row_id
                    ),
                )?;
                r.key["prefix_tokens"] = json!(lengths[j]);
                r.key["equal_length_context"] = json!(equal_length);
                r.key["equal_length_pair"] = json!(lengths[start] == lengths[start + 1]);
                r.key["equal_length_selector_pair"] = if lengths.len() == 4 {
                    json!(lengths[j] == lengths[3 - j])
                } else {
                    Value::Null
                };
            }
        }
        Ok(())
    }

    pub fn split(
        &mut self,
        design: &Design,
        reserved: &mut BTreeSet<u32>,
    ) -> Result<(Vec<Record>, Census)> {
        let pool = label_pool(self.wire.vocab, reserved)?;
        let mut all = Vec::new();
        let mut census = Census {
            labels: pool.iter().map(|label| (label.id, label.word)).collect(),
            ..Census::default()
        };
        for (i, family) in design.scoped.iter().enumerate() {
            let mut counts = FamilyCensus::default();
            for j in 0..SCOPE_CONTEXTS {
                let record_name = format!("J{}{:02}{j:02}", &design.split[..1], i);
                let labels = assigned_labels(&pool, i, j);
                let order = layout(j);
                let punctuation = if j % 2 == 0 { '.' } else { ';' };
                let facts = order
                    .map(|role| {
                        format!(
                            "{}{}",
                            design
                                .fact
                                .replace("{role}", ["first", "second", "third", "fourth"][role])
                                .replace("{word}", labels[role].word),
                            punctuation
                        )
                    })
                    .join(" ");
                let header = design.fact_header.replace("{name}", &record_name);
                let context = Context {
                    id: format!("jf-{}-{}", design.split, record_name),
                    facts: format!("{header} {facts}"),
                    question: design.question.replace("{name}", &record_name),
                    labels,
                    order,
                    punctuation,
                };
                let mut records = Vec::with_capacity(4);
                for operative in 0..2 {
                    for side in 0..2 {
                        let short = operative ^ side;
                        let directive = family.directive(operative, short);
                        records.push(context.record(
                            self.wire,
                            design,
                            family.name,
                            &directive,
                            0,
                            side as u8,
                            Some(&Cell { operative, short }),
                        )?);
                    }
                }
                self.check(&context, &mut records)?;
                counts.record(&records, &context);
                all.extend(records);
            }
            counts.verify(family.name, true, 0)?;
            census.scoped_quartets += counts.contexts;
            census.equal_length_quartets += counts.equal_length_contexts;
            census.equal_length_clause_order_pairs += counts.equal_length_clause_order_pairs;
            census.equal_length_selector_pairs += counts.equal_length_selector_pairs;
            census.scoped_families.insert(family.name, counts);
        }
        for (i, family) in design.direct.iter().enumerate() {
            let mut counts = FamilyCensus::default();
            for j in 0..DIRECT_CONTEXTS {
                let record_name = format!("J{}D{i:02}{j:02}", &design.split[..1]);
                let labels = assigned_labels(&pool, i + 8 + j / 8, j);
                let words = labels.each_ref().map(|label| label.word).join(", ");
                let facts = format!(
                    "{}: {words}.",
                    family.listing.replace("{name}", &record_name)
                );
                let context = Context {
                    id: format!("jf-{}-{}", design.split, record_name),
                    facts,
                    question: family.question.to_owned(),
                    labels,
                    order: [0, 1, 2, 3],
                    punctuation: ',',
                };
                let mut records = Vec::with_capacity(2);
                for (side, directive) in family.directives.iter().enumerate() {
                    records.push(context.record(
                        self.wire,
                        design,
                        family.name,
                        directive,
                        family.contrast,
                        side as u8,
                        None,
                    )?);
                }
                self.check(&context, &mut records)?;
                counts.record(&records, &context);
                all.extend(records);
            }
            counts.verify(family.name, false, usize::from(family.contrast))?;
            census.equal_length_direct_pairs += counts.equal_length_contexts;
            if family.contrast == 0 {
                census.direct_one_two_pairs += counts.contexts;
            } else {
                census.direct_two_three_pairs += counts.contexts;
            }
            census.direct_families.insert(family.name, counts);
        }
        census.rows = all.len();
        census.prefix_min = all
            .iter()
            .map(|r| r.row.prefix.len())
            .min()
            .ok_or("empty panel")?;
        census.prefix_max = all
            .iter()
            .map(|r| r.row.prefix.len())
            .max()
            .ok_or("empty panel")?;
        require(
            census.scoped_quartets >= 50 && census.equal_length_quartets > 0,
            "insufficient scoped transfer coverage",
        )?;
        Ok((all, census))
    }
}

fn layout(index: usize) -> [usize; 4] {
    if index < 4 {
        [0, 1, 2, 3].map(|role| (role + index) % 4)
    } else {
        [3, 2, 1, 0].map(|role| (role + index) % 4)
    }
}

fn assigned_labels(pool: &[Label; 8], family: usize, index: usize) -> [Label; 4] {
    let offsets = [
        [0, 1, 3, 5],
        [0, 1, 5, 3],
        [0, 3, 1, 5],
        [0, 3, 5, 1],
        [0, 5, 1, 3],
        [0, 5, 3, 1],
        [0, 2, 4, 6],
        [0, 2, 6, 4],
        [0, 4, 2, 6],
        [0, 4, 6, 2],
        [0, 6, 2, 4],
        [0, 6, 4, 2],
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
    ];
    offsets[family].map(|offset| pool[(index + offset) % pool.len()].clone())
}
