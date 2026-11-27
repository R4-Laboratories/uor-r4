#![forbid(unsafe_code)]

mod archive;
mod design;

use archive::{normalized, words, Archive};
use design::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
const TOKENIZER_SHA256: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const EOS: u32 = 1;
const PERIOD: u32 = 16;
const QUESTION_MARK: u32 = 33;
const NEWLINE: u32 = 201;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const WRITER_LIMIT: usize = 192;
const CAP: usize = 5;

#[derive(Serialize)]
struct Row {
    row_id: String,
    pair_id: String,
    side: u8,
    contrast: u8,
    positions: [u32; 4],
    ids: [u32; 4],
    prefix: Vec<u32>,
    expected: Vec<u32>,
    family: &'static str,
}

#[derive(Clone)]
struct Label {
    id: u32,
    word: String,
}

struct Wire<'a> {
    vocab: &'a ByteBpeTokenizer,
    dialogue: DialogueEncoder<'a>,
}

struct Context {
    id: String,
    text: String,
    question: String,
    order: [usize; 4],
    punctuation: char,
    labels: [Label; 4],
    byte_offsets: [usize; 4],
}

struct Cell {
    operative: usize,
    short: usize,
}

impl Cell {
    fn side(&self) -> u8 {
        u8::from(self.operative != self.short)
    }
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if !condition {
        return Err(message.into().into());
    }
    Ok(())
}

fn hash(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    io::copy(&mut File::open(path)?, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(&line?)
                .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1).into())
        })
        .collect()
}

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn jsonl(output: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    writeln!(output)?;
    Ok(())
}

fn verify_manifest(dir: &Path) -> Result<()> {
    let manifest = dir.join("SHA256SUMS");
    for line in fs::read_to_string(&manifest)?.lines() {
        let (expected, relative) = line
            .split_once("  ")
            .ok_or("invalid checksum manifest entry")?;
        require(
            hash(&dir.join(relative))? == expected,
            format!("{} checksum changed: {relative}", manifest.display()),
        )?;
    }
    Ok(())
}

fn write_manifest(dir: &Path, members: &[&str]) -> Result<()> {
    let mut manifest = writer(&dir.join("SHA256SUMS"))?;
    for member in members {
        writeln!(manifest, "{}  {member}", hash(&dir.join(member))?)?;
    }
    manifest.flush()?;
    verify_manifest(dir)
}

fn verify_design(archive: &Archive) -> Result<BTreeSet<String>> {
    let mut templates = BTreeSet::from([FACT.to_owned(), QUESTION.to_owned(), FORMAT.to_owned()]);
    for family in &SCOPED {
        templates.insert(family.name.to_owned());
        templates.extend(family.clauses.map(str::to_owned));
        templates.extend(family.selectors.map(str::to_owned));
        require(
            family.selectors.iter().all(|selector| {
                selector.matches("{a}").count() == 1
                    && selector.matches("{b}").count() == 1
                    && selector.find("{a}") < selector.find("{b}")
            }),
            format!("{} has invalid scope slots", family.name),
        )?;
        let rendered: BTreeSet<_> = (0..2)
            .flat_map(|operative| (0..2).map(move |short| family.directive(operative, short)))
            .collect();
        require(
            rendered.len() == 4,
            format!("{} has a duplicate scope cell", family.name),
        )?;
        templates.extend(rendered);
    }
    for family in &DIRECT {
        templates.insert(family.name.to_owned());
        templates.extend(family.directives.map(str::to_owned));
    }
    for text in &templates {
        archive.check_template(text)?;
    }
    Ok(templates)
}

fn label_pool(
    vocab: &ByteBpeTokenizer,
    archive: &Archive,
    templates: &BTreeSet<String>,
) -> Result<[Label; 8]> {
    let mut reserved = archive.labels.clone();
    for text in templates {
        reserved.extend(vocab.encode(text));
        reserved.extend(vocab.encode(&format!(" {text}")));
    }
    // Each selected label must encode and decode to a whole, space-prefixed word.
    let candidates = "steel bridge wolf coast thunder crow shark book wool goat iron hat gold
        crown fish care room light rock ring flower table warm shore store ground stone root
        wind rain boot trail tent wool fog note needle salt tea clock tower face space glove
        rope pearl hall north noon fruit glass scale grass sport iron lake map pond needle
        coil fence leaf mist flame cold warm age cave page sound drum mask bottle shell
        card paint wire floor train car plane plant cloud linen shop rail soap bowl silver
        green silver chest pocket mouth beach tooth fields beans bacon grain bread milk
        shoulder diamond canvas desk carpet mirror brush blade rope horse mouse lion deer
        ship board wheels bottles engine hammer needle pocket thread spin grass fence
        earth scent heat track dust moon solar paper plate metal wood beam stool cart
        square frame tape work play song screen clay cake river sand pearl violin cotton
        spice dream mountain garden darkness animal skin dust age net rule power station";
    let mut pool = Vec::new();
    for word in candidates.split_whitespace() {
        let text = format!(" {word}");
        if let [id] = vocab.encode(&text).as_slice() {
            if reserved.insert(*id) && vocab.decode_bytes(&[*id]) == text.as_bytes() {
                pool.push(Label {
                    id: *id,
                    word: word.to_owned(),
                });
            }
        }
        if pool.len() == 8 {
            break;
        }
    }
    let available = pool.len();
    pool.try_into()
        .map_err(|_| format!("only {available} novel whole-word labels").into())
}

fn context(family: usize, layout: usize, pool: &[Label; 8]) -> Result<Context> {
    let name = format!("RF{family:02}{layout:02}");
    let question = QUESTION.replace("{name}", &name);
    let order = ORDERS[layout];
    let punctuation = if (layout / 2 + family) % 2 == 0 {
        '.'
    } else {
        ';'
    };
    let labels = LABEL_OFFSETS[family].map(|offset| pool[(layout + offset) % pool.len()].clone());
    let mut byte_offsets = [0; 4];
    let mut text = String::new();
    let (before, after) = FACT.split_once("{word}").ok_or("missing fact value")?;
    for role in order {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(
            &before
                .replace("{name}", &name)
                .replace("{role}", ROLES[role]),
        );
        require(text.ends_with(' '), "source lacks its leading space")?;
        byte_offsets[role] = text.len();
        text.push_str(&labels[role].word);
        text.push_str(after);
        text.push(punctuation);
    }
    let digest = format!(
        "{:x}",
        Sha256::digest(format!("codex-4324\n{text}\n{question}\n{FORMAT}"))
    );
    let id = format!("rf-{}", &digest[..24]);
    Ok(Context {
        id,
        text,
        question,
        order,
        punctuation,
        labels,
        byte_offsets,
    })
}

impl Wire<'_> {
    fn encode(
        &self,
        context: &Context,
        user: &str,
        reply: &str,
        expected: &[u32],
    ) -> Result<(Vec<u32>, [u32; 4])> {
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
            .ok_or("dialogue marker drift")?;
        let fences: Vec<_> = body
            .windows(2)
            .enumerate()
            .filter_map(|(i, ids)| (ids == [NEWLINE, NEWLINE]).then_some(i))
            .collect();
        let questions: Vec<_> = body
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == QUESTION_MARK).then_some(i))
            .collect();
        let ([fence], [question]) = (fences.as_slice(), questions.as_slice()) else {
            return Err(format!("unexpected fence or question boundary: {user}").into());
        };
        let user_wire = format!(" {user}");
        let (ordinary, fallbacks) = self.vocab.encode_lossy(&user_wire);
        require(
            *question > fence + 1
                && open.tokens.len() + CAP < WRITER_LIMIT
                && open.special_token_occurrences == 0
                && complete.special_token_occurrences == 0
                && open.emitted_turns == 1
                && complete.emitted_turns == 2
                && !open.tokens.contains(&EOS)
                && fallbacks == 0
                && ordinary == body
                && self.vocab.decode_bytes(body) == user_wire.as_bytes()
                && self.vocab.decode_bytes(&body[..*fence])
                    == format!(" {}", context.text).as_bytes()
                && complete.tokens.strip_prefix(open.tokens.as_slice()) == Some(expected)
                && complete.response_mask.get(open.tokens.len()..)
                    == Some(vec![1; expected.len()].as_slice())
                && self.vocab.decode_bytes(&expected[..expected.len() - 1])
                    == format!(" {reply}").as_bytes(),
            format!("text, budget or reference drift for {user}"),
        )?;
        let mut positions = [0; 4];
        for (role, label) in context.labels.iter().enumerate() {
            let hits: Vec<_> = open
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(i, &id)| (id == label.id).then_some(i))
                .collect();
            let [position] = hits.as_slice() else {
                return Err(format!("{} occurs {} times in {user}", label.word, hits.len()).into());
            };
            require(
                (head.len()..head.len() + fence).contains(position)
                    && self
                        .vocab
                        .decode_bytes(&open.tokens[head.len()..*position])
                        .len()
                        == context.byte_offsets[role]
                    && self.vocab.decode_bytes(&[label.id])
                        == format!(" {}", label.word).as_bytes(),
                format!(
                    "wrong physical source occurrence for {} in {user}",
                    label.word
                ),
            )?;
            positions[role] = (*position).try_into()?;
        }
        Ok((open.tokens, positions))
    }
}

struct Request<'a> {
    family: &'static str,
    context: &'a Context,
    side: u8,
    contrast: u8,
    directive: String,
    cell: Option<Cell>,
    quoted: bool,
}

impl Request<'_> {
    fn encode(&self, wire: &Wire<'_>) -> Result<(Row, serde_json::Value)> {
        let context = self.context;
        let suffix_count = usize::from(self.contrast + self.side);
        let ids = std::array::from_fn(|role| context.labels[role].id);
        let mut expected = vec![ids[0], PERIOD];
        expected.extend_from_slice(&ids[1..1 + suffix_count]);
        expected.push(EOS);
        let mut reply = format!("{}.", context.labels[0].word);
        for label in &context.labels[1..1 + suffix_count] {
            reply.push(' ');
            reply.push_str(&label.word);
        }
        let user = format!(
            "{}\n\n{} {FORMAT} {}",
            context.text, context.question, self.directive
        );
        let (prefix, positions) = wire.encode(context, &user, &reply, &expected)?;
        let selector = self.cell.as_ref().map(|cell| cell.operative + 1);
        let clause_order = self.cell.as_ref().map(|cell| cell.short + 1);
        let pair_id = selector
            .map(|selector| format!("{}-s{selector}", context.id))
            .unwrap_or_else(|| format!("{}-d", context.id));
        let row_id = format!("{pair_id}:{}", self.side);
        let keys = json!({
            "row_id": row_id, "pair_id": pair_id, "context_id": context.id,
            "side": self.side, "contrast": self.contrast, "family": self.family,
            "pair_axis": if selector.is_some() { "clause-order-swap" } else { "direct-count-control" },
            "selector_swap_id": clause_order.map(|short| format!("{}-o{short}", context.id)),
            "operative_clause_ordinal": selector, "one_word_clause_ordinal": clause_order,
            "expected_word_count": suffix_count + 1,
            "expected_first_period_action": if suffix_count == 0 { "stop" } else { "copy" },
            "expected_post_b_action": match suffix_count { 0 => None, 1 => Some("stop"), _ => Some("copy") },
            "context": context.text, "question": context.question, "directive": self.directive,
            "user": user, "reply": reply, "quoted": self.quoted, "fact_role_order": context.order,
            "fact_punctuation": if context.punctuation == '.' { "period" } else { "semicolon" },
        });
        Ok((
            Row {
                row_id,
                pair_id,
                side: self.side,
                contrast: self.contrast,
                positions,
                ids,
                prefix,
                expected,
                family: self.family,
            },
            keys,
        ))
    }
}

fn check_matched(rows: &[Row]) -> Result<()> {
    let first = &rows[0];
    let question = first
        .prefix
        .iter()
        .position(|&id| id == QUESTION_MARK)
        .ok_or("missing question mark")?;
    require(
        rows.iter().all(|row| {
            row.ids == first.ids
                && row.positions == first.positions
                && row.family == first.family
                && row.prefix.get(..=question) == Some(&first.prefix[..=question])
        }),
        "fact, position, family or question drift within context",
    )?;
    for pair in rows.chunks_exact(2) {
        require(
            pair[0].pair_id == pair[1].pair_id
                && pair[0].side == 0
                && pair[1].side == 1
                && pair[0].contrast == pair[1].contrast
                && pair[0].expected[0..pair[0].expected.len() - 1]
                    == pair[1].expected[0..pair[0].expected.len() - 1],
            "non-matching response prefix or side order",
        )?;
    }
    Ok(())
}

#[derive(Default, Serialize)]
struct Census {
    contexts: usize,
    rows: usize,
    cells: [[usize; 2]; 2],
    word_counts: [usize; 3],
    punctuation: [usize; 2],
    a_before_b: usize,
    b_before_a: usize,
    labels_by_role: BTreeMap<u32, [usize; 4]>,
    roles_by_slot: [[usize; 4]; 4],
    order_edges_equal_tokens: usize,
    selector_edges_equal_tokens: usize,
    all_cells_equal_tokens: usize,
}

impl Census {
    fn record(&mut self, context: &Context, rows: &[Row]) {
        self.contexts += 1;
        self.rows += rows.len();
        for (role, label) in context.labels.iter().enumerate() {
            self.labels_by_role.entry(label.id).or_default()[role] += 1;
        }
        for (slot, &role) in context.order.iter().enumerate() {
            self.roles_by_slot[role][slot] += 1;
        }
        self.punctuation[usize::from(context.punctuation == ';')] += 1;
        if rows[0].positions[0] < rows[0].positions[1] {
            self.a_before_b += 1;
        } else {
            self.b_before_a += 1;
        }
        for row in rows {
            self.word_counts[usize::from(row.side + row.contrast)] += 1;
        }
        let length = |i: usize| rows[i].prefix.len();
        self.order_edges_equal_tokens += usize::from(length(0) == length(1));
        if rows.len() == 4 {
            for cell in self.cells.iter_mut().flatten() {
                *cell += 1;
            }
            self.order_edges_equal_tokens += usize::from(length(2) == length(3));
            // Rows are S1 STOP, S1 COPY, S2 STOP, S2 COPY. The opposite selector uses the opposite side.
            self.selector_edges_equal_tokens +=
                usize::from(length(0) == length(3)) + usize::from(length(1) == length(2));
            self.all_cells_equal_tokens += usize::from((1..4).all(|i| length(i) == length(0)));
        }
    }

    fn verify(&self, family: &str, contrast: u8, scoped: bool) -> Result<()> {
        let mut expected_counts = [0; 3];
        let count = if scoped { 16 } else { 8 };
        expected_counts[usize::from(contrast)] = count;
        expected_counts[usize::from(contrast) + 1] = count;
        require(
            self.contexts == 8
                && self.rows == 2 * count
                && self.cells == if scoped { [[8; 2]; 2] } else { [[0; 2]; 2] }
                && self.word_counts == expected_counts
                && self.punctuation == [4; 2]
                && self.a_before_b == 4
                && self.b_before_a == 4
                && self.labels_by_role.len() == 8
                && self.labels_by_role.values().all(|roles| *roles == [1; 4])
                && self.roles_by_slot == [[2; 4]; 4],
            format!("unbalanced coverage for {family}"),
        )
    }
}

struct Panel<'a> {
    wire: &'a Wire<'a>,
    archive: &'a Archive,
    rows: BufWriter<File>,
    keys: BufWriter<File>,
    unique_contexts: BTreeSet<String>,
    unique_role_tuples: BTreeSet<[u32; 4]>,
    unique_rows: BTreeSet<String>,
    unique_prompts: BTreeSet<String>,
    unique_prefixes: BTreeSet<Vec<u32>>,
    directive_words: BTreeSet<String>,
    prefix_min: usize,
    prefix_max: usize,
}

impl Panel<'_> {
    fn add_context(&mut self, requests: Vec<Request<'_>>, census: &mut Census) -> Result<()> {
        let context = requests.first().ok_or("empty context")?.context;
        let text = normalized(&context.text);
        require(
            requests
                .iter()
                .all(|request| request.context.id == context.id)
                && !self.archive.contexts.contains(&text)
                && self.unique_contexts.insert(text)
                && self
                    .unique_role_tuples
                    .insert(context.labels.each_ref().map(|label| label.id)),
            "reused context or ordered source tuple",
        )?;
        let mut rows = Vec::new();
        for request in requests {
            let (row, key) = request.encode(self.wire)?;
            let prompt = normalized(key["user"].as_str().ok_or("missing canonical user text")?);
            require(
                !self.archive.prompts.contains(&prompt)
                    && !self.archive.prefixes.contains(&row.prefix)
                    && !self.archive.pair_ids.contains(&row.pair_id)
                    && self.unique_rows.insert(row.row_id.clone())
                    && self.unique_prompts.insert(prompt)
                    && self.unique_prefixes.insert(row.prefix.clone()),
                format!("reused prompt, prefix or identity at {}", row.row_id),
            )?;
            self.directive_words.extend(words(&request.directive));
            self.prefix_min = self.prefix_min.min(row.prefix.len());
            self.prefix_max = self.prefix_max.max(row.prefix.len());
            jsonl(&mut self.rows, &row)?;
            jsonl(&mut self.keys, &key)?;
            rows.push(row);
        }
        check_matched(&rows)?;
        census.record(context, &rows);
        Ok(())
    }
}

fn prepare(runs: &Path, root: &Path) -> Result<()> {
    let tokenizer = runs.join(TOKENIZER);
    require(
        hash(&tokenizer)? == TOKENIZER_SHA256,
        "tokenizer identity drift",
    )?;
    let vocab = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?)
        .ok_or("invalid tokenizer")?;
    require(
        vocab.vocab_size() == 4096
            && vocab.encode(".") == [PERIOD]
            && vocab.encode("?") == [QUESTION_MARK]
            && vocab.encode("\n\n") == [NEWLINE, NEWLINE],
        "canonical special token identity drift",
    )?;
    let protocol = DialogueProtocol::literal_roles_v2(&vocab)?;
    let wire = Wire {
        vocab: &vocab,
        dialogue: protocol.bind(&vocab)?,
    };
    let archive = Archive::read(runs, &vocab)?;
    let templates = verify_design(&archive)?;
    let pool = label_pool(&vocab, &archive, &templates)?;
    fs::create_dir(root)?;
    let dir = root.join("fresh");
    fs::create_dir(&dir)?;
    let mut panel = Panel {
        wire: &wire,
        archive: &archive,
        rows: writer(&dir.join("rows.jsonl"))?,
        keys: writer(&dir.join("keys.jsonl"))?,
        unique_contexts: BTreeSet::new(),
        unique_role_tuples: BTreeSet::new(),
        unique_rows: BTreeSet::new(),
        unique_prompts: BTreeSet::new(),
        unique_prefixes: BTreeSet::new(),
        directive_words: BTreeSet::new(),
        prefix_min: usize::MAX,
        prefix_max: 0,
    };
    let mut scoped_census = BTreeMap::new();
    for (family_index, family) in SCOPED.iter().enumerate() {
        let mut census = Census::default();
        for layout in 0..ORDERS.len() {
            let context = context(family_index, layout, &pool)?;
            let requests = (0..2)
                .flat_map(|operative| {
                    let context = &context;
                    (0..2).map(move |side| {
                        let cell = Cell {
                            operative,
                            short: operative ^ side,
                        };
                        Request {
                            family: family.name,
                            context,
                            side: cell.side(),
                            contrast: 0,
                            directive: family.directive(cell.operative, cell.short),
                            cell: Some(cell),
                            quoted: family.quoted,
                        }
                    })
                })
                .collect();
            panel.add_context(requests, &mut census)?;
        }
        census.verify(family.name, 0, true)?;
        scoped_census.insert(family.name, census);
    }
    let mut direct_census = BTreeMap::new();
    for (family_index, family) in DIRECT.iter().enumerate() {
        let mut census = Census::default();
        for layout in 0..ORDERS.len() {
            let context = context(SCOPED.len() + family_index, layout, &pool)?;
            let requests = family
                .directives
                .iter()
                .enumerate()
                .map(|(side, directive)| Request {
                    family: family.name,
                    context: &context,
                    side: side as u8,
                    contrast: family.contrast,
                    directive: (*directive).to_owned(),
                    cell: None,
                    quoted: false,
                })
                .collect();
            panel.add_context(requests, &mut census)?;
        }
        census.verify(family.name, family.contrast, false)?;
        direct_census.insert(family.name, census);
    }
    panel.rows.flush()?;
    panel.keys.flush()?;
    let scoped_rows: usize = scoped_census.values().map(|census| census.rows).sum();
    let direct_rows: usize = direct_census.values().map(|census| census.rows).sum();
    require(
        scoped_rows >= 200
            && scoped_census.len() >= 8
            && panel.unique_rows.len() == scoped_rows + direct_rows,
        "panel misses scoped row, wording or identity coverage",
    )?;
    let shared_directive_words: BTreeMap<_, Vec<_>> = archive
        .directive_words
        .iter()
        .map(|(run, known)| (run, known.intersection(&panel.directive_words).collect()))
        .collect();
    let mut receipt = writer(&dir.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema": "codex-4324.release-factorial-panel/1",
            "split": "fresh",
            "authorship": "Codex agent-authored wording and fact frames; deterministic synthetic instantiation",
            "cap": CAP,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "archives": archive.roots,
            "archive_rows_checked": archive.rows,
            "archive_unique_template_signatures_checked": archive.templates.len(),
            "new_unique_templates_checked": templates.len(),
            "shared_directive_word_types": shared_directive_words,
            "archived_exact_overlap": { "texts": 0, "encoded_prompts": 0, "contexts": 0, "template_signatures": 0, "source_label_ids": 0 },
            "rows": panel.unique_rows.len(),
            "contexts": panel.unique_contexts.len(),
            "scoped_rows": scoped_rows,
            "scoped_quartets": scoped_rows / 4,
            "direct_rows": direct_rows,
            "direct_pairs": direct_rows / 2,
            "scoped_families": scoped_census,
            "direct_families": direct_census,
            "source_role_order": ["A", "B", "C", "D"],
            "labels": pool.iter().map(|label| (label.id, &label.word)).collect::<BTreeMap<_, _>>(),
            "prefix_min": panel.prefix_min,
            "prefix_max": panel.prefix_max,
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    write_manifest(&dir, &["rows.jsonl", "keys.jsonl", "receipt.json"])?;
    write_manifest(root, &["fresh/SHA256SUMS"])
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [mode, runs, out] if mode == "prepare" => prepare(Path::new(runs), Path::new(out)),
        _ => Err("usage: release-factorial-panel prepare RUNS NEW_PANEL".into()),
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
