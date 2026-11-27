use crate::*;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use std::collections::{BTreeMap, HashSet};
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};

use crate::rubric::{contains_words, fact_alias, polarity, score, words};

const DONOR_BANK: &str =
    include_str!("../../../../../../labs/m2-donor-corpus/instrument/src/bank.rs");
const MAX_TURN_INDEX: usize = 4;
const WINDOW_BUDGET: usize = 117;
const PROSE_PREFIX: usize = 127;
const PROSE_SCAN: usize = 4096;
const PROSE_MIN_NEW: usize = 8;

#[derive(Deserialize)]
struct Dataset {
    data: Vec<Conversation>,
}

#[derive(Deserialize)]
struct Conversation {
    id: String,
    filename: String,
    source: String,
    story: String,
    questions: Vec<Turn>,
    answers: Vec<Turn>,
    additional_answers: BTreeMap<String, Vec<Turn>>,
}

#[derive(Deserialize)]
struct Turn {
    input_text: String,
    turn_id: usize,
    #[serde(default)]
    span_text: String,
}

#[derive(Deserialize)]
pub(crate) struct Prior {
    pub(crate) id: String,
    domain: String,
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum Task {
    Yes,
    No,
    Fact,
}

impl Task {
    fn label(self) -> &'static str {
        match self {
            Self::Yes => "yes",
            Self::No => "no",
            Self::Fact => "fact",
        }
    }
}

struct Candidate {
    source: String,
    group: String,
    rank: [u8; 32],
    window: &'static str,
    task: Task,
    row: Teacher,
    judgment: Judgment,
}

#[derive(Default, Serialize)]
struct Census {
    licensed_conversations: usize,
    duplicate_passage: usize,
    donor_turn_excluded: usize,
    ambiguous_turn: usize,
    shape_excluded: usize,
    context_excluded: usize,
    eligible_candidates: usize,
    eligible_source_groups: usize,
    full_eligible_source_groups: usize,
    chosen_windows: BTreeMap<&'static str, usize>,
    chosen_cells: BTreeMap<String, usize>,
    retained_reference_pass: usize,
    retained_reference_total: usize,
    alternate_reference_pass: usize,
    alternate_reference_total: usize,
    prose_scanned: usize,
    prose_insufficient_or_non_prose: usize,
    prose_existing_article: usize,
}

pub(crate) fn bank_normalize(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn donor_hit(bank: &HashSet<String>, text: &str) -> bool {
    let digest = Sha256::digest(bank_normalize(text));
    bank.contains(&hex::encode(&digest[..8]))
}

fn read_bank(path: &Path) -> Result<HashSet<String>> {
    let mut rows = HashSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if line.len() != 16 || !line.bytes().all(|ch| ch.is_ascii_hexdigit()) || !rows.insert(line)
        {
            return Err("invalid or duplicate D87 digest".into());
        }
    }
    if rows.is_empty() {
        return Err("no D87 digests".into());
    }
    Ok(rows)
}

pub(crate) fn donor_continuations() -> Result<HashSet<String>> {
    let (lines, _) = DONOR_BANK
        .split_once("pub const FOLLOW_UPS: [&str; 40] = [\n")
        .and_then(|(_, rest)| rest.split_once("\n];"))
        .ok_or("D87 follow-up source changed")?;
    let normalized: Option<HashSet<_>> = lines
        .lines()
        .map(|line| {
            line.trim()
                .strip_prefix('"')?
                .strip_suffix("\",")
                .map(bank_normalize)
        })
        .collect();
    normalized
        .filter(|list| list.len() == 40)
        .ok_or_else(|| "D87 follow-ups not recovered".into())
}

fn teacher(prompt: Prompt, gold: Vec<u16>) -> Teacher {
    let mut seen: HashSet<_> = prompt.prefix.iter().copied().collect();
    let new_teacher_tokens = gold
        .iter()
        .map(|&token| token != EOS && seen.insert(token))
        .collect();
    Teacher {
        prompt,
        gold,
        new_teacher_tokens,
    }
}

fn conversation_row(
    id: String,
    domain: Domain,
    messages: &[Message<'_>],
    reply: &str,
    encoder: &DialogueEncoder<'_>,
    context: usize,
) -> Result<Option<Teacher>> {
    let prefix = encoder.encode_assistant_prefix(messages);
    if prefix.tokens.len() > context
        || prefix.special_token_occurrences != 0
        || prefix.emitted_turns != messages.len()
    {
        return Ok(None);
    }
    let declared_history = messages.iter().filter(|m| m.role == "assistant").count();
    let preserved_history = prefix
        .tokens
        .iter()
        .filter(|&&t| t == u32::from(EOS))
        .count();
    if declared_history == 0 || declared_history != preserved_history {
        return Err("assistant EOS missing from genuine prompt history".into());
    }
    let mut full = messages.to_vec();
    full.push(Message {
        role: "assistant",
        content: reply,
    });
    let complete = encoder.encode_document(&full);
    if complete.special_token_occurrences != 0 {
        return Ok(None);
    }
    let Some(gold) = complete.tokens.strip_prefix(prefix.tokens.as_slice()) else {
        return Err("full dialogue does not replay the exact open prompt".into());
    };
    if gold.len() < 2
        || gold.len() > CHAT_CAP
        || gold.last() != Some(&u32::from(EOS))
        || complete.response_mask[prefix.tokens.len()..]
            .iter()
            .any(|&m| m != 1)
    {
        return Ok(None);
    }
    Ok(Some(teacher(
        Prompt {
            id,
            domain,
            prefix: ids(&prefix.tokens)?,
            cap: CHAT_CAP,
        },
        ids(gold)?,
    )))
}

fn sentence_boundaries(text: &str) -> Vec<usize> {
    let mut result = vec![0];
    let mut preceding_terminal = false;
    let mut previous_newline = false;
    for (i, ch) in text.char_indices() {
        if !ch.is_whitespace()
            && (preceding_terminal || previous_newline)
            && text[..i].ends_with(char::is_whitespace)
        {
            result.push(i);
            preceding_terminal = false;
        }
        if ch == '.' || ch == '!' || ch == '?' {
            preceding_terminal = true;
        } else if !ch.is_whitespace() && !matches!(ch, '\'' | '"' | '\u{2019}' | '\u{201d}' | ')') {
            preceding_terminal = false;
        }
        previous_newline = ch == '\n';
    }
    result.push(text.len());
    result.sort_unstable();
    result.dedup();
    result
}

fn windows<'a>(story: &'a str, tokenizer: &ByteBpeTokenizer) -> Vec<(usize, usize, &'a str)> {
    // Every start and end uses only original passage bytes and the tokenizer.
    // Evidence and corresponding answer annotations never select substring boundaries.
    let bounds = sentence_boundaries(story);
    let mut out = Vec::new();
    for (i, &start) in bounds
        .iter()
        .enumerate()
        .take(bounds.len().saturating_sub(1))
    {
        let mut chosen = None;
        for &end in &bounds[i + 1..] {
            let text = story[start..end].trim();
            if tokenizer.encode(text).len() > WINDOW_BUDGET {
                break;
            }
            chosen = Some((start, end, text));
        }
        if let Some(window) = chosen {
            out.push(window);
        }
    }
    out
}

fn answer_for(
    history: &[Turn],
    refs: &[String],
    question: &str,
    evidence: &str,
) -> Option<(Task, Expected)> {
    let does_not_echo = !refs
        .iter()
        .any(|reference| words(reference) == words(question));
    if !does_not_echo {
        return None;
    }
    for (yes, task, expected) in [
        (true, Task::Yes, Expected::Yes),
        (false, Task::No, Expected::No),
    ] {
        if refs
            .iter()
            .all(|reference| polarity(reference) == Some(yes))
            && score(&refs[0], question, &expected) == "useful"
        {
            return Some((task, expected));
        }
    }
    let main = fact_alias(&refs[0])?;
    if refs
        .iter()
        .filter_map(|s| fact_alias(s))
        .filter(|v| *v == main)
        .count()
        < 3
    {
        return None;
    }
    // The consensus slot must be available inside the original human-marked evidence
    // and must not simply repeat an already provided assistant answer or the final query.
    let main_words = words(&refs[0]);
    let normalized = words(&main);
    let occurs = |container: &str| {
        let container = words(container);
        contains_words(&container, &main_words) || contains_words(&container, &normalized)
    };
    if !occurs(evidence) || history.iter().any(|turn| occurs(&turn.input_text)) || occurs(question)
    {
        return None;
    }
    let expected = Expected::Fact(vec![main]);
    (score(&refs[0], question, &expected) == "useful").then_some((Task::Fact, expected))
}

fn coqa(
    data: &Dataset,
    tokenizer: &ByteBpeTokenizer,
    encoder: &DialogueEncoder<'_>,
    bank: &HashSet<String>,
    context: usize,
    census: &mut Census,
) -> Result<Vec<Candidate>> {
    let follow_ups = donor_continuations()?;
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();
    let mut seen_passage = HashSet::new();
    for dialog in &data.data {
        if !matches!(dialog.source.as_str(), "wikipedia" | "gutenberg" | "cnn") {
            continue;
        }
        census.licensed_conversations += 1;
        let story = dialog.story.trim();
        if !seen.insert(&dialog.id) || !seen_passage.insert(bank_normalize(story)) {
            census.duplicate_passage += 1;
            continue;
        }
        if dialog.questions.len() != dialog.answers.len() || dialog.additional_answers.len() != 3 {
            return Err(format!(
                "mismatched CoQA turns or annotation counts in {}",
                dialog.id
            )
            .into());
        }
        let original_windows = windows(story, tokenizer);
        for n in 1..=MAX_TURN_INDEX.min(dialog.questions.len().saturating_sub(1)) {
            let question = &dialog.questions[n];
            let answer = &dialog.answers[n];
            if dialog.questions[..=n]
                .iter()
                .zip(&dialog.answers[..=n])
                .enumerate()
                .any(|(i, (q, a))| q.turn_id != i + 1 || a.turn_id != i + 1)
            {
                return Err(format!("mismatched CoQA history turn IDs in {}", dialog.id).into());
            }
            if dialog.questions[..=n].iter().any(|q| {
                donor_hit(bank, &q.input_text)
                    || follow_ups.contains(&bank_normalize(&q.input_text))
            }) {
                census.donor_turn_excluded += 1;
                continue;
            }
            if question.input_text.matches('?').count() != 1 || question.input_text.contains("<|") {
                census.shape_excluded += 1;
                continue;
            }
            let mut references = vec![answer.input_text.clone()];
            for annotation in dialog.additional_answers.values() {
                let annotated_turn = annotation
                    .get(n)
                    .filter(|a| a.turn_id == n + 1)
                    .ok_or("missing CoQA independent answer")?;
                references.push(annotated_turn.input_text.clone());
            }
            let Some((task, expected)) = answer_for(
                &dialog.answers[..n],
                &references,
                &question.input_text,
                &answer.span_text,
            ) else {
                census.ambiguous_turn += 1;
                continue;
            };
            let evidence = words(&answer.span_text);
            if evidence.is_empty() {
                census.ambiguous_turn += 1;
                continue;
            }
            let key = format!("{}:{}", dialog.id, n + 1);
            let id = opaque_id("coqa", &key);
            let make = |story: &str| -> Result<Option<Teacher>> {
                let passage = format!("Read this passage, then give a short direct answer to the question.\n\n{story}");
                let mut history = vec![Message {
                    role: "system",
                    content: &passage,
                }];
                for (q, a) in dialog.questions[..n].iter().zip(&dialog.answers[..n]) {
                    history.push(Message {
                        role: "user",
                        content: &q.input_text,
                    });
                    history.push(Message {
                        role: "assistant",
                        content: &a.input_text,
                    });
                }
                history.push(Message {
                    role: "user",
                    content: &question.input_text,
                });
                conversation_row(
                    id.clone(),
                    Domain::Conversation,
                    &history,
                    &answer.input_text,
                    encoder,
                    context,
                )
            };
            let full = make(story)?;
            let (row, mode, start, end) = if let Some(row) = full {
                (row, "full_passage", 0, story.len())
            } else {
                let mut eligible = Vec::new();
                for &(start, end, excerpt) in &original_windows {
                    let text = words(excerpt);
                    if !contains_words(&text, &evidence) {
                        continue;
                    }
                    if let Some(row) = make(excerpt)? {
                        eligible.push((
                            hash("passage-window", &format!("{key}:{start}:{end}")),
                            row,
                            start,
                            end,
                        ));
                    }
                }
                eligible.sort_by_key(|(rank, _, _, _)| *rank);
                let Some((_, row, start, end)) = eligible.into_iter().next() else {
                    census.context_excluded += 1;
                    continue;
                };
                (row, "preset_window_with_evidence_admission", start, end)
            };
            candidates.push(Candidate {
                source: dialog.source.clone(),
                group: dialog.id.clone(),
                rank: hash("candidate", &key),
                window: mode,
                task,
                row,
                judgment: Judgment {
                    id,
                    question: question.input_text.clone(),
                    answer: expected,
                    references,
                    custody: json!({
                        "dataset": "CoQA dev v1.0", "source": dialog.source,
                        "source_conversation_id": dialog.id, "passage_filename": dialog.filename,
                        "turn": n + 1, "passage_mode": mode, "passage_utf8_start": start,
                        "passage_utf8_end": end, "earlier_human_assistant_turns": n,
                    }),
                },
            });
        }
    }
    census.eligible_candidates = candidates.len();
    census.eligible_source_groups = candidates
        .iter()
        .map(|c| &c.group)
        .collect::<HashSet<_>>()
        .len();
    census.full_eligible_source_groups = candidates
        .iter()
        .filter(|c| c.window == "full_passage")
        .map(|c| &c.group)
        .collect::<HashSet<_>>()
        .len();
    Ok(candidates)
}

fn choose(mut items: Vec<Candidate>, census: &mut Census) -> Result<(Vec<Teacher>, Vec<Judgment>)> {
    items.sort_by_key(|c| (c.window != "full_passage", c.rank));
    let mut rows = Vec::new();
    let mut judgments = Vec::new();
    let mut seen = HashSet::new();
    for (task, quota) in [(Task::Yes, 24), (Task::No, 24), (Task::Fact, 152)] {
        let mut cell = 0;
        for candidate in &items {
            if candidate.task != task || seen.contains(&candidate.group) {
                continue;
            }
            seen.insert(candidate.group.clone());
            rows.push(candidate.row.prompt.clone());
            judgments.push(candidate.judgment.id.clone());
            cell += 1;
            if cell == quota {
                break;
            }
        }
        if cell != quota {
            return Err(format!(
                "CoQA {} quota underfilled at {cell}/{quota}; eligible groups {}, full groups {}, context rejections {}",
                task.label(), census.eligible_source_groups, census.full_eligible_source_groups, census.context_excluded
            ).into());
        }
    }
    let id_set: HashSet<_> = judgments.into_iter().collect();
    let mut selected: Vec<_> = items
        .into_iter()
        .filter(|c| id_set.contains(&c.judgment.id))
        .collect();
    if selected.len() != CHAT_ROWS || rows.len() != CHAT_ROWS {
        return Err("CoQA panel cardinality mismatch".into());
    }
    selected.sort_by(|a, b| a.judgment.id.cmp(&b.judgment.id));
    for c in &selected {
        *census.chosen_windows.entry(c.window).or_default() += 1;
        *census
            .chosen_cells
            .entry(format!("{}:{}", c.source, c.task.label()))
            .or_default() += 1;
        for (i, reference) in c.judgment.references.iter().enumerate() {
            let passed = score(reference, &c.judgment.question, &c.judgment.answer) == "useful";
            if i == 0 {
                census.retained_reference_pass += usize::from(passed);
                census.retained_reference_total += 1;
            } else {
                census.alternate_reference_pass += usize::from(passed);
                census.alternate_reference_total += 1;
            }
        }
    }
    Ok(selected.into_iter().map(|c| (c.row, c.judgment)).unzip())
}

const RECALL_KEYS: [&str; 4] = ["sample", "parcel", "registry", "archive"];

fn recall(encoder: &DialogueEncoder<'_>, context: usize) -> Result<(Vec<Teacher>, Vec<Judgment>)> {
    let mut rows = Vec::new();
    let mut grading = Vec::new();
    let mut values = HashSet::new();
    for pair in 0..RECALL_PAIRS {
        let key = RECALL_KEYS[pair % RECALL_KEYS.len()];
        let entry = |which: &str, values: &mut HashSet<u32>| {
            for retry in 0..1000 {
                let rank = hash("recall-value", &format!("{pair}:{which}:{retry}"));
                let value = 10000 + u32::from(u16::from_be_bytes([rank[0], rank[1]]));
                if values.insert(value) {
                    return Ok::<_, Box<dyn Error>>(value.to_string());
                }
            }
            Err("too many recall ID collisions".into())
        };
        let a = entry("a", &mut values)?;
        let b = entry("b", &mut values)?;
        let location = if pair % 2 == 0 { "north" } else { "south" };
        let question = format!("Which code did I give you for the {location} {key}?");
        let pair_id = opaque_id("recall-pair", &pair.to_string());
        for (version, target, alternate) in [(0, &a, &b), (1, &b, &a)] {
            let (north, south) = if location == "north" {
                (target, alternate)
            } else {
                (alternate, target)
            };
            let facts = format!("Remember these two {key} codes. The north code is {north}. The south code is {south}.");
            let messages = [
                Message {
                    role: "user",
                    content: &facts,
                },
                Message {
                    role: "assistant",
                    content: "I have both codes recorded under their respective entries.",
                },
                Message {
                    role: "user",
                    content: "Which way does the sun rise?",
                },
                Message {
                    role: "assistant",
                    content: "It rises in the east.",
                },
                Message {
                    role: "user",
                    content: &question,
                },
            ];
            let id = opaque_id("recall", &format!("{pair}:{version}"));
            let row = conversation_row(
                id.clone(),
                Domain::Recall,
                &messages,
                target,
                encoder,
                context,
            )?
            .ok_or("recall prompt failed exact context/EOS validation")?;
            rows.push(row);
            grading.push(Judgment {
                id, question: question.clone(), answer: Expected::Fact(vec![target.clone()]),
                references: vec![target.clone()],
                custody: json!({"source":"synthetic counterfactual recall control derived from the entity-recall fixture family",
                    "pair_id":pair_id,"version":version,"alternative":alternate}),
            });
        }
    }
    Ok((rows, grading))
}

fn column<'a>(row: &'a Row, field: &str) -> Result<&'a str> {
    match row
        .get_column_iter()
        .find(|(name, _)| name.as_str() == field)
    {
        Some((_, Field::Str(text))) => Ok(text),
        _ => Err(format!("missing Wikipedia string column {field}").into()),
    }
}

fn prose(
    path: &Path,
    data: &Dataset,
    prior: &[Prior],
    tokenizer: &ByteBpeTokenizer,
    census: &mut Census,
) -> Result<Vec<(Teacher, Value)>> {
    let prior: HashSet<_> = prior
        .iter()
        .filter(|r| r.domain == "prose")
        .filter_map(|r| r.id.strip_prefix("wikipedia-en:"))
        .collect();
    if prior.is_empty() {
        return Err("no prior Wikipedia IDs to exclude".into());
    }
    let coqa_titles: HashSet<_> = data
        .data
        .iter()
        .filter(|r| r.source == "wikipedia")
        .map(|r| bank_normalize(r.filename.trim_end_matches(".txt")))
        .collect();
    let reader = SerializedFileReader::new(File::open(path)?)?;
    let mut eligible = Vec::new();
    let mut seen = HashSet::new();
    for (i, row) in reader.get_row_iter(None)?.take(PROSE_SCAN).enumerate() {
        census.prose_scanned += 1;
        let row = row?;
        let source_id = column(&row, "id")?;
        let title = column(&row, "title")?;
        if prior.contains(source_id)
            || coqa_titles.contains(&bank_normalize(title))
            || !seen.insert(source_id.to_owned())
        {
            census.prose_existing_article += 1;
            continue;
        }
        let article = column(&row, "text")?;
        let mut blocks = article.split("\n\n").filter(|p| p.chars().count() >= 100);
        let Some(opening) = blocks.next() else {
            census.prose_insufficient_or_non_prose += 1;
            continue;
        };
        let snippet: String = opening.chars().take(4096).collect();
        let text = format!(" {title}\n\n{snippet}");
        let tokens = tokenizer.encode(&text);
        if tokens.len() < PROSE_PREFIX + PROSE_CAP
            || tokens[..PROSE_PREFIX + PROSE_CAP].iter().any(|&t| t < 3)
        {
            census.prose_insufficient_or_non_prose += 1;
            continue;
        }
        let Some(url) = column(&row, "url")
            .ok()
            .filter(|url| url.starts_with("https://en.wikipedia.org/"))
        else {
            census.prose_insufficient_or_non_prose += 1;
            continue;
        };
        let mut prefix = vec![BOS];
        prefix.extend(ids(&tokens[..PROSE_PREFIX])?);
        let gold = ids(&tokens[PROSE_PREFIX..PROSE_PREFIX + PROSE_CAP])?;
        let prefix_set: HashSet<_> = prefix.iter().copied().collect();
        if gold.iter().filter(|v| !prefix_set.contains(v)).count() < PROSE_MIN_NEW {
            census.prose_insufficient_or_non_prose += 1;
            continue;
        }
        let id = opaque_id("prose", source_id);
        let row = teacher(
            Prompt {
                id: id.clone(),
                domain: Domain::Prose,
                prefix,
                cap: PROSE_CAP,
            },
            gold,
        );
        eligible.push((
            hash("prose-rank", source_id),
            row,
            json!({
                "id":id,"source_article_id":source_id,"url":url,"title":title,
                "source_row":i,"source_shard":"20231101.en/train-00021-of-00041.parquet",
            }),
        ));
    }
    eligible.sort_by_key(|(rank, _, _)| *rank);
    if eligible.len() < PROSE_ROWS {
        return Err(format!("prose anchor underfilled at {}", eligible.len()).into());
    }
    Ok(eligible
        .into_iter()
        .take(PROSE_ROWS)
        .map(|(_, row, custody)| (row, custody))
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    tok: &Path,
    coqa_path: &Path,
    wiki: &Path,
    oasst: &Path,
    bank: &Path,
    prior_fit: &Path,
    prior_held: &Path,
    context: usize,
    out: &Path,
) -> Result<()> {
    if !(192..=2048).contains(&context) {
        return Err("configured context must be at least 192 and at most 2048".into());
    }
    let tokenizer = tokenizer(tok)?;
    check_pin(coqa_path, COQA_SHA)?;
    check_pin(wiki, WIKI_SHA)?;
    check_pin(bank, BANK_SHA)?;
    check_pin(
        prior_fit,
        "e5b1c14eb96686b7fdee3f78bd70634f79b114e2912d5c53aca635c791edcc52",
    )?;
    check_pin(
        prior_held,
        "d8630bf60f06cc65ff4b32732cf5b1b5aba2a2837008408a207e32b950d24ef0",
    )?;
    let data: Dataset = serde_json::from_reader(File::open(coqa_path)?)?;
    let bank_rows = read_bank(bank)?;
    let mut prior: Vec<Prior> = read_jsonl(prior_fit)?;
    prior.extend(read_jsonl::<Prior>(prior_held)?);
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let mut census = Census::default();
    let candidates = coqa(
        &data,
        &tokenizer,
        &encoder,
        &bank_rows,
        context,
        &mut census,
    )?;
    let (mut rows, mut judgments) = choose(candidates, &mut census)?;
    let (recall_rows, recall_judgments) = recall(&encoder, context)?;
    rows.extend(recall_rows);
    judgments.extend(recall_judgments);
    let (prose_rows, provenance): (Vec<_>, Vec<_>) =
        prose(wiki, &data, &prior, &tokenizer, &mut census)?
            .into_iter()
            .unzip();
    rows.extend(prose_rows);
    rows.sort_by(|a, b| a.prompt.id.cmp(&b.prompt.id));
    judgments.sort_by(|a, b| a.id.cmp(&b.id));
    let (open_prompts, open_sources, open_census) =
        crate::open::build(oasst, &prior, &bank_rows, &encoder, context)?;
    let mut prompts: Vec<_> = rows
        .iter()
        .map(|row| &row.prompt)
        .chain(&open_prompts)
        .collect();
    prompts.sort_by(|a, b| a.id.cmp(&b.id));
    if !prompts.windows(2).all(|w| w[0].id < w[1].id) {
        return Err("collision in opaque IDs".into());
    }

    let calibration = crate::grade::calibrate(&tokenizer, &rows, &judgments)?;
    fs::create_dir(out)?;
    write_jsonl(&out.join("prompts.jsonl"), prompts.iter().copied())?;
    write_jsonl(&out.join("teacher.jsonl"), &rows)?;
    write_jsonl(&out.join("judgments.jsonl"), &judgments)?;
    write_jsonl(&out.join("prose_sources.jsonl"), &provenance)?;
    write_jsonl(&out.join("open_sources.jsonl"), &open_sources)?;
    write_json(&out.join("calibration.json"), &calibration)?;
    let mut output_pins = BTreeMap::new();
    for name in [
        "prompts.jsonl",
        "teacher.jsonl",
        "judgments.jsonl",
        "prose_sources.jsonl",
        "open_sources.jsonl",
        "calibration.json",
    ] {
        output_pins.insert(name, bytes_sha(&out.join(name))?);
    }
    write_json(
        &out.join("receipt.json"),
        &json!({
            "schema":"uor-r4.visible-answer-panel/1","seed":SEED,"context_limit":context,
            "rows":{"conversation":CHAT_ROWS,"recall":RECALL_PAIRS * 2,"prose":PROSE_ROWS,"open":OPEN_ROWS},
            "chat_cap":CHAT_CAP,"prose_cap":PROSE_CAP,"open_cap":OPEN_CAP,
            "max_prompt":prompts.iter().map(|r| r.prefix.len()).max(),
            "source_pins":{"coqa":COQA_SHA,"wiki":WIKI_SHA,"oasst2":crate::open::source_pin(),"d87_prompt_digests":BANK_SHA,
                "tokenizer":TOKENIZER_SHA,"prior_fit":bytes_sha(prior_fit)?,"prior_held":bytes_sha(prior_held)?},
            "outputs":output_pins,"census":census,"open_census":open_census,
        }),
    )?;
    Ok(())
}
