#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::{self, File};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Counts = BTreeMap<&'static str, usize>;

const TRAIN: &str = "squad-train-v2.0.json";
const DEV: &str = "squad-dev-v2.0.json";
const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
const COQA: &str = "20261127-codex-visible-answer/panel/inputs/coqa-dev-v1.0.json";
const BANK: &str = "20261127-codex-visible-answer/panel/inputs/d87-bank-prompt-digests.txt";
const PRIOR: &str = "20261127-codex-shared-source/panel-v1/human/keys.jsonl";
const PINS: [(&str, &str); 6] = [
    (
        TRAIN,
        "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002",
    ),
    (
        DEV,
        "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    ),
    (
        TOKENIZER,
        "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377",
    ),
    (
        COQA,
        "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a",
    ),
    (
        BANK,
        "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541",
    ),
    (
        PRIOR,
        "fcd87c28afde5a9e1dd9284d31e4630fe28535c8cde475e153b669cfc6f9f231",
    ),
];
const PREFIX_LIMIT: usize = 180;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const SHINGLE_WORDS: usize = 16;
const TRAIN_CAP: usize = 320;
const DEV_CAP: usize = 200;
const RESERVE_CAP: usize = 200;
const ARTICLE_CAP: usize = 10;

#[derive(Deserialize)]
struct Dataset {
    version: String,
    data: Vec<Article>,
}

#[derive(Deserialize)]
struct Article {
    title: String,
    paragraphs: Vec<Paragraph>,
}

#[derive(Deserialize)]
struct Paragraph {
    context: String,
    qas: Vec<Question>,
}

#[derive(Deserialize)]
struct Question {
    id: String,
    question: String,
    is_impossible: bool,
    answers: Vec<Annotation>,
    #[serde(default)]
    plausible_answers: Vec<Annotation>,
}

#[derive(Deserialize)]
struct Annotation {
    text: String,
    answer_start: usize,
}

#[derive(Deserialize)]
struct Coqa {
    data: Vec<CoqaSource>,
}

#[derive(Deserialize)]
struct CoqaSource {
    source: String,
    filename: String,
    story: String,
}

#[derive(Deserialize)]
struct PriorKey {
    user: String,
    question: String,
}

struct Blocklist {
    titles: HashSet<String>,
    passages: HashSet<String>,
    shingles: HashSet<u64>,
    prior_questions: HashSet<String>,
    prior_prompts: HashSet<String>,
    bank: HashSet<String>,
}

#[derive(Serialize)]
struct Label {
    /// Original character offset and half-open UTF-8 byte range of the official span.
    char_start: usize,
    bytes: [usize; 2],
    text: String,
    /// Original ID and zero-based occurrence in the untruncated prefix.
    first_id: u32,
    first_position: usize,
    /// All original IDs touching the span. The final token can extend beyond it.
    source_ids: Vec<u32>,
    whole_span_ids: bool,
    /// Original protocol-2 answer IDs, including a leading space ID if emitted.
    response_ids: Vec<u32>,
    response_lexical_offset: usize,
    agreement: usize,
}

struct Candidate<'a> {
    question: &'a Question,
    body: String,
    question_key: String,
    input: Vec<u32>,
    user_bounds: [usize; 2],
    label: Label,
}

struct Pair<'a> {
    article: &'a str,
    paragraph: usize,
    passage_sha256: String,
    positive: Candidate<'a>,
    negative: Candidate<'a>,
    same_span: bool,
    same_first_position: bool,
    question_edit: usize,
}

#[derive(Serialize)]
struct Prompt<'a> {
    case_id: &'a str,
    prompt: &'a str,
}

#[derive(Serialize)]
struct Truth<'a> {
    case_id: &'a str,
    paired_with: &'a str,
    article: &'a str,
    paragraph: usize,
    passage_sha256: &'a str,
    question: &'a str,
    is_impossible: bool,
    prefix: &'a [u32],
    user_bounds: [usize; 2],
    gold_source: Option<&'a Label>,
    plausible_only: Option<&'a Label>,
    same_annotated_span: bool,
    same_first_position: bool,
    question_edit: usize,
}

fn normalize(text: &str) -> String {
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

fn shingles(normalized: &str) -> impl Iterator<Item = u64> + '_ {
    let words: Vec<_> = normalized.split_whitespace().collect();
    {
        let hashes: Vec<_> = words
            .windows(SHINGLE_WORDS)
            .map(|window| {
                let mut hasher = DefaultHasher::new();
                window.hash(&mut hasher);
                hasher.finish()
            })
            .collect();
        hashes.into_iter()
    }
}

fn count(counts: &mut Counts, name: &'static str, value: usize) {
    *counts.entry(name).or_default() += value;
}

fn bad_input(message: &str) -> Box<dyn Error> {
    message.to_owned().into()
}

fn checked_bytes(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if hex::encode(Sha256::digest(&bytes)) != expected {
        return Err(format!("digest mismatch: {}", path.display()).into());
    }
    Ok(bytes)
}

fn jsonl<T: Serialize>(output: &mut BufWriter<File>, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    writeln!(output)?;
    Ok(())
}

impl Blocklist {
    fn load(runs: &Path) -> Result<Self> {
        let coqa: Coqa = serde_json::from_slice(&checked_bytes(&runs.join(COQA), PINS[3].1)?)?;
        let mut list = Self {
            titles: HashSet::new(),
            passages: HashSet::new(),
            shingles: HashSet::new(),
            prior_questions: HashSet::new(),
            prior_prompts: HashSet::new(),
            bank: HashSet::new(),
        };
        for document in coqa.data {
            if document.source == "wikipedia" {
                let filename = document.filename.trim_end_matches(".txt");
                list.titles.insert(normalize(filename));
                if let Some((stem, _)) = filename.rsplit_once('_') {
                    list.titles.insert(normalize(stem));
                }
            }
            let story = normalize(&document.story);
            list.shingles.extend(shingles(&story));
            list.passages.insert(story);
        }
        let prior = checked_bytes(&runs.join(PRIOR), PINS[5].1)?;
        for line in BufReader::new(prior.as_slice()).lines() {
            let key: PriorKey = serde_json::from_str(&line?)?;
            list.prior_questions.insert(normalize(&key.question));
            list.prior_prompts.insert(normalize(&key.user));
        }
        let bank = checked_bytes(&runs.join(BANK), PINS[4].1)?;
        for line in BufReader::new(bank.as_slice()).lines() {
            list.bank.insert(line?);
        }
        Ok(list)
    }

    fn blocked_prompt(&self, normalized_question: &str, user: &str) -> bool {
        let normalized_user = normalize(user);
        self.prior_questions.contains(normalized_question)
            || self.prior_prompts.contains(&normalized_user)
            || [normalized_question, &normalized_user].iter().any(|value| {
                self.bank
                    .contains(&hex::encode(&Sha256::digest(value)[..8]))
            })
    }
}

fn verified_annotation<'a>(
    paragraph: &Paragraph,
    question: &'a Question,
    is_dev: bool,
) -> std::result::Result<(&'a Annotation, [usize; 2], usize), &'static str> {
    if question.is_impossible && !question.answers.is_empty() {
        return Err("impossible_with_gold");
    }
    let annotations = if question.is_impossible {
        &question.plausible_answers
    } else {
        &question.answers
    };
    let mut spans = BTreeMap::new();
    for annotation in annotations {
        if annotation.text.is_empty() || annotation.text.trim() != annotation.text {
            continue;
        }
        let Some((start, _)) = paragraph
            .context
            .char_indices()
            .nth(annotation.answer_start)
        else {
            continue;
        };
        let end = start + annotation.text.len();
        if paragraph.context.get(start..end) != Some(&annotation.text) {
            continue;
        }
        let entry = spans.entry([start, end]).or_insert((annotation, 0));
        entry.1 += 1;
    }
    let Some((bytes, (annotation, agreement))) = spans
        .into_iter()
        .max_by_key(|(bytes, (_, agreement))| (*agreement, std::cmp::Reverse(*bytes)))
    else {
        return Err(if question.is_impossible {
            "missing_exact_plausible"
        } else {
            "missing_exact_gold"
        });
    };
    if !question.is_impossible && is_dev && (agreement < 2 || 2 * agreement <= annotations.len()) {
        return Err("dev_gold_without_majority");
    }
    if annotation.text.split_whitespace().count() > 5
        || !annotation
            .text
            .chars()
            .next()
            .is_some_and(char::is_alphanumeric)
        || !annotation
            .text
            .chars()
            .last()
            .is_some_and(char::is_alphanumeric)
    {
        return Err("span_not_short_lexical");
    }
    Ok((annotation, bytes, agreement))
}

struct Render<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    user_head: Vec<u32>,
}

impl<'a> Render<'a> {
    fn new(tokenizer: &'a ByteBpeTokenizer) -> Result<Self> {
        let protocol = DialogueProtocol::literal_roles_v2(tokenizer)?;
        let encoder = protocol.bind(tokenizer)?;
        let user_head = [vec![0], tokenizer.encode("User:")].concat();
        if [
            tokenizer.encode("\n"),
            tokenizer.encode(protocol.assistant_marker()),
        ]
        .concat()
            != MARKER
        {
            return Err(bad_input("protocol-2 marker identity drift"));
        }
        Ok(Self {
            tokenizer,
            encoder,
            user_head,
        })
    }

    fn candidate<'b>(
        &self,
        paragraph: &Paragraph,
        question: &'b Question,
        is_dev: bool,
        blocked: &Blocklist,
        dev_questions: &HashSet<String>,
    ) -> std::result::Result<Candidate<'b>, &'static str> {
        if paragraph.context.trim() != paragraph.context
            || paragraph.context.contains(['\n', '\r'])
            || question.question.trim() != question.question
            || question.question.contains(['\n', '\r'])
        {
            return Err("not_verbatim_single_paragraph");
        }
        let question_key = normalize(&question.question);
        if question_key.split_whitespace().count() < 4 || !question.question.ends_with('?') {
            return Err("not_self_contained_question_shape");
        }
        let body = format!("{} {}", paragraph.context, question.question);
        if blocked.blocked_prompt(&question_key, &body) || dev_questions.contains(&question_key) {
            return Err("prior_question_or_prompt");
        }
        let (annotation, bytes, agreement) = verified_annotation(paragraph, question, is_dev)?;
        let rendered_body = format!(" {body}");
        let user_ids = self.tokenizer.encode(&rendered_body);
        let user = Message {
            role: "user",
            content: &body,
        };
        let open = self.encoder.encode_assistant_prefix(&[user]);
        if open.tokens.len() > PREFIX_LIMIT {
            return Err("exceeds_full_context_with_12_id_reserve");
        }
        let user_bounds = [self.user_head.len(), self.user_head.len() + user_ids.len()];
        if open.special_token_occurrences != 0
            || open.emitted_turns != 1
            || open.response_mask.iter().any(|&bit| bit != 0)
            || open.tokens != [self.user_head.as_slice(), &user_ids, &MARKER].concat()
            || self.tokenizer.decode_bytes(&user_ids) != rendered_body.as_bytes()
        {
            return Err("role_or_raw_render_parity");
        }
        let complete = self.encoder.encode_document(&[
            user,
            Message {
                role: "assistant",
                content: &annotation.text,
            },
        ]);
        let Some(tail) = complete.tokens.strip_prefix(&open.tokens[..]) else {
            return Err("response_prefix_changed");
        };
        let Some(response_ids) = tail.strip_suffix(&[1]) else {
            return Err("response_eos_missing");
        };
        if complete.special_token_occurrences != 0
            || complete.emitted_turns != 2
            || complete.response_mask[open.tokens.len()..]
                .iter()
                .any(|&bit| bit != 1)
            || self.tokenizer.decode_bytes(response_ids)
                != format!(" {}", annotation.text).as_bytes()
        {
            return Err("response_raw_parity");
        }
        let Some(response_lexical_offset) = response_ids.iter().position(|&id| id != 223) else {
            return Err("response_lexical_empty");
        };
        if response_ids.len() + 1 > 12 {
            return Err("exceeds_full_reply_with_eos_reserve");
        }
        let first_id = response_ids[response_lexical_offset];
        let [start, end] = bytes.map(|offset| offset + 1);
        let mut cursor = 0;
        let mut positions = Vec::new();
        let mut first_aligned = false;
        let mut last_aligned = false;
        for (index, &id) in user_ids.iter().enumerate() {
            let token_bytes = self.tokenizer.decode_bytes(&[id]);
            let next = cursor + token_bytes.len();
            if cursor < end && start < next {
                if positions.is_empty() {
                    let whitespace = token_bytes
                        .iter()
                        .take_while(|b| b.is_ascii_whitespace())
                        .count();
                    first_aligned = cursor + whitespace == start && next <= end && id == first_id;
                }
                last_aligned = next == end;
                positions.push(index);
            }
            cursor = next;
        }
        if !first_aligned {
            return Err("first_exact_id_not_at_native_span");
        }
        if user_ids.iter().filter(|&&id| id == first_id).count() != 1 {
            return Err("first_exact_id_ambiguous_in_whole_user");
        }
        let first = positions[0];
        let last = *positions.last().ok_or("span_has_no_source_tokens")?;
        let source_ids = user_ids[first..=last].to_vec();
        let whole_span_ids = last_aligned && *source_ids == response_ids[response_lexical_offset..];
        Ok(Candidate {
            question,
            body,
            question_key,
            input: open.tokens,
            user_bounds,
            label: Label {
                char_start: annotation.answer_start,
                bytes,
                text: annotation.text.clone(),
                first_id,
                first_position: user_bounds[0] + first,
                source_ids,
                whole_span_ids,
                response_ids: response_ids.to_vec(),
                response_lexical_offset,
                agreement,
            },
        })
    }
}

fn question_edit(first: &str, second: &str) -> usize {
    let second: Vec<_> = second.split_whitespace().collect();
    let mut previous: Vec<_> = (0..=second.len()).collect();
    let mut next = previous.clone();
    for (row, word) in first.split_whitespace().enumerate() {
        next[0] = row + 1;
        for (col, other) in second.iter().enumerate() {
            next[col + 1] = (next[col] + 1)
                .min(previous[col + 1] + 1)
                .min(previous[col] + usize::from(word != *other));
        }
        std::mem::swap(&mut previous, &mut next);
    }
    previous[second.len()]
}

fn collect<'a>(
    render: &Render<'_>,
    articles_for_split: &[&'a Article],
    blocked: &Blocklist,
    held_articles: &[&Article],
    official_dev: bool,
    cap: usize,
    counts: &mut Counts,
) -> Result<Vec<Pair<'a>>> {
    let mut matched = Vec::new();
    let mut input_ids = HashSet::new();
    let mut questions = HashSet::new();
    let mut contexts = HashSet::new();
    let mut articles = BTreeMap::new();
    let mut prior = HashSet::new();
    let mut dev_shingles = HashSet::new();
    let mut dev_titles = HashSet::new();
    let mut dev_passages = HashSet::new();
    let mut dev_questions = HashSet::new();
    for article in held_articles {
        dev_titles.insert(normalize(&article.title));
        for paragraph in &article.paragraphs {
            let context = normalize(&paragraph.context);
            dev_shingles.extend(shingles(&context));
            dev_passages.insert(context);
            dev_questions.extend(paragraph.qas.iter().map(|q| normalize(&q.question)));
            prior.extend(paragraph.qas.iter().map(|q| q.id.as_str()));
        }
    }
    for &article in articles_for_split {
        count(counts, "articles_seen", 1);
        let title = normalize(&article.title);
        let article_blocked = blocked.titles.contains(&title) || dev_titles.contains(&title);
        for (index, paragraph) in article.paragraphs.iter().enumerate() {
            count(counts, "paragraphs_seen", 1);
            for question in &paragraph.qas {
                if !input_ids.insert(&question.id) || prior.contains(question.id.as_str()) {
                    return Err(format!(
                        "duplicate or cross-split dataset question ID {}",
                        question.id
                    )
                    .into());
                }
                count(
                    counts,
                    if question.is_impossible {
                        "official_impossible"
                    } else {
                        "official_answerable"
                    },
                    1,
                );
            }
            if article_blocked {
                count(counts, "excluded_title_overlap_rows", paragraph.qas.len());
                continue;
            }
            let context = normalize(&paragraph.context);
            if blocked.passages.contains(&context)
                || dev_passages.contains(&context)
                || shingles(&context)
                    .any(|hash| blocked.shingles.contains(&hash) || dev_shingles.contains(&hash))
            {
                count(
                    counts,
                    "excluded_passage_or_shingle_overlap_rows",
                    paragraph.qas.len(),
                );
                continue;
            }
            let mut positives = Vec::new();
            let mut negatives = Vec::new();
            for question in &paragraph.qas {
                match render.candidate(paragraph, question, official_dev, blocked, &dev_questions) {
                    Ok(candidate) if question.is_impossible => {
                        count(counts, "eligible_no_read_with_exact_plausible", 1);
                        negatives.push(candidate);
                    }
                    Ok(candidate) => {
                        count(counts, "eligible_supported_with_exact_gold", 1);
                        positives.push(candidate);
                    }
                    Err(reason) => count(counts, reason, 1),
                }
            }
            if positives.is_empty() || negatives.is_empty() {
                continue;
            }
            count(counts, "paragraphs_with_both", 1);
            let mut best = None;
            let mut exact_count = 0;
            for (pos_index, positive) in positives.iter().enumerate() {
                for (neg_index, negative) in negatives.iter().enumerate() {
                    if positive.question_key == negative.question_key
                        || questions.contains(&positive.question_key)
                        || questions.contains(&negative.question_key)
                    {
                        continue;
                    }
                    let same_span = positive.label.bytes == negative.label.bytes
                        && positive.label.first_position == negative.label.first_position
                        && positive.label.first_id == negative.label.first_id;
                    exact_count += usize::from(same_span);
                    let same_opening = positive.question_key.split_whitespace().next()
                        == negative.question_key.split_whitespace().next();
                    let edit = question_edit(&positive.question_key, &negative.question_key);
                    let length_gap = positive.input.len().abs_diff(negative.input.len());
                    let rank = (
                        !same_span,
                        !same_opening,
                        edit,
                        length_gap,
                        pos_index,
                        neg_index,
                    );
                    if best.is_none_or(|current| rank < current) {
                        best = Some(rank);
                    }
                }
            }
            if exact_count != 0 {
                count(counts, "paragraphs_with_same_annotated_span", 1);
                count(counts, "same_annotated_span_combinations", exact_count);
            }
            let Some((different_span, _, edit, _, pos_index, neg_index)) = best else {
                continue;
            };
            if !contexts.insert(context) {
                count(counts, "skipped_duplicate_pair_passages", 1);
                continue;
            }
            let positive = positives.swap_remove(pos_index);
            let negative = negatives.swap_remove(neg_index);
            let same_first_position = positive.label.first_position
                == negative.label.first_position
                && positive.label.first_id == negative.label.first_id;
            questions.insert(positive.question_key.clone());
            questions.insert(negative.question_key.clone());
            matched.push(Pair {
                article: &article.title,
                paragraph: index,
                passage_sha256: hex::encode(Sha256::digest(paragraph.context.as_bytes())),
                positive,
                negative,
                same_span: !different_span,
                same_first_position,
                question_edit: edit,
            });
        }
    }
    // Favor one exact shared occurrence over approximate matching. SHA-256 is a
    // stable tie break across hosts and prevents source file order selecting a topic.
    matched.sort_by_cached_key(|pair| {
        (
            !pair.same_span,
            Sha256::digest(pair.passage_sha256.as_bytes()),
        )
    });
    matched.retain(|pair| {
        let per_article = articles.entry(pair.article).or_insert(0);
        *per_article += 1;
        *per_article <= ARTICLE_CAP
    });
    matched.truncate(cap);
    Ok(matched)
}

fn write_split(output: &Path, split: &str, pairs: &[Pair<'_>], counts: &mut Counts) -> Result<()> {
    let mut prompts = BufWriter::new(File::create(output.join(format!("{split}-prompts.jsonl")))?);
    let mut truth = BufWriter::new(File::create(output.join(format!("{split}-truth.jsonl")))?);
    let articles: HashSet<_> = pairs.iter().map(|pair| pair.article).collect();
    count(counts, "selected_articles", articles.len());
    count(counts, "selected_pairs", pairs.len());
    let mut selected_prompts = HashSet::new();
    let mut min_prefix = usize::MAX;
    let mut max_prefix = 0;
    for pair in pairs {
        if pair.same_first_position {
            count(counts, "selected_same_first_position_pairs", 1);
        }
        if pair.same_span {
            count(counts, "selected_same_annotated_span_pairs", 1);
            if pair.question_edit == 1 {
                count(counts, "selected_same_span_single_word_edit_pairs", 1);
            }
        }
        if pair.positive.input.len() == pair.negative.input.len() {
            count(counts, "selected_equal_prefix_length_pairs", 1);
        }
        for (candidate, other) in [
            (&pair.positive, &pair.negative),
            (&pair.negative, &pair.positive),
        ] {
            if !selected_prompts.insert(&candidate.input) {
                return Err(bad_input("duplicate selected model-facing prefix"));
            }
            let impossible = candidate.question.is_impossible;
            let label = &candidate.label;
            min_prefix = min_prefix.min(candidate.input.len());
            max_prefix = max_prefix.max(candidate.input.len());
            if !impossible {
                let lexical_ids = label.response_ids.len() - label.response_lexical_offset;
                count(
                    counts,
                    if lexical_ids == 1 {
                        "selected_gold_single_lexical_id"
                    } else {
                        "selected_gold_multiple_lexical_ids"
                    },
                    1,
                );
                if label.whole_span_ids {
                    count(counts, "selected_gold_all_lexical_ids_match", 1);
                }
            }
            if label.response_lexical_offset != 0 {
                count(
                    counts,
                    if impossible {
                        "selected_plausible_leading_space"
                    } else {
                        "selected_gold_leading_space"
                    },
                    1,
                );
            }
            jsonl(
                &mut prompts,
                &Prompt {
                    case_id: &candidate.question.id,
                    prompt: &candidate.body,
                },
            )?;
            jsonl(
                &mut truth,
                &Truth {
                    case_id: &candidate.question.id,
                    paired_with: &other.question.id,
                    article: pair.article,
                    paragraph: pair.paragraph,
                    passage_sha256: &pair.passage_sha256,
                    question: &candidate.question.question,
                    is_impossible: impossible,
                    prefix: &candidate.input,
                    user_bounds: candidate.user_bounds,
                    gold_source: (!impossible).then_some(label),
                    plausible_only: impossible.then_some(label),
                    same_annotated_span: pair.same_span,
                    same_first_position: pair.same_first_position,
                    question_edit: pair.question_edit,
                },
            )?;
        }
    }
    if !pairs.is_empty() {
        count(counts, "selected_min_prefix_ids", min_prefix);
        count(counts, "selected_max_prefix_ids", max_prefix);
    }
    prompts.flush()?;
    truth.flush()?;
    Ok(())
}

fn run(runs: &Path, inputs: &Path, output: &Path) -> Result<()> {
    let train: Dataset = serde_json::from_slice(&checked_bytes(&inputs.join(TRAIN), PINS[0].1)?)?;
    let dev: Dataset = serde_json::from_slice(&checked_bytes(&inputs.join(DEV), PINS[1].1)?)?;
    if train.version != "v2.0" || dev.version != "v2.0" {
        return Err(bad_input("unexpected official dataset version"));
    }
    let tokenizer_bytes = checked_bytes(&runs.join(TOKENIZER), PINS[2].1)?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&tokenizer_bytes)
        .ok_or_else(|| bad_input("invalid original tokenizer"))?;
    if tokenizer.vocab_size() != 4096 || tokenizer.encode(" ") != [223] {
        return Err(bad_input("unexpected original tokenizer identity"));
    }
    let blocked = Blocklist::load(runs)?;
    let render = Render::new(&tokenizer)?;
    let mut train_counts = BTreeMap::new();
    let mut dev_counts = BTreeMap::new();
    let mut reserve_counts = BTreeMap::new();
    let dev_articles: Vec<_> = dev.data.iter().collect();
    let (reserve_articles, fit_articles): (Vec<_>, Vec<_>) =
        train.data.iter().partition(|article| {
            let key = format!(
                "uor-whole-turn-squad-reserve-v1 {}",
                normalize(&article.title)
            );
            Sha256::digest(key.as_bytes())[0] < 115
        });
    let held = collect(
        &render,
        &dev_articles,
        &blocked,
        &[],
        true,
        DEV_CAP,
        &mut dev_counts,
    )?;
    let reserve = collect(
        &render,
        &reserve_articles,
        &blocked,
        &dev_articles,
        false,
        RESERVE_CAP,
        &mut reserve_counts,
    )?;
    let mut protected = dev_articles;
    protected.extend(reserve_articles);
    let fit = collect(
        &render,
        &fit_articles,
        &blocked,
        &protected,
        false,
        TRAIN_CAP,
        &mut train_counts,
    )?;
    if fit.is_empty() || held.is_empty() || reserve.is_empty() {
        return Err(bad_input("empty matched candidate split"));
    }
    fs::create_dir(output)?;
    write_split(output, "train", &fit, &mut train_counts)?;
    write_split(output, "dev", &held, &mut dev_counts)?;
    write_split(output, "reserve", &reserve, &mut reserve_counts)?;
    let report = serde_json::json!({
        "schema": "whole-turn-natural-supervision-census/1",
        "input_sha256": PINS.into_iter().collect::<BTreeMap<_, _>>(),
        "prefix_limit": PREFIX_LIMIT,
        "shingle_words": SHINGLE_WORDS,
        "article_cap": ARTICLE_CAP,
        "train_pair_cap": TRAIN_CAP,
        "dev_pair_cap": DEV_CAP,
        "reserve_pair_cap": RESERVE_CAP,
        "train": train_counts,
        "dev": dev_counts,
        "reserve": reserve_counts,
    });
    let mut out = BufWriter::new(File::create(output.join("census.json"))?);
    serde_json::to_writer_pretty(&mut out, &report)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let arguments: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [runs, inputs, output] = arguments.as_slice() else {
        return Err(bad_input(
            "usage: whole-turn-supervision WICK_RUNS OFFICIAL_INPUTS NEW_OUTPUT",
        ));
    };
    run(runs, inputs, output)
}
