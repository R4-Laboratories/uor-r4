use crate::*;
use inputs::{Inputs, Prior};
use serde_json::json;

#[derive(Deserialize)]
struct Dataset {
    data: Vec<Passage>,
}

#[derive(Deserialize)]
struct Passage {
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
    #[serde(default)]
    span_start: isize,
    #[serde(default)]
    span_end: isize,
}

#[derive(Clone, Debug, Serialize)]
struct Rival {
    text: String,
    id: u32,
    pos: usize,
}

struct Candidate<'a> {
    passage: &'a Passage,
    turn: &'a Turn,
    evidence: (usize, usize),
    excerpt: (usize, usize),
    answer_match: (usize, usize),
    case: Case,
    consensus: usize,
    gold_pos: Option<usize>,
    status: Status,
    rivals: Vec<Rival>,
}

fn valid_reply(text: &str) -> bool {
    let value = normalized(text);
    let count = value.split_whitespace().count();
    count > 0 && count <= 5 && !matches!(value.as_str(), "yes" | "no" | "unknown")
}

fn contains_words(text: &str, query: &str) -> bool {
    let (text, query) = (normalized(text), normalized(query));
    !query.is_empty() && format!(" {text} ").contains(&format!(" {query} "))
}

fn evidence(passage: &Passage, turn: &Turn) -> Option<(usize, usize)> {
    let start = usize::try_from(turn.span_start).ok()?;
    let end = usize::try_from(turn.span_end).ok()?;
    let char_boundaries: Vec<_> = passage
        .story
        .char_indices()
        .map(|(idx, _)| idx)
        .chain([passage.story.len()])
        .collect();
    let from = *char_boundaries.get(start)?;
    let to = *char_boundaries.get(end)?;
    let expected = turn.span_text.trim();
    let slice = passage.story.get(from..to)?;
    if slice.trim() == expected && !expected.is_empty() {
        let begin = from + (slice.len() - slice.trim_start().len());
        return Some((begin, begin + expected.len()));
    }
    let mut copies = passage.story.match_indices(expected);
    let (begin, _) = copies.next()?;
    (!expected.is_empty() && copies.next().is_none()).then_some((begin, begin + expected.len()))
}

fn excerpts(story: &str, evidence: (usize, usize)) -> Vec<(usize, usize)> {
    let mut cuts = vec![0];
    cuts.extend(
        story
            .match_indices(char::is_whitespace)
            .filter_map(|(pos, space)| {
                let past_sentence = story[..pos].ends_with(['.', '?', '!', '”', '"']);
                (past_sentence || space.contains('\n')).then_some(pos + space.len())
            }),
    );
    cuts.push(story.len());
    cuts.sort_unstable();
    cuts.dedup();
    let mut ranges = Vec::new();
    for (index, &start) in cuts.iter().enumerate() {
        if start > evidence.0 {
            break;
        }
        for &end in &cuts[index + 1..] {
            if end < evidence.1 {
                continue;
            }
            if end - start > 1_100 {
                break;
            }
            let (left, right) = (story[start..end].trim_start(), story[start..end].trim_end());
            if left.is_empty() {
                continue;
            }
            let trimmed = (end - left.len(), start + right.len());
            if trimmed.0 <= evidence.0 && evidence.1 <= trimmed.1 {
                ranges.push(trimmed);
            }
        }
    }
    ranges.sort_by_key(|&(start, end)| (end - start, start));
    ranges.dedup();
    ranges
}

fn word_boundaries(text: &str, literal: &str) -> Vec<(usize, usize)> {
    let ascii = text.to_ascii_lowercase();
    let needle = literal.to_ascii_lowercase();
    ascii
        .match_indices(&needle)
        .filter_map(|(start, matched)| {
            let end = start + matched.len();
            let before_word = text[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric);
            let after_word = text[end..]
                .chars()
                .next()
                .is_some_and(char::is_alphanumeric);
            (!before_word && !after_word).then_some((start, end))
        })
        .collect()
}

fn assess(
    builder: &Builder<'_>,
    case: &Case,
    target: u32,
    evidence: (usize, usize),
    answer_match: (usize, usize),
) -> (Status, Option<usize>, Vec<Rival>) {
    // Token byte ranges include the protocol's one leading content space.
    let ranges = case.context_token_ranges(builder.tokenizer);
    let in_evidence = |start: usize, end: usize| start < evidence.1 && evidence.0 < end;
    let candidate = ranges.iter().find(|&&(pos, start, end)| {
        case.prefix[pos] == target
            && start <= answer_match.0
            && answer_match.0 < end
            && in_evidence(start, end)
    });
    let (status, gold_pos) = match (case.context_positions(target).len(), candidate) {
        (1, Some(&(pos, _, _))) => (Status::Unique, Some(pos)),
        (count, _) if count > 1 => (Status::Ambiguous, None),
        _ => (Status::Absent, None),
    };
    let mut rivals = Vec::new();
    let context = format!(" {}", case.context);
    for &(pos, start, end) in &ranges {
        let token_id = case.prefix[pos];
        if token_id == target
            || in_evidence(start, end)
            || case.context_positions(token_id).len() != 1
        {
            continue;
        }
        let decoded = builder.tokenizer.decode(&[token_id]);
        let label = decoded.trim();
        if label.chars().count() < 3
            || !label.chars().all(char::is_alphanumeric)
            || contains_words(&case.question, label)
        {
            continue;
        }
        let word_at = end - label.len();
        if !word_boundaries(&context, label).contains(&(word_at, end)) {
            continue;
        }
        rivals.push(Rival {
            text: label.to_owned(),
            id: token_id,
            pos,
        });
    }
    (status, gold_pos, rivals)
}

fn collect<'a>(
    builder: &Builder<'_>,
    dataset: &'a Dataset,
    prior: &Prior,
) -> Result<Vec<Candidate<'a>>> {
    let blocked_stories: HashSet<_> = dataset
        .data
        .iter()
        .filter(|passage| prior.coqa_ids.contains(&passage.id))
        .map(|passage| normalized(&passage.story))
        .collect();
    let excluded: HashSet<_> = ["yes", "no", "unknown"].into_iter().collect();
    let mut found = Vec::new();
    for passage in &dataset.data {
        if prior.coqa_ids.contains(&passage.id)
            || blocked_stories.contains(&normalized(&passage.story))
        {
            continue;
        }
        require(
            passage.answers.len() == passage.questions.len(),
            "CoQA turn counts disagree",
        )?;
        for (index, (question, answer)) in
            passage.questions.iter().zip(&passage.answers).enumerate()
        {
            require(
                question.turn_id == index + 1 && answer.turn_id == index + 1,
                "CoQA turn IDs disagree",
            )?;
            let q = normalized(&question.input_text);
            let opening = q.split_whitespace().next().unwrap_or_default();
            let answer_text = answer.input_text.trim();
            if !matches!(opening, "who" | "what" | "when" | "where" | "which" | "how")
                || q.split_whitespace().count() < 4
                || !question.input_text.trim().ends_with('?')
                || !valid_reply(answer_text)
                || excluded.contains(normalized(answer_text).as_str())
                || contains_words(&q, answer_text)
            {
                continue;
            }
            let expected = normalized(answer_text);
            let mut consensus = 1;
            for answers in passage.additional_answers.values() {
                let additional = answers
                    .get(index)
                    .filter(|turn| turn.turn_id == index + 1)
                    .ok_or("missing CoQA independent annotation")?;
                consensus += usize::from(normalized(&additional.input_text) == expected);
            }
            if consensus < 3 {
                continue;
            }
            let Some(support) = evidence(passage, answer) else {
                continue;
            };
            let support_text = &passage.story[support.0..support.1];
            let matches = word_boundaries(support_text, answer_text);
            if matches.len() != 1 {
                continue;
            }
            let answer_match = (support.0 + matches[0].0, support.0 + matches[0].1);
            let mut bounded = Vec::new();
            for excerpt in excerpts(&passage.story, support) {
                let context = &passage.story[excerpt.0..excerpt.1];
                let Some(case) =
                    builder.case(context, question.input_text.trim(), Some(answer_text))?
                else {
                    continue;
                };
                if case.context_end - case.context_start < 28 || !prior.allows_case(&case) {
                    continue;
                }
                let target = case.target().ok_or("missing CoQA lexical target")?;
                let first = builder.tokenizer.decode(&[target]);
                if !first.trim_start().starts_with(char::is_alphanumeric) {
                    continue;
                }
                let relative = |absolute: (usize, usize)| {
                    (absolute.0 + 1 - excerpt.0, absolute.1 + 1 - excerpt.0)
                };
                let (status, gold_pos, rivals) = assess(
                    builder,
                    &case,
                    target,
                    relative(support),
                    relative(answer_match),
                );
                if rivals.is_empty() {
                    continue;
                }
                bounded.push(Candidate {
                    passage,
                    turn: answer,
                    evidence: support,
                    excerpt,
                    answer_match,
                    case,
                    consensus,
                    gold_pos,
                    status,
                    rivals,
                });
            }
            bounded.sort_by_key(|candidate| {
                let length = candidate.case.context_end - candidate.case.context_start;
                let status = match candidate.status {
                    Status::Unique => 0,
                    Status::Ambiguous => 1,
                    Status::Absent => 2,
                };
                // Keep multiple sentences for contextual reading if a 65-token window fits.
                (status, length.abs_diff(65), std::cmp::Reverse(length))
            });
            if let Some(candidate) = bounded.into_iter().next() {
                found.push(candidate);
            }
        }
    }
    Ok(found)
}

fn dataset(inputs: &Inputs) -> Result<Dataset> {
    Ok(serde_json::from_reader(BufReader::new(File::open(
        &inputs.coqa,
    )?))?)
}

pub(crate) fn preview(builder: &Builder<'_>, inputs: &Inputs, prior: &Prior) -> Result<()> {
    let data = dataset(inputs)?;
    for candidate in collect(builder, &data, prior)? {
        println!(
            "{}",
            json!({
                "id": candidate.passage.id, "turn": candidate.turn.turn_id, "domain": candidate.passage.source,
                "question": candidate.case.question, "answer": candidate.case.reply,
                "context": candidate.case.context, "support": candidate.turn.span_text,
                "status": candidate.status, "rivals": candidate.rivals.iter().map(|r| &r.text).collect::<Vec<_>>(),
                "lexical_offset": candidate.case.lexical_offset, "consensus": candidate.consensus,
            })
        );
    }
    Ok(())
}

struct Review {
    conversation_id: String,
    turn: usize,
    topic: String,
    rival: String,
    rationale: String,
}

fn reviews(path: &Path) -> Result<Vec<Review>> {
    let mut result = Vec::new();
    let mut unique = HashSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        let [conversation_id, turn, topic, rival, rationale] = fields.as_slice() else {
            return Err(
                "expected source ID, turn, topic, rival and rationale in CoQA review".into(),
            );
        };
        require(
            unique.insert((*conversation_id).to_owned()),
            "CoQA review repeats a source passage",
        )?;
        require(
            !topic.is_empty() && !rival.is_empty() && !rationale.is_empty(),
            "empty CoQA review field",
        )?;
        result.push(Review {
            conversation_id: (*conversation_id).to_owned(),
            turn: turn.parse()?,
            topic: (*topic).to_owned(),
            rival: (*rival).to_owned(),
            rationale: (*rationale).to_owned(),
        });
    }
    require(
        result.len() >= 16,
        "fewer than sixteen text-reviewed independent CoQA sources",
    )?;
    Ok(result)
}

pub(crate) fn build(
    builder: &Builder<'_>,
    inputs: &Inputs,
    prior: &Prior,
    review: &Path,
) -> Result<Tape> {
    let data = dataset(inputs)?;
    let candidates = collect(builder, &data, prior)?;
    let mut tape = Tape::default();
    for row in reviews(review)? {
        let candidate = candidates
            .iter()
            .find(|c| c.passage.id == row.conversation_id && c.turn.turn_id == row.turn)
            .ok_or_else(|| {
                format!(
                    "CoQA reviewed turn no longer eligible: {}:{}",
                    row.conversation_id, row.turn
                )
            })?;
        let case = candidate.case.clone();
        let target = case.target().ok_or("missing reviewed target")?;
        let raw_target = case.gold[0];
        let rival = candidate
            .rivals
            .iter()
            .find(|rival| rival.text == row.rival)
            .ok_or_else(|| {
                format!(
                    "rival {} is not an exact competing occurrence in {}",
                    row.rival, row.conversation_id
                )
            })?;
        let group = id("coqa-group", &row.conversation_id);
        let pair = id(
            "coqa-pair",
            &format!("{}:{}", row.conversation_id, row.turn),
        );
        tape.push(
            Entry {
                group: &group,
                pair: &pair,
                cohort: "coqa",
                topic: &row.topic,
                assignment: 0,
                side: 0,
                equal_length: false,
            },
            case,
            (target, raw_target),
            (rival.id, rival.id),
            Annotation {
                gold_pos: candidate.gold_pos,
                rival_pos: Some(rival.pos),
                status: candidate.status,
            },
            json!({
                "dataset": "CoQA dev v1.0",
                "source": candidate.passage.source,
                "source_conversation_id": row.conversation_id,
                "passage_filename": candidate.passage.filename,
                "turn": row.turn,
                "passage_utf8_start": candidate.excerpt.0,
                "passage_utf8_end": candidate.excerpt.1,
                "annotator_span_start_char": candidate.turn.span_start,
                "annotator_span_end_char": candidate.turn.span_end,
                "evidence_utf8_start": candidate.evidence.0,
                "evidence_utf8_end": candidate.evidence.1,
                "evidence": &candidate.passage.story[candidate.evidence.0..candidate.evidence.1],
                "answer_utf8_start": candidate.answer_match.0,
                "answer_utf8_end": candidate.answer_match.1,
                "answer_id_context_occurrences": candidate.case.context_positions(target),
                "normalized_agreeing_annotations": candidate.consensus,
                "reference_count": 1 + candidate.passage.additional_answers.len(),
                "rival_text": rival.text,
                "rival_kind": "distinct_complete_word_outside_gold_evidence",
                "review_rationale": row.rationale,
            }),
        );
    }
    Ok(tape)
}

pub(crate) fn verify(
    builder: &Builder<'_>,
    inputs: &Inputs,
    tape: &Tape,
    review: &Path,
) -> Result<()> {
    let data = dataset(inputs)?;
    let reviews = reviews(review)?;
    require(
        reviews.len() == tape.keys.len(),
        "CoQA review/readback row count mismatch",
    )?;
    for ((row, key), review) in tape.rows.iter().zip(&tape.keys).zip(reviews) {
        let passage = data
            .data
            .iter()
            .find(|p| p.id == review.conversation_id)
            .ok_or("audited CoQA source missing")?;
        let source = &key.source;
        let offset = |field: &str| {
            source[field]
                .as_u64()
                .map(|v| v as usize)
                .ok_or("missing source offset")
        };
        let excerpt = (offset("passage_utf8_start")?, offset("passage_utf8_end")?);
        let evidence = (offset("evidence_utf8_start")?, offset("evidence_utf8_end")?);
        let answer = (offset("answer_utf8_start")?, offset("answer_utf8_end")?);
        let answer_turn = passage
            .answers
            .get(review.turn.checked_sub(1).ok_or("invalid turn number")?)
            .ok_or("invalid answer turn")?;
        let question_turn = passage
            .questions
            .get(review.turn - 1)
            .ok_or("invalid question turn")?;
        require(
            passage.story.get(excerpt.0..excerpt.1) == Some(&key.context)
                && passage.story.get(evidence.0..evidence.1) == Some(answer_turn.span_text.trim())
                && passage
                    .story
                    .get(answer.0..answer.1)
                    .is_some_and(|literal| {
                        literal.eq_ignore_ascii_case(answer_turn.input_text.trim())
                    })
                && source["source_conversation_id"] == passage.id
                && source["passage_filename"] == passage.filename
                && source["turn"] == review.turn
                && excerpt.0 <= evidence.0
                && evidence.0 <= answer.0
                && answer.1 <= evidence.1
                && evidence.1 <= excerpt.1
                && question_turn.input_text.trim() == key.question
                && key.reply.as_deref() == Some(answer_turn.input_text.trim())
                && key.topic == review.topic
                && source["rival_text"] == review.rival
                && source["review_rationale"] == review.rationale,
            format!(
                "CoQA source excerpt, gold evidence or human review changed for {}",
                review.conversation_id
            ),
        )?;
        let pos = row.rival_pos.ok_or("CoQA rival has no address")?;
        require(
            builder.tokenizer.decode(&[row.prefix[pos]]).trim() == review.rival,
            "CoQA rival changed",
        )?;
    }
    Ok(())
}
