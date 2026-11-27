use crate::*;

const DATA_SHA: &str = "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a";
const PRIOR_SHA: &str = "bea3926ba99a205a2035ae81c8aaa78cc864f6cccf408a071bcccca4553f86aa";
const BANK_SHA: &str = "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541";
const DONOR_BANK: &str =
    include_str!("../../../../../../labs/m2-donor-corpus/instrument/src/bank.rs");
const MAX_PASSAGE_IDS: usize = 150;
const INSTRUCTION: &str = "Read this passage and give a short direct answer.";
const EXCLUDED_WORDS: &str = "he him his she her hers they them their theirs it its these those this that there former latter same then also other else";

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
struct Judgment {
    custody: PriorCustody,
}

#[derive(Deserialize)]
struct PriorCustody {
    source_conversation_id: Option<String>,
}

#[derive(Default, Serialize)]
pub(crate) struct Census {
    licensed: usize,
    prior_conversations: usize,
    prior_passage_excluded: usize,
    duplicate_passage: usize,
    eligible_turns: usize,
    bounded_pairs_scanned: usize,
    standalone_space_pair_excluded: usize,
    positive_target_collision: usize,
    raw_candidates: usize,
    candidate_passages: usize,
    reviewed_pairs: usize,
    equal_length_pairs: usize,
    from: BTreeMap<String, usize>,
    pins: BTreeMap<&'static str, String>,
}

struct Candidate<'a> {
    turn: usize,
    question: &'a str,
    answer: &'a str,
    evidence: &'a str,
}

fn contains_words(haystack: &str, needle: &str) -> bool {
    !needle.is_empty() && format!(" {haystack} ").contains(&format!(" {needle} "))
}

fn load_bank(path: &Path) -> Result<HashSet<String>> {
    let lines = BufReader::new(File::open(path)?)
        .lines()
        .collect::<std::io::Result<HashSet<_>>>()?;
    if lines.is_empty() || lines.iter().any(|line| line.len() != 16) {
        return Err("invalid D87 prompt digest file".into());
    }
    Ok(lines)
}

fn followups() -> Result<HashSet<String>> {
    let (lines, _) = DONOR_BANK
        .split_once("pub const FOLLOW_UPS: [&str; 40] = [\n")
        .and_then(|(_, rest)| rest.split_once("\n];"))
        .ok_or("D87 follow-up source changed")?;
    let result: Option<HashSet<_>> = lines
        .lines()
        .map(|line| {
            line.trim()
                .strip_prefix('"')?
                .strip_suffix("\",")
                .map(normalized)
        })
        .collect();
    result
        .filter(|set| set.len() == 40)
        .ok_or_else(|| "D87 follow-ups not recovered".into())
}

fn windows<'a>(story: &'a str, builder: &Builder<'_>) -> Vec<(usize, usize, &'a str)> {
    let mut endpoints = vec![0];
    for (i, space) in story.match_indices(char::is_whitespace) {
        if story[..i].ends_with(['.', '!', '?', '"', '”']) {
            endpoints.push(i + space.len());
        }
    }
    endpoints.push(story.len());
    endpoints.sort_unstable();
    endpoints.dedup();
    let mut ranges = Vec::new();
    for (idx, &start) in endpoints.iter().enumerate() {
        for &end in &endpoints[idx + 1..] {
            let slice = story[start..end].trim();
            if builder.tokenizer.encode(slice).len() > MAX_PASSAGE_IDS {
                break;
            }
            if !slice.is_empty() {
                let leading = story[start..end].len() - story[start..end].trim_start().len();
                ranges.push((start + leading, start + leading + slice.len(), slice));
            }
        }
    }
    ranges.sort_by_key(|(start, end, _)| (std::cmp::Reverse(end - start), *start));
    ranges
}

fn candidate<'a>(
    conversation: &'a Conversation,
    i: usize,
    bank: &HashSet<String>,
    followups: &HashSet<String>,
) -> Result<Option<Candidate<'a>>> {
    let question = &conversation.questions[i];
    let reply = &conversation.answers[i];
    if question.turn_id != i + 1 || reply.turn_id != i + 1 {
        return Err(format!("misaligned CoQA turn in {}", conversation.id).into());
    }
    let q = normalized(&question.input_text);
    let q_words: Vec<_> = q.split_whitespace().collect();
    let Some(&opening) = q_words.first() else {
        return Ok(None);
    };
    if question.input_text.matches('?').count() != 1
        || !question.input_text.trim().ends_with('?')
        || question.input_text.contains("<|")
        || q_words.len() < 4
        || !matches!(opening, "who" | "what" | "when" | "where" | "which" | "how")
        || q_words.iter().any(|word| {
            EXCLUDED_WORDS
                .split_whitespace()
                .any(|excluded| &excluded == word)
        })
        || followups.contains(&q)
        || bank.contains(&hex::encode(&Sha256::digest(&q)[..8]))
    {
        return Ok(None);
    }
    let a = normalized(&reply.input_text);
    let evidence = normalized(&reply.span_text);
    if a.is_empty()
        || a.split_whitespace().count() > 4
        || matches!(a.as_str(), "yes" | "no" | "unknown")
        || contains_words(&q, &a)
        || !contains_words(&evidence, &a)
    {
        return Ok(None);
    }
    for annotator in conversation.additional_answers.values() {
        let annotation = annotator
            .get(i)
            .filter(|turn| turn.turn_id == i + 1)
            .ok_or("missing CoQA independent answer")?;
        if normalized(&annotation.input_text) != a {
            return Ok(None);
        }
    }
    Ok(Some(Candidate {
        turn: i + 1,
        question: question.input_text.trim(),
        answer: reply.input_text.trim(),
        evidence: reply.span_text.trim(),
    }))
}

fn paired(
    builder: &Builder<'_>,
    conversation: &Conversation,
    windows: &[(usize, usize, &str)],
    left: &Candidate<'_>,
    right: &Candidate<'_>,
    census: &mut Census,
) -> Result<Option<Pair>> {
    let left_evidence = normalized(left.evidence);
    let right_evidence = normalized(right.evidence);
    if normalized(left.question) == normalized(right.question)
        || normalized(left.answer) == normalized(right.answer)
        || contains_words(&normalized(right.question), &normalized(left.answer))
        || contains_words(&normalized(left.question), &normalized(right.answer))
    {
        return Ok(None);
    }
    let group_id = id("coqa-group", &conversation.id);
    let key = format!("{}:{}:{}", conversation.id, left.turn, right.turn);
    let pair_id = id("coqa-pair", &key);
    for &(start, end, excerpt) in windows {
        let text = normalized(excerpt);
        if !contains_words(&text, &left_evidence) || !contains_words(&text, &right_evidence) {
            continue;
        }
        let context = format!("{INSTRUCTION}\n\n{excerpt}");
        let Some(a) = builder.case(&context, left.question, left.answer)? else {
            continue;
        };
        let Some(b) = builder.case(&context, right.question, right.answer)? else {
            continue;
        };
        census.bounded_pairs_scanned += 1;
        if a.lexical_offset != 0 || b.lexical_offset != 0 {
            census.standalone_space_pair_excluded += 1;
            return Ok(None);
        }
        if a.target() == b.target() {
            census.positive_target_collision += 1;
            return Ok(None);
        }
        let mut cases = [a, b];
        let mut turns = [left.turn, right.turn];
        let mut evidence = [left.evidence, right.evidence];
        if hash("coqa-question-order", &key)[0] & 1 != 0 {
            cases.swap(0, 1);
            turns.swap(0, 1);
            evidence.swap(0, 1);
        }
        return Ok(Some(Pair {
            group_id,
            pair_id,
            cohort: "coqa",
            cases,
            source: json!({
                "dataset":"CoQA dev v1.0",
                "source":conversation.source,
                "source_conversation_id":conversation.id,
                "passage_filename":conversation.filename,
                "passage_utf8_start":start,
                "passage_utf8_end":end,
                "turns":turns,
                "evidence":evidence,
            }),
        }));
    }
    Ok(None)
}

pub(crate) fn pairs(
    builder: &Builder<'_>,
    dataset_file: &Path,
    prior_file: &Path,
    bank_file: &Path,
    review: &str,
) -> Result<(Vec<Pair>, Census)> {
    for (file, digest) in [
        (dataset_file, DATA_SHA),
        (prior_file, PRIOR_SHA),
        (bank_file, BANK_SHA),
    ] {
        pin(file, digest)?;
    }
    let bank = load_bank(bank_file)?;
    let followups = followups()?;
    let dataset: Dataset = serde_json::from_reader(BufReader::new(File::open(dataset_file)?))?;
    let mut prior = HashSet::new();
    for line in BufReader::new(File::open(prior_file)?).lines() {
        let judgment: Judgment = serde_json::from_str(&line?)?;
        if let Some(conversation) = judgment.custody.source_conversation_id {
            if !prior.insert(conversation) {
                return Err("duplicate conversation in previous panel".into());
            }
        }
    }
    let passages: HashSet<_> = dataset
        .data
        .iter()
        .filter(|source| prior.contains(&source.id))
        .map(|source| normalized(source.story.trim()))
        .collect();
    if prior.len() != 200 || passages.len() != prior.len() {
        return Err("could not resolve every prior held passage".into());
    }
    let mut census = Census {
        prior_conversations: prior.len(),
        pins: [
            ("dataset", DATA_SHA.to_owned()),
            ("prior_judgments", PRIOR_SHA.to_owned()),
            ("d87_prompt_digests", BANK_SHA.to_owned()),
        ]
        .into_iter()
        .collect(),
        ..Census::default()
    };
    let preview = review == "--preview";
    let mut reviewed = HashSet::new();
    if !preview {
        let review = Path::new(review);
        census.pins.insert("review", sha(review)?);
        for line in BufReader::new(File::open(review)?).lines() {
            let id = line?;
            if id.len() != 24
                || !id.bytes().all(|byte| byte.is_ascii_hexdigit())
                || !reviewed.insert(id)
            {
                return Err("invalid pair ID in the passage review".into());
            }
        }
        if reviewed.is_empty() {
            return Err("no CoQA candidates admitted by passage review".into());
        }
    }
    let mut seen_passage = HashSet::new();
    let mut candidate_passages = HashSet::new();
    let mut accepted_passages = HashSet::new();
    let mut output = Vec::new();
    for conversation in &dataset.data {
        if !matches!(
            conversation.source.as_str(),
            "wikipedia" | "cnn" | "gutenberg"
        ) {
            continue;
        }
        census.licensed += 1;
        let story = conversation.story.trim();
        let passage = normalized(story);
        if prior.contains(&conversation.id) || passages.contains(&passage) {
            census.prior_passage_excluded += 1;
            continue;
        }
        if !seen_passage.insert(passage) {
            census.duplicate_passage += 1;
            continue;
        }
        if conversation.questions.len() != conversation.answers.len()
            || conversation.additional_answers.len() != 3
        {
            return Err(format!("bad CoQA annotation shape in {}", conversation.id).into());
        }
        let options = (0..conversation.questions.len())
            .map(|i| candidate(conversation, i, &bank, &followups))
            .collect::<Result<Vec<_>>>()?;
        let turns: Vec<_> = options.into_iter().flatten().collect();
        census.eligible_turns += turns.len();
        if turns.len() < 2 {
            continue;
        }
        let windows = windows(story, builder);
        for (i, left) in turns.iter().enumerate() {
            for right in &turns[i + 1..] {
                let Some(pair) = paired(builder, conversation, &windows, left, right, &mut census)?
                else {
                    continue;
                };
                census.raw_candidates += 1;
                candidate_passages.insert(&conversation.id);
                if preview {
                    println!(
                        "{}",
                        json!({
                            "pair_id":pair.pair_id,
                            "group_id":pair.group_id,
                            "context":pair.cases[0].user.split_once("\n\n").map(|(_, tail)| tail.rsplit_once("\n\n").map(|(context, _)| context)),
                            "questions":[&pair.cases[0].question,&pair.cases[1].question],
                            "answers":[&pair.cases[0].reply,&pair.cases[1].reply],
                            "lengths":[pair.cases[0].ids.len(),pair.cases[1].ids.len()],
                            "source":pair.source,
                        })
                    );
                } else if reviewed.remove(&pair.pair_id) {
                    if !accepted_passages.insert(&conversation.id) {
                        return Err("more than one reviewed pair in a passage".into());
                    }
                    if pair.cases[0].ids.len() == pair.cases[1].ids.len() {
                        census.equal_length_pairs += 1;
                    }
                    *census.from.entry(conversation.source.clone()).or_default() += 1;
                    output.push(pair);
                }
            }
        }
    }
    if !reviewed.is_empty() {
        return Err("passage review lists a pair that failed deterministic eligibility".into());
    }
    census.candidate_passages = candidate_passages.len();
    census.reviewed_pairs = output.len();
    Ok((output, census))
}
