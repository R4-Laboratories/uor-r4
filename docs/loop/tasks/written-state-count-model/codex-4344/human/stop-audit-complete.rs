#![forbid(unsafe_code)]

use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs, io,
};
use uor_r4_tokenizer::{
    dialogue::{DialogueProtocol, Message},
    ByteBpeTokenizer,
};

// Use the original pinned DROP intake's address and cap checks. Its executable
// entry point is used separately to replay the original pairing receipt.
#[allow(dead_code)]
mod pinned_drop {
    include!("../../codex-4336/supply/drop/check.rs");

    pub(super) struct Single<'a> {
        pub question: &'a str,
        pub answer: &'a str,
        pub start: usize,
        pub ids: Vec<u32>,
    }

    pub(super) fn single<'a>(
        tokenizer: &ByteBpeTokenizer,
        encoder: &DialogueEncoder<'_>,
        passage: &str,
        row: &'a Value,
        counts: &mut BTreeMap<String, usize>,
    ) -> Result<Option<Single<'a>>, &'static str> {
        let Some(case) = case(tokenizer, encoder, passage, row, counts)? else {
            return Ok(None);
        };
        let [answer] = case.answers.as_slice() else {
            return Err("DROP single cohort has more than one answer");
        };
        let mut spans = case.spans.into_iter();
        let span = spans.next().ok_or("DROP single cohort missing span")?;
        Ok(Some(Single {
            question: case.question,
            answer,
            start: span.start,
            ids: span.ids,
        }))
    }
}

const COMMON: &str = "a an and are as at be been but by can could did do does for from had has have he her here him his how i if in is it its may me my no not of on one or our out she so than that the their them then there these they this those to up us was we were what when where which who will with would yes you your";
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const COPY_STOPS: [u32; 4] = [16, 1, 0, 201];

fn ids(value: &Value) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(serde_json::from_value(value.clone())?)
}

fn screen_stop(
    t: &ByteBpeTokenizer,
    user_ids: &[u32],
    source_start: usize,
    source: &[u32],
    text: &str,
) -> Result<&'static str, Box<dyn std::error::Error>> {
    if source.is_empty() || user_ids.get(source_start..source_start + source.len()) != Some(source)
    {
        return Err("source interval mismatch".into());
    }
    if !text.chars().any(char::is_alphanumeric) {
        return Ok("punctuation_or_whitespace");
    }
    if source.iter().any(|id| COPY_STOPS.contains(id)) {
        return Ok("writer_delimiter");
    }
    let boundaries = t.decode_bytes(&user_ids[..source_start]);
    let following = t.decode_bytes(&user_ids[source_start + source.len()..]);
    let actual = t.decode_bytes(source);
    let before = std::str::from_utf8(&boundaries)?;
    let after = std::str::from_utf8(&following)?;
    let actual = std::str::from_utf8(&actual)?;
    let value = actual.strip_prefix(' ').unwrap_or(actual);
    if value != text {
        return Err("source surface mismatch".into());
    }
    let crosses_left = !actual.starts_with(' ')
        && text.chars().next().is_some_and(char::is_alphanumeric)
        && before
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric);
    let crosses_right = text.chars().next_back().is_some_and(char::is_alphanumeric)
        && after.chars().next().is_some_and(char::is_alphanumeric);
    if crosses_left || crosses_right {
        return Ok("partial_word");
    }
    let whole = [before, actual, after].concat();
    let repeated_surface = whole
        .char_indices()
        .filter(|(start, _)| {
            if !whole[*start..].starts_with(text) {
                return false;
            }
            let end = start + text.len();
            (!text.chars().next().is_some_and(char::is_alphanumeric)
                || !whole[..*start]
                    .chars()
                    .next_back()
                    .is_some_and(char::is_alphanumeric))
                && (!text.chars().next_back().is_some_and(char::is_alphanumeric)
                    || !whole[end..]
                        .chars()
                        .next()
                        .is_some_and(char::is_alphanumeric))
        })
        .nth(1)
        .is_some();
    if repeated_surface
        || user_ids
            .windows(source.len())
            .filter(|window| *window == source)
            .count()
            != 1
    {
        return Ok("repeated_full_interval");
    }
    if source.len() == 1 {
        if COMMON
            .split_ascii_whitespace()
            .any(|word| text.eq_ignore_ascii_case(word))
        {
            return Ok("common_one_id");
        }
        return Ok("eligible_one_id");
    }
    Ok("eligible_multiple_ids")
}

fn bump(counts: &mut BTreeMap<&'static str, usize>, key: &'static str) {
    *counts.entry(key).or_default() += 1;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: tokenizer squad-truth squad-prompts squad-original; DROP on stdin".into(),
        );
    }
    let t = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(&args[1])?).ok_or("tokenizer")?;
    let enc = DialogueProtocol::literal_roles_v2(&t)?.bind(&t)?;
    let head = [vec![0], t.encode("User:")].concat();

    let squad: Value = serde_json::from_slice(&fs::read(&args[4])?)?;
    let mut squad_titles = BTreeMap::new();
    for article in squad["data"].as_array().ok_or("SQuAD articles")? {
        squad_titles.insert(
            article["title"].as_str().ok_or("SQuAD title")?,
            &article["paragraphs"],
        );
    }
    let mut prompts = BTreeMap::new();
    let prompt_json = fs::read_to_string(&args[3])?;
    for line in prompt_json.lines() {
        let row: Value = serde_json::from_str(line)?;
        let case_id = row["case_id"]
            .as_str()
            .ok_or("SQuAD prompt case id")?
            .to_owned();
        let prompt = row["prompt"]
            .as_str()
            .ok_or("SQuAD prompt text")?
            .to_owned();
        if prompts.insert(case_id, prompt).is_some() {
            return Err("duplicate SQuAD prompt case id".into());
        }
    }
    let mut squad_counts = BTreeMap::new();
    let mut s_questions = BTreeSet::new();
    let squad_truth = fs::read_to_string(&args[2])?;
    for line in squad_truth.lines() {
        let row: Value = serde_json::from_str(line)?;
        let case_id = row["case_id"].as_str().ok_or("SQuAD truth case id")?;
        let title = row["article"].as_str().ok_or("SQuAD truth article")?;
        let paragraph = row["paragraph"].as_u64().ok_or("SQuAD truth paragraph")? as usize;
        let original = &squad_titles[title][paragraph];
        let passage = original["context"]
            .as_str()
            .ok_or("SQuAD original context")?;
        use sha2::Digest;
        if row["passage_sha256"] != format!("{:x}", sha2::Sha256::digest(passage)) {
            return Err("SQuAD paragraph checksum mismatch".into());
        }
        let question = row["question"].as_str().ok_or("SQuAD truth question")?;
        let body = format!("{passage} {question}");
        let official = original["qas"]
            .as_array()
            .ok_or("SQuAD questions")?
            .iter()
            .find(|qa| qa["id"] == case_id)
            .ok_or("SQuAD official question ID")?;
        if !s_questions.insert(case_id.to_owned())
            || prompts.get(case_id) != Some(&body)
            || official["question"] != question
            || official["is_impossible"] != row["is_impossible"]
        {
            return Err("SQuAD question provenance mismatch".into());
        }
        let user = Message {
            role: "user",
            content: &body,
        };
        let opening = enc.encode_assistant_prefix(&[user.clone()]);
        let user_ids = t.encode(&format!(" {body}"));
        if opening.tokens != ids(&row["prefix"])?
            || opening.tokens != [&head[..], &user_ids, &MARKER].concat()
            || opening.response_mask.iter().any(|&mask| mask != 0)
            || t.decode_bytes(&user_ids) != format!(" {body}").as_bytes()
        {
            return Err("SQuAD original opening mismatch".into());
        }
        bump(&mut squad_counts, "official_rows_rejoined");
        if row["is_impossible"] == true {
            bump(&mut squad_counts, "published_impossible_only");
            continue;
        }
        let gold = &row["gold_source"];
        let answer = gold["text"].as_str().ok_or("SQuAD answer")?;
        let char_start = gold["char_start"]
            .as_u64()
            .ok_or("SQuAD answer char offset")? as usize;
        let byte_start = passage
            .char_indices()
            .nth(char_start)
            .ok_or("SQuAD answer start")?
            .0;
        let byte_end = byte_start + answer.len();
        if passage.get(byte_start..byte_end) != Some(answer)
            || gold["bytes"] != json!([byte_start, byte_end])
            || !official["answers"]
                .as_array()
                .ok_or("SQuAD official annotations")?
                .iter()
                .any(|a| a["answer_start"] == char_start && a["text"] == answer)
        {
            return Err("SQuAD published annotation mismatch".into());
        }
        let source = ids(&gold["source_ids"])?;
        let start = gold["first_position"]
            .as_u64()
            .ok_or("SQuAD token address")? as usize
            - head.len();
        let left = t.decode_bytes(&user_ids[..start]).len();
        let span = t.decode_bytes(&source);
        if left + usize::from(span.starts_with(b" ")) != byte_start + 1
            || left + span.len() != byte_end + 1
        {
            return Err("SQuAD original byte and token offsets differ".into());
        }
        let reply = ids(&gold["response_ids"])?;
        let target = enc.encode_document(&[
            user,
            Message {
                role: "assistant",
                content: answer,
            },
        ]);
        if target.tokens != [&opening.tokens[..], &reply, &[1]].concat()
            || target.response_mask[..opening.tokens.len()]
                .iter()
                .any(|&mask| mask != 0)
            || target.response_mask[opening.tokens.len()..]
                .iter()
                .any(|&mask| mask != 1)
            || target.tokens.len() > 192
        {
            return Err("SQuAD answer and observed mask-1 EOS mismatch".into());
        }
        bump(&mut squad_counts, "observable_published_positive_eos");
        if source.len() > 8 {
            bump(&mut squad_counts, "over_eight_source_ids");
            continue;
        }
        bump(
            &mut squad_counts,
            screen_stop(&t, &user_ids, start, &source, answer)?,
        );
    }
    if s_questions.len() != prompts.len() {
        return Err("SQuAD TRAIN prompt without truth row".into());
    }

    let train: Value = serde_json::from_reader(io::stdin().lock())?;
    let mut drop_counts = BTreeMap::new();
    let mut prior = BTreeMap::new();
    for record in train.as_object().ok_or("DROP passages")?.values() {
        let passage = record["passage"].as_str().ok_or("passage")?;
        for row in record["qa_pairs"].as_array().ok_or("qa_pairs")? {
            if row["answer"]["spans"].as_array().ok_or("DROP spans")?.len() != 1 {
                continue;
            }
            let Some(c) = pinned_drop::single(&t, &enc, passage, row, &mut prior)? else {
                continue;
            };
            bump(&mut drop_counts, "prior_single_span_cap_and_budget");
            let body = format!("{passage} {}", c.question);
            let user = Message {
                role: "user",
                content: &body,
            };
            let answer = c.answer;
            let opening = enc.encode_assistant_prefix(&[user.clone()]);
            let target = enc.encode_document(&[
                user,
                Message {
                    role: "assistant",
                    content: answer,
                },
            ]);
            let suffix = target
                .tokens
                .strip_prefix(opening.tokens.as_slice())
                .ok_or("DROP full prefix mismatch")?;
            let Some(before_eos) = suffix.strip_suffix(&[1]) else {
                return Err("DROP terminal EOS missing".into());
            };
            if target.response_mask[..opening.tokens.len()]
                .iter()
                .any(|&mask| mask != 0)
                || target.response_mask[opening.tokens.len()..]
                    .iter()
                    .any(|&mask| mask != 1)
            {
                return Err("DROP mask mismatch".into());
            }
            if target.tokens.len() > 192 {
                bump(&mut drop_counts, "over_window_with_exact_reply");
                continue;
            }
            if before_eos == c.ids || before_eos.strip_prefix(&[223]) == Some(&c.ids) {
                let user_ids = t.encode(&format!(" {body}"));
                bump(&mut drop_counts, "observable_published_positive_eos");
                bump(
                    &mut drop_counts,
                    screen_stop(&t, &user_ids, c.start - head.len(), &c.ids, answer)?,
                );
            } else {
                bump(&mut drop_counts, "opening_ids_mismatch");
            }
        }
    }
    if prior.get("single_copy_ids_and_eos_fit") != Some(&276) {
        return Err("DROP pinned single cohort mismatch".into());
    }
    let mut drop_control_counts = BTreeMap::new();
    let exported = fs::read_to_string("drop-control.private.jsonl")?;
    for line in exported.lines() {
        let control: Value = serde_json::from_str(line)?;
        let body = control["text"].as_str().ok_or("DROP control user text")?;
        let user = Message {
            role: "user",
            content: body,
        };
        let opening = enc.encode_assistant_prefix(&[user.clone()]);
        let user_ids = t.encode(&format!(" {body}"));
        if opening.tokens != ids(&control["prompt_ids"])?
            || opening.tokens != [&head[..], &user_ids, &MARKER].concat()
        {
            return Err("DROP control original prefix differs".into());
        }
        let mut answer_parts = Vec::new();
        let mut exact_spans = Vec::new();
        let mut admitted = true;
        for key in ["first", "second"] {
            let span = &control[key];
            let source = ids(&span["ids"])?;
            let start = span["start"]
                .as_u64()
                .ok_or("DROP control original address")? as usize;
            if span["end"] != start + source.len() || source.len() > 8 {
                return Err("DROP control span width differs".into());
            }
            let actual = String::from_utf8(t.decode_bytes(&source))?;
            let answer = actual.strip_prefix(' ').unwrap_or(&actual);
            let status = screen_stop(&t, &user_ids, start - head.len(), &source, answer)?;
            bump(&mut drop_control_counts, status);
            admitted &= matches!(status, "eligible_multiple_ids" | "eligible_one_id");
            answer_parts.push(answer.to_owned());
            exact_spans.extend(source);
        }
        exact_spans.push(1);
        let answer = answer_parts.join(" ");
        let full = enc.encode_document(&[
            user,
            Message {
                role: "assistant",
                content: &answer,
            },
        ]);
        if exact_spans != ids(&control["expected_ids_through_eos"])?
            || full.tokens != [&opening.tokens[..], &exact_spans].concat()
            || full.response_mask[opening.tokens.len()..]
                .iter()
                .any(|&mask| mask != 1)
            || full.tokens.len() > 192
            || control["first"]["end"].as_u64() > control["second"]["start"].as_u64()
        {
            return Err("DROP control reply protocol, budget or disjoint source differs".into());
        }
        bump(&mut drop_control_counts, "canonical_annotation_order_eos");
        if admitted {
            bump(
                &mut drop_control_counts,
                "strict_unique_copyable_continuations",
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "squad": squad_counts,
            "drop": drop_counts,
            "drop_canonical_continuation": drop_control_counts
        }))?
    );
    Ok(())
}
