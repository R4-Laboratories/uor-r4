use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{self, Write},
};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
    ByteBpeTokenizer,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    ids: Vec<u32>,
}
struct Case<'a> {
    id: &'a str,
    question: &'a str,
    answers: Vec<&'a str>,
    spans: Vec<Span>,
    prefix: usize,
    occurrences: Vec<usize>,
}
fn case<'a>(
    t: &ByteBpeTokenizer,
    enc: &DialogueEncoder<'_>,
    passage: &str,
    row: &'a Value,
    count: &mut BTreeMap<String, usize>,
) -> Result<Option<Case<'a>>, &'static str> {
    let entries = row["answer"]["spans"].as_array().ok_or("spans")?;
    if entries.is_empty() {
        return Ok(None);
    }
    let len = entries.len();
    let group = if len == 1 {
        "single"
    } else if len == 2 {
        "two"
    } else {
        "three_plus"
    };
    let mut stage = |event: &str| {
        *count.entry(format!("{group}_{event}")).or_default() += 1;
    };
    stage("annotated");
    let question = row["question"].as_str().ok_or("question")?;
    let id = row["query_id"].as_str().ok_or("query id")?;
    let body = format!("{passage} {question}");
    let open = enc.encode_assistant_prefix(&[Message {
        role: "user",
        content: &body,
    }]);
    let prefix = open.tokens.len();
    if prefix >= 192 {
        return Ok(None);
    }
    stage("prompt_fits");
    let delimiter_body = format!(" {body}");
    let user_ids = t.encode(&delimiter_body);
    let head = [vec![0], t.encode("User:")].concat();
    if open.tokens != [&head[..], &user_ids, &[201, 35, 560, 652, 714, 28]].concat()
        || t.decode_bytes(&user_ids) != delimiter_body.as_bytes()
    {
        return Ok(None);
    }
    stage("verbatim_user");
    let mut offsets = Vec::with_capacity(user_ids.len());
    let mut from = 0;
    for &word in &user_ids {
        let bytes = t.decode_bytes(&[word]);
        let leading = bytes.iter().take_while(|b| b.is_ascii_whitespace()).count();
        let to = from + bytes.len();
        offsets.push((from + leading, to));
        from = to;
    }
    let answers = entries
        .iter()
        .map(|a| a.as_str().ok_or("span text"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut spans = Vec::new();
    let mut occurrences = Vec::new();
    for answer in &answers {
        if answer.is_empty() || answer.trim() != *answer {
            return Ok(None);
        }
        let mut offsets_in_passage = passage
            .char_indices()
            .filter_map(|(start, _)| passage[start..].starts_with(answer).then_some(start));
        let Some(start) = offsets_in_passage.next() else {
            return Ok(None);
        };
        if offsets_in_passage.next().is_some() {
            return Ok(None);
        }
        let Some(first) = offsets
            .iter()
            .position(|&(left, right)| left == start + 1 && right > start + 1)
        else {
            return Ok(None);
        };
        let Some(last) = offsets[first..]
            .iter()
            .position(|&(_, right)| right == start + 1 + answer.len())
            .map(|n| first + n)
        else {
            return Ok(None);
        };
        let ids = user_ids[first..=last].to_vec();
        let target = t.encode(&format!(" {answer}"));
        let phase = target.iter().take_while(|&&id| id == 223).count();
        if ids != target[phase..] || ids.iter().any(|id| [0, 1, 201].contains(id)) {
            return Ok(None);
        }
        occurrences.push(user_ids.iter().filter(|&&token| token == ids[0]).count());
        spans.push(Span {
            start: head.len() + first,
            end: head.len() + last + 1,
            ids,
        });
    }
    let mut ordered: Vec<_> = spans.iter().collect();
    ordered.sort_by_key(|span| span.start);
    if ordered
        .windows(2)
        .any(|neighbor| neighbor[0].end > neighbor[1].start)
    {
        return Ok(None);
    }
    stage("unique_disjoint_original");
    if spans.iter().any(|span| span.ids.len() > 8) {
        return Ok(None);
    }
    stage("each_span_fits_cap");
    if prefix + spans.iter().map(|span| span.ids.len()).sum::<usize>() + 1 > 192 {
        return Ok(None);
    }
    stage("copy_ids_and_eos_fit");
    Ok(Some(Case {
        id,
        question,
        answers,
        spans,
        prefix,
        occurrences,
    }))
}
fn copy_matches(t: &ByteBpeTokenizer, c: &Case<'_>, order: &[usize]) -> bool {
    let response = order
        .iter()
        .map(|&i| c.answers[i])
        .collect::<Vec<_>>()
        .join(" ");
    let actual = order
        .iter()
        .flat_map(|&i| c.spans[i].ids.iter().copied())
        .collect::<Vec<_>>();
    t.encode(&format!(" {response}")) == actual
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let private_export = env::args().nth(2);
    let t = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(
        env::args().nth(1).ok_or("tokenizer")?,
    )?)
    .ok_or("tokenizer")?;
    let enc = DialogueProtocol::literal_roles_v2(&t)?.bind(&t)?;
    let train: Value = serde_json::from_reader(io::stdin().lock())?;
    let mut count = BTreeMap::new();
    let mut matched_singles = BTreeSet::new();
    let mut matched_multis = BTreeSet::new();
    let mut pairs = Vec::new();
    let mut missing_serialized = 0;
    let mut canonical_two_span_queries = 0;
    let mut exports = Vec::new();
    for (passage_id, record) in train.as_object().ok_or("expected DROP passages")? {
        let passage = record["passage"].as_str().ok_or("passage")?;
        let mut candidates = Vec::new();
        for row in record["qa_pairs"].as_array().ok_or("qa_pairs")? {
            if let Some(c) = case(&t, &enc, passage, row, &mut count)? {
                candidates.push(c);
            }
        }
        for multi in candidates.iter().filter(|c| c.spans.len() > 1) {
            let mut source_order = (0..multi.spans.len()).collect::<Vec<_>>();
            source_order.sort_by_key(|&i| multi.spans[i].start);
            if multi.spans.len() == 2 {
                let answer = multi.answers.join(" ");
                let text = format!("{passage} {}", multi.question);
                let user = Message {
                    role: "user",
                    content: &text,
                };
                let assistant = Message {
                    role: "assistant",
                    content: &answer,
                };
                let prompt = enc.encode_assistant_prefix(&[user.clone()]).tokens;
                let target = enc.encode_document(&[user, assistant]).tokens;
                let expected_ids = multi
                    .spans
                    .iter()
                    .flat_map(|span| span.ids.iter().copied())
                    .chain([1])
                    .collect::<Vec<_>>();
                if target.strip_prefix(prompt.as_slice()) == Some(expected_ids.as_slice()) {
                    canonical_two_span_queries += 1;
                    if private_export.is_some() {
                        let spans = multi
                            .spans
                            .iter()
                            .map(|span| json!({"start":span.start,"end":span.end,"ids":span.ids}))
                            .collect::<Vec<_>>();
                        exports.push(json!({
                            "query_id": multi.id, "text": text, "prompt_ids": prompt,
                            "first": spans[0], "second": spans[1],
                            "expected_ids_through_eos": expected_ids,
                        }));
                    }
                }
            }
            for single in candidates
                .iter()
                .filter(|c| c.spans.len() == 1 && c.question != multi.question)
            {
                let shared = &single.spans[0];
                let listed_first = multi.spans[0] == *shared;
                let source_first = multi.spans[source_order[0]] == *shared;
                if !(listed_first || source_first) {
                    continue;
                }
                let annotation_order = (0..multi.spans.len()).collect::<Vec<_>>();
                let annotation_copy = listed_first && copy_matches(&t, multi, &annotation_order);
                let source_copy = source_first && copy_matches(&t, multi, &source_order);
                let stop_copy = copy_matches(&t, single, &[0]);
                if !(stop_copy && (annotation_copy || source_copy)) {
                    missing_serialized += 1;
                }
                matched_singles.insert(single.id);
                matched_multis.insert(multi.id);
                pairs.push(json!({
                    "passage":passage_id,"single":single.id,"multi":multi.id,
                    "spans":multi.spans.len(),"stop_copy":stop_copy,
                    "listed_first":listed_first,"source_first":source_first,
                    "annotation_copy":annotation_copy,"source_copy":source_copy,
                    "source_first_second_same_id":multi.spans[source_order[0]].ids[0]==multi.spans[source_order[1]].ids[0],
                    "listed_first_second_same_id":multi.spans[0].ids[0]==multi.spans[1].ids[0],
                    "opener_occurrences_single":single.occurrences,
                    "openers_occurrences_multi_listed":multi.occurrences,
                    "prefix_single":single.prefix,"prefix_multi":multi.prefix,
                }));
            }
        }
    }
    if let Some(path) = private_export.filter(|_| !exports.is_empty()) {
        let mut output = fs::File::create_new(path)?;
        for row in exports {
            writeln!(output, "{row}")?;
        }
    }
    println!(
        "{}",
        json!({"stages":count,"canonical_two_span_queries":canonical_two_span_queries,"pair_count":pairs.len(),"single_queries":matched_singles.len(),"multi_queries":matched_multis.len(),"pairs_without_exact_writer_concatenation":missing_serialized,"pairs":pairs})
    );
    Ok(())
}
