#![forbid(unsafe_code)]
use serde::Deserialize;
use serde_json::{json, Value};
use std::{error::Error, fs, path::Path};
use uor_r4_integer::{
    discrete_d11::{ExactCopyState, IntegerDiscreteModel, ROLE_COPY_STOPS},
    paired_role::RoleSpan,
};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Site {
    pos: usize,
    id: u32,
}
#[derive(Deserialize)]
struct Row {
    case_id: String,
    prefix: Vec<u32>,
    user_bounds: [usize; 2],
    question_start: usize,
    candidates: Vec<Site>,
}

fn copy_range(ids: &[u32], pos: usize, fence: usize) -> Result<(usize, u32, bool)> {
    let mut cursor = ExactCopyState::new(Some(pos as u32), fence as u32);
    loop {
        let step = cursor
            .write(ids, 0, ROLE_COPY_STOPS)?
            .finish_with_eos(true, 1);
        if step.copied == 0 {
            return Ok((cursor.cursor as usize, step.release, step.completed_span()));
        }
        if step.exact_id != ids[cursor.cursor as usize] || step.emitted != step.exact_id {
            return Err("copy did not emit its original ID".into());
        }
        cursor = step.state;
    }
}

fn intervals(model: &IntegerDiscreteModel, row: &Row) -> Result<(Value, Vec<Value>)> {
    let ids = &row.prefix;
    let [start, end] = row.user_bounds;
    let sites = model.ordered_role_sites()?;
    let projection = sites.prompt(ids, 1..ids.len())?;
    let words: Vec<RoleSpan> = projection
        .candidates()
        .filter(|w| w.start >= start && w.end <= end)
        .collect();
    let tail_start = start.max(end.saturating_sub(32));
    let search_start = tail_start.min(row.question_start);
    let mut tail_equal = 0;
    let mut separator_equal = 0;
    let mut out = Vec::new();
    for query in words.iter().filter(|q| q.start >= search_start) {
        for fact in words.iter().filter(|f| f.end <= query.start) {
            if ids[fact.start..fact.end] != ids[query.start..query.end] {
                continue;
            }
            tail_equal += usize::from(fact.end <= tail_start && query.start >= tail_start);
            separator_equal +=
                usize::from(fact.end <= row.question_start && query.start >= row.question_start);
            let viable = match model.contextual_source_candidates(ids, *fact, *query) {
                Ok(candidates) => candidates,
                Err(err)
                    if err
                        .to_string()
                        .contains("requires an original D21 bound assertion") =>
                {
                    continue
                }
                Err(err) => return Err(err.into()),
            };
            let (selected, trace) = model.contextual_source_trace(ids, *fact, *query)?;
            if trace.len() != viable.len()
                || trace.iter().zip(&viable).any(|((a, _), b)| {
                    a.completed_source_range != b.completed_source_range || a.span != b.span
                })
            {
                return Err("product candidate and scoring traces disagree".into());
            }
            out.push(json!({"query":[query.start,query.end],"fact":[fact.start,fact.end],
                "ranges":viable.iter().map(|v|v.completed_source_range).collect::<Vec<_>>(),"selected":selected.source}));
        }
    }
    Ok((json!({"tail":tail_equal,"separator":separator_equal}), out))
}

fn observe(
    models: &[IntegerDiscreteModel],
    tokenizer: &ByteBpeTokenizer,
    row: Row,
) -> Result<Value> {
    let [start, end] = row.user_bounds;
    let ids = &row.prefix;
    if !(0 < start && start < row.question_start && row.question_start < end && end < ids.len()) {
        return Err("unexpected authenticated prefix range".into());
    }
    let mut native = Vec::new();
    for model in models {
        let decision = model.paired_copy_decision(ids, true)?;
        let disabled = model.paired_copy_decision(ids, false)?;
        if disabled.as_ref().is_some_and(|d| d.choice.source.is_some()) {
            return Err("disabled source unexpectedly selected".into());
        }
        native.push(json!({"artifact":model.artifact_sha256(),"schema":model.artifact_schema(),
            "bound_source_bank":model.contextual_source_bindings().is_some(),
            "requested_framing":model.requested_role_framing().is_some(),"paired_decision":decision}));
    }
    let (equal_words, bound) = intervals(&models[2], &row)?;
    // The canonical single-user serializer prepends one space to the raw text.
    let mut byte_cursor = 0usize;
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for &id in &ids[start..] {
        let bytes = tokenizer.decode_bytes(&[id]);
        let spaces = bytes.iter().take_while(|b| b.is_ascii_whitespace()).count();
        starts.push((byte_cursor + spaces).saturating_sub(1));
        byte_cursor += bytes.len();
        ends.push(byte_cursor.saturating_sub(1));
    }
    let mut releases = Vec::new();
    for site in row.candidates {
        if site.pos < start || site.pos >= end || ids[site.pos] != site.id {
            return Err("site identity changed".into());
        }
        let (at, reason, done) = copy_range(ids, site.pos, ids.len())?;
        if at <= site.pos {
            return Err("non-lexical input site".into());
        }
        releases.push(
            json!({"pos":site.pos,"id":site.id,"end":at,"release":reason,"complete":done,
            "bytes":[starts[site.pos-start],ends[at-1-start]]}),
        );
    }
    Ok(
        json!({"case_id":row.case_id,"tail_cut":start.max(end.saturating_sub(32)),
        "separator":row.question_start,"native":native,"equal_word_pairs":equal_words,"bound":bound,"copies":releases}),
    )
}

#[derive(Deserialize)]
struct GoldSite {
    pos: usize,
    id: u32,
    bytes: [usize; 2],
}
#[derive(Deserialize)]
struct Key {
    case_id: String,
    target: Option<GoldSite>,
}

fn endpoint_replay(tokenizer: &ByteBpeTokenizer, input: &str, key_file: &str) -> Result<()> {
    let mut keys = std::collections::BTreeMap::new();
    for line in fs::read_to_string(key_file)?.lines() {
        let key: Key = serde_json::from_str(line)?;
        if keys.insert(key.case_id, key.target).is_some() {
            return Err("duplicate key".into());
        }
    }
    for line in input.lines() {
        let row: Row = serde_json::from_str(line)?;
        let Some(site) = keys.remove(&row.case_id).ok_or("unmatched identity")? else {
            continue;
        };
        let [user_start, user_end] = row.user_bounds;
        let raw = tokenizer.decode_bytes(&row.prefix[user_start..user_end]);
        let prompt = std::str::from_utf8(
            raw.strip_prefix(b" ")
                .ok_or("opening lacks canonical space")?,
        )?;
        let answer = prompt
            .get(site.bytes[0]..site.bytes[1])
            .ok_or("invalid byte span")?;
        let tokens = tokenizer.encode(&format!(" {answer}"));
        let end = site.pos + tokens.len();
        if tokens.first() != Some(&site.id)
            || row.prefix.get(site.pos..end) != Some(tokens.as_slice())
            || end >= user_end
        {
            return Err("gold does not match original token interval".into());
        }
        let mut arms = Vec::new();
        for (arm, fence) in [
            ("original_prompt", row.prefix.len()),
            ("exact_end", end),
            ("one_beyond", end + 1),
        ] {
            let (at, release, complete) = copy_range(&row.prefix, site.pos, fence)?;
            arms.push(json!({"arm":arm,"copied":at-site.pos,"release":release,"exact_words":at==end,
                "native_certified_eos":complete,"exact_words_and_certified_eos":complete && at==end}));
        }
        println!(
            "{}",
            json!({"case_id":row.case_id,"start":site.pos,"end":end,"gold_tokens":tokens.len(),"arms":arms})
        );
    }
    if !keys.is_empty() {
        return Err("missing capture".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(args.len() == 6 || (args.len() == 7 && args[4] == "endpoints")) {
        return Err("usage: probe BALANCED FROZEN D26 TOKENIZER MODE JSON_INPUT".into());
    }
    let models = [0, 1, 2]
        .map(|i| IntegerDiscreteModel::load(Path::new(&args[i])))
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(&args[3])?)
        .ok_or("invalid pinned tokenizer")?;
    let input = fs::read_to_string(&args[5])?;
    match args[4].as_str() {
        "endpoints" if args.len() == 7 => endpoint_replay(&tokenizer, &input, &args[6])?,
        "capture" => {
            for line in input.lines() {
                println!(
                    "{}",
                    observe(&models, &tokenizer, serde_json::from_str(line)?)?
                )
            }
        }
        "control" => {
            let record: Value = serde_json::from_str(&input)?;
            let ids: Vec<u32> = serde_json::from_value(record["prompt_ids"].clone())?;
            let decision = models[2]
                .paired_copy_decision(&ids, true)?
                .ok_or("D26 positive had no binder")?;
            if serde_json::to_value(&decision)? != record["paired"]
                || decision.choice.source != Some(51)
                || record["artifact_sha256"] != models[2].artifact_sha256()
            {
                return Err("historical positive replay differs".into());
            }
            println!(
                "{}",
                json!({"model":models[2].artifact_sha256(),"paired":decision})
            );
        }
        _ => return Err("unknown mode".into()),
    }
    Ok(())
}
