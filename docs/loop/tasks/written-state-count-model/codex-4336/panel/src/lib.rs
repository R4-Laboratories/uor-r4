#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
use uor_r4_tokenizer::{
    dialogue::{DialogueProtocol, Message},
    ByteBpeTokenizer,
};

pub mod capture;
pub mod grade;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const ARTIFACT_SHA: &str = "5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa";
pub const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
pub const OLD_CHAT_SHA: &str = "87cfb3ca166a313c983a9e217ae3474ad92ce433aae4daf4fe2006fbd1d305d3";
pub const DECODE_LIMIT: usize = 16;

pub fn require(condition: bool, message: impl FnOnce() -> String) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message().into())
    }
}

pub fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut state = Sha256::new();
    std::io::copy(&mut file, &mut state)?;
    Ok(hex::encode(state.finalize()))
}

pub fn require_digest(path: &Path, expected: &str) -> Result<()> {
    require(digest(path)? == expected, || {
        format!("wrong digest for {}", path.display())
    })
}

pub fn tokenizer(path: &Path) -> Result<ByteBpeTokenizer> {
    require_digest(path, TOKENIZER_SHA)?;
    ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(path)?)
        .ok_or_else(|| "invalid bound tokenizer".into())
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut writer = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer(&mut writer, value)?;
    writeln!(writer)?;
    Ok(writer.flush()?)
}

pub fn read_lines<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

pub fn write_lines(path: &Path, values: &[impl Serialize]) -> Result<()> {
    let mut writer = BufWriter::new(File::create_new(path)?);
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writeln!(writer)?;
    }
    Ok(writer.flush()?)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Authored {
    families: Vec<Family>,
    unsupported: Vec<Unsupported>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Family {
    length: usize,
    first_label: String,
    rival_label: String,
    second_label: String,
    first: String,
    rival: String,
    second: String,
    stop: [String; 2],
    continuation: [String; 2],
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Unsupported {
    length: usize,
    kind: String,
    first_label: String,
    question: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Key {
    pub id: String,
    pub pair: Option<String>,
    pub request: String,
    pub length: usize,
    pub order: usize,
    pub wording: usize,
    pub first: Span,
    pub rival: Span,
    pub second: Span,
    pub expected_text: Option<String>,
    pub expected_ids: Option<Vec<u32>>,
    pub expected_addresses: Vec<usize>,
    pub prompt_ids: Vec<u32>,
    pub user_bounds: [usize; 2],
}

#[derive(Clone, Deserialize, PartialEq, Eq, Serialize)]
pub struct Prompt {
    pub id: String,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Job {
    pub id: String,
    pub arm: String,
}

fn render(family: &Family, first_label: &str, order: usize, question: &str) -> String {
    let first = format!("{first_label} {}.", family.first);
    let rival = format!("{} {}.", family.rival_label, family.rival);
    let second = format!("{} {}.", family.second_label, family.second);
    let facts = if order == 0 {
        [first, rival, second]
    } else {
        [rival, first, second]
    };
    format!("{}\n\n{question}", facts.join("\n"))
}

fn locate(
    tokenizer: &ByteBpeTokenizer,
    prompt: &Prompt,
    prompt_ids: &[u32],
    user_bounds: [usize; 2],
    value: &str,
) -> Result<Span> {
    let [user_start, user_end] = user_bounds;
    let rendered = format!(" {}", prompt.text);
    let (facts, _) = rendered
        .rsplit_once("\n\n")
        .ok_or("missing question boundary")?;
    let surface = format!(" {value}");
    let mut matches = facts.match_indices(&surface);
    let (byte_start, _) = matches.next().ok_or("authored value absent from facts")?;
    require(matches.next().is_none(), || {
        format!("{}: nonunique surface", prompt.id)
    })?;
    let user = &prompt_ids[user_start..user_end];
    require(tokenizer.decode_bytes(user) == rendered.as_bytes(), || {
        format!(
            "{}: protocol user range differs from the raw input",
            prompt.id
        )
    })?;
    let ids = tokenizer.encode(&surface);
    require(tokenizer.decode_bytes(&ids) == surface.as_bytes(), || {
        format!("{}: authored value does not round trip", prompt.id)
    })?;
    let mut cursor = 0;
    for (offset, &id) in user.iter().enumerate() {
        if cursor == byte_start {
            let start = user_start + offset;
            let end = start + ids.len();
            require(
                end <= user_end && prompt_ids.get(start..end) == Some(ids.as_slice()),
                || {
                    format!(
                        "{}: authored value does not align to original tokenizer positions",
                        prompt.id
                    )
                },
            )?;
            return Ok(Span { start, end, ids });
        }
        cursor += tokenizer.decode_bytes(&[id]).len();
    }
    Err(format!(
        "{}: value has no address at its actual byte start",
        prompt.id
    )
    .into())
}

fn derive(tokenizer: &ByteBpeTokenizer, authored: &Authored) -> Result<(Vec<Prompt>, Vec<Key>)> {
    require(
        authored.families.iter().map(|f| f.length).eq([7, 8, 9]),
        || "authored panel needs the three frozen length strata".into(),
    )?;
    require(
        authored
            .unsupported
            .iter()
            .map(|u| (u.length, u.kind.as_str()))
            .eq([(7, "unspecified"), (8, "denied"), (9, "excluded")]),
        || "authored panel needs all three frozen unsupported cases".into(),
    )?;
    let protocol = DialogueProtocol::literal_roles_version(tokenizer, 2)?;
    let encoder = protocol.bind(tokenizer)?;
    let mut prompts = Vec::new();
    let mut keys = Vec::new();

    let mut add = |id: String,
                   family: &Family,
                   pair: Option<String>,
                   request: &str,
                   order: usize,
                   wording: usize,
                   first_label: &str,
                   question: &str|
     -> Result<()> {
        let prompt = Prompt {
            id,
            text: render(family, first_label, order, question),
        };
        let opening = encoder
            .encode_single_user_opening(&prompt.text)
            .ok_or("invalid user message")?;
        let usual = encoder.encode_assistant_prefix(&[Message {
            role: "user",
            content: &prompt.text,
        }]);
        let prompt_ids = opening.tokens();
        require(prompt_ids == usual.tokens, || {
            format!("{}: protocol encoders disagree", prompt.id)
        })?;
        let user_bounds = [opening.user_range().start, opening.user_range().end];
        let first = locate(tokenizer, &prompt, prompt_ids, user_bounds, &family.first)?;
        let rival = locate(tokenizer, &prompt, prompt_ids, user_bounds, &family.rival)?;
        let second = locate(tokenizer, &prompt, prompt_ids, user_bounds, &family.second)?;
        require(first.ids.len() == family.length, || {
            format!(
                "{}: intended {} first IDs, got {}",
                prompt.id,
                family.length,
                first.ids.len()
            )
        })?;
        require(
            first.ids.first().is_some()
                && first.ids.first() == rival.ids.first()
                && first.ids != rival.ids
                && first.start != rival.start
                && !second.ids.is_empty()
                && second.ids.len() <= 8
                && [
                    first.ids.as_slice(),
                    rival.ids.as_slice(),
                    second.ids.as_slice(),
                ]
                .into_iter()
                .flatten()
                .all(|id| ![16, 1, 0, 201].contains(id)),
            || {
                format!(
                    "{}: unmatched same-first-ID rival or in-span delimiter",
                    prompt.id
                )
            },
        )?;
        require(prompt_ids.get(first.end) == Some(&16), || {
            format!(
                "{}: first span needs a period immediately beyond its authenticated interval",
                prompt.id
            )
        })?;

        let expected_text = match request {
            "stop" => Some(format!(" {}", family.first)),
            "continue" => Some(format!(" {} {}", family.first, family.second)),
            _ => None,
        };
        let mut expected_addresses = Vec::new();
        let expected_ids = expected_text.as_ref().map(|text| {
            let mut ids = tokenizer.encode(text);
            ids.push(protocol.eos_id);
            ids
        });
        if let Some(ids) = &expected_ids {
            expected_addresses.extend(first.start..first.end);
            let mut parts = first.ids.clone();
            if request == "continue" {
                expected_addresses.extend(second.start..second.end);
                parts.extend(&second.ids);
            }
            parts.push(protocol.eos_id);
            require(&parts == ids && ids.len() < DECODE_LIMIT, || {
                format!(
                    "{}: raw target differs from concatenated source tokens or exceeds budget",
                    prompt.id
                )
            })?;
        }
        keys.push(Key {
            id: prompt.id.clone(),
            pair,
            request: request.into(),
            length: family.length,
            order,
            wording,
            first,
            rival,
            second,
            expected_text,
            expected_ids,
            expected_addresses,
            prompt_ids: prompt_ids.to_vec(),
            user_bounds,
        });
        prompts.push(prompt);
        Ok(())
    };

    for family in &authored.families {
        for order in 0..2 {
            for wording in 0..2 {
                let pair = format!("l{}-o{order}-w{wording}", family.length);
                add(
                    format!("{pair}-stop"),
                    family,
                    Some(pair.clone()),
                    "stop",
                    order,
                    wording,
                    &family.first_label,
                    &family.stop[wording],
                )?;
                add(
                    format!("{pair}-continue"),
                    family,
                    Some(pair.clone()),
                    "continue",
                    order,
                    wording,
                    &family.first_label,
                    &family.continuation[wording],
                )?;
            }
        }
    }
    for (family, unsupported) in authored.families.iter().zip(&authored.unsupported) {
        add(
            format!("l{}-{}", family.length, unsupported.kind),
            family,
            None,
            &unsupported.kind,
            0,
            0,
            &unsupported.first_label,
            &unsupported.question,
        )?;
    }
    Ok((prompts, keys))
}

fn bounded_span(key: &Key, start: usize, end: usize) -> Span {
    Span {
        start,
        end,
        ids: key.prompt_ids[start..end].to_vec(),
    }
}

fn claim(key: &Key, arm: &str) -> Value {
    let supported = key.pair.is_some();
    let continue_request = key.request == "continue";
    let mut authority_first = supported.then(|| key.first.clone());
    let authority_second = supported.then(|| key.second.clone());
    let mut prompt_ids = key.prompt_ids.clone();
    let mut first = Some(key.first.clone());
    let mut second = Some(key.second.clone());
    let mut action = if continue_request { "continue" } else { "stop" };
    match arm {
        "native" => action = "native",
        "swapped" => action = if continue_request { "stop" } else { "continue" },
        "rival_address" => first = Some(key.rival.clone()),
        "early_end" => first = Some(bounded_span(key, key.first.start, key.first.end - 1)),
        "late_end" => first = Some(bounded_span(key, key.first.start, key.first.end + 1)),
        "missing_claim" => first = None,
        "missing_proof" => authority_first = None,
        "wrong_prompt" => prompt_ids[0] = prompt_ids[0].wrapping_add(1),
        "wrong_second" => second = Some(key.rival.clone()),
        "delimiter" => {
            let span = bounded_span(key, key.first.start, key.first.end + 1);
            first = Some(span.clone());
            authority_first = Some(span);
        }
        _ => {}
    }
    json!({
        "authority": { "prompt_ids":prompt_ids, "first":authority_first, "second":authority_second },
        "claim": { "first":first, "second":second, "first_action":action, "second_action":"stop" }
    })
}

fn schedule(keys: &[Key]) -> Vec<Job> {
    let mut jobs = Vec::new();
    for key in keys {
        let mut arms = vec!["historical_auth", "base_auth", "new_auth"];
        let anchor = key.order == 0 && key.wording == 0;
        if anchor {
            arms.extend([
                "historical_stock",
                "base_stock",
                "new_stock",
                "new_auth_quiet",
            ]);
        }
        if key.pair.is_some() {
            arms.extend(["oracle", "native", "swapped"]);
            if anchor {
                arms.extend([
                    "oracle_quiet",
                    "rival_address",
                    "early_end",
                    "late_end",
                    "missing_claim",
                    "missing_proof",
                    "wrong_prompt",
                ]);
                if key.request == "continue" && key.length < 9 {
                    arms.push("wrong_second");
                }
                if key.length == 7 {
                    arms.push("delimiter");
                }
            }
        } else {
            arms.push("unsupported_claim");
        }
        jobs.extend(arms.into_iter().map(|arm| Job {
            id: key.id.clone(),
            arm: arm.into(),
        }));
    }
    jobs
}

pub fn freeze(tokenizer: &ByteBpeTokenizer, author: &Path, out: &Path) -> Result<()> {
    let authored: Authored = read_json(author)?;
    let (prompts, keys) = derive(tokenizer, &authored)?;
    let jobs = schedule(&keys);
    fs::create_dir(out)?;
    fs::create_dir(out.join("claims"))?;
    write_json(&out.join("author.json"), &authored)?;
    write_lines(&out.join("prompts.jsonl"), &prompts)?;
    write_lines(&out.join("key.jsonl"), &keys)?;
    write_lines(&out.join("jobs.jsonl"), &jobs)?;
    for job in &jobs {
        if !capture::baseline(&job.arm) {
            let key = keys
                .iter()
                .find(|key| key.id == job.id)
                .ok_or("missing frozen key")?;
            write_json(&claim_path(out, job), &claim(key, &job.arm))?;
        }
    }
    Ok(())
}

pub fn claim_path(panel: &Path, job: &Job) -> std::path::PathBuf {
    panel
        .join("claims")
        .join(format!("{}.{}.json", job.id, job.arm))
}

pub fn open_panel(
    tokenizer: &ByteBpeTokenizer,
    panel: &Path,
) -> Result<(Vec<Prompt>, Vec<Key>, Vec<Job>)> {
    let authored = read_json(&panel.join("author.json"))?;
    let (prompts, keys) = derive(tokenizer, &authored)?;
    let jobs = schedule(&keys);
    require(
        prompts == read_lines::<Prompt>(&panel.join("prompts.jsonl"))?
            && keys == read_lines::<Key>(&panel.join("key.jsonl"))?
            && jobs == read_lines::<Job>(&panel.join("jobs.jsonl"))?,
        || "sealed panel metadata differs from protocol-derived grade".into(),
    )?;
    Ok((prompts, keys, jobs))
}
