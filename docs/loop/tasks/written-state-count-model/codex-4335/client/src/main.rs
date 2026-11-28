#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};
use uor_r4_tokenizer::{
    dialogue::{DialogueProtocol, Message},
    ByteBpeTokenizer,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CAP: usize = 16;
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

#[derive(Deserialize)]
struct Input {
    case_id: String,
    prefix: Vec<u32>,
    user_bounds: [usize; 2],
    question_start: usize,
}

#[derive(Deserialize)]
struct Prompt {
    case_id: String,
    text: String,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
struct Site {
    pos: usize,
    id: u32,
    bytes: [usize; 2],
}

#[derive(Deserialize)]
struct Key {
    case_id: String,
    sites: [Site; 2],
    target: Option<Site>,
}

#[derive(Deserialize)]
struct Trace {
    case_id: String,
    arm: String,
    client: Client,
}

#[derive(Deserialize)]
struct Client {
    schema: String,
    protocol: String,
    artifact_sha256: String,
    tokenizer_sha256: String,
    generated_ids: Vec<u32>,
    prompt_ids: Vec<u32>,
    stop: String,
    text: String,
    #[serde(default)]
    authenticated_copy_steps: Vec<Copy>,
    #[serde(default)]
    native_occurrence_steps: Vec<Occurrence>,
    #[serde(default)]
    native_period_release: Value,
    #[serde(default)]
    selected: Value,
    #[serde(default)]
    paired: Value,
    #[serde(default)]
    copy_step_words: Vec<[u32; 8]>,
}

#[derive(Deserialize)]
struct Copy {
    read_pos: Option<usize>,
    claim_pos: Option<usize>,
    source_id: Option<u32>,
    committed_id: u32,
    copied: bool,
}

#[derive(Deserialize)]
struct Occurrence {
    decision: String,
    source_pos: Option<usize>,
    claim_pos: Option<usize>,
    committed_id: u32,
    copied: bool,
}

const ARMS: [(&str, &str); 3] = [
    (
        "balanced",
        "5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa",
    ),
    (
        "frozen",
        "6415f6cb9fbf4694b449c08170d17c7e821c168d28e811cfa4e7a3448a333422",
    ),
    (
        "contextual",
        "b17352679c64a96bf657fad7640d4bb05ea3de0ecd08dadd13e9d2756e979bc4",
    ),
];

fn require(condition: bool, message: &'static str) -> Result<()> {
    if !condition {
        return Err(message.into());
    }
    Ok(())
}

fn load<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn expected_source(
    tokenizer: &ByteBpeTokenizer,
    input: &Input,
    text: &str,
    site: Site,
) -> Result<Vec<u32>> {
    let [user_start, user_end] = input.user_bounds;
    let [start, end] = site.bytes;
    let (passage, _) = text
        .rsplit_once("\n\n")
        .ok_or("missing passage separator")?;
    let span = passage
        .get(start..end)
        .ok_or("invalid target UTF-8 byte span")?;
    let rendered = format!(" {text}");
    let user = &input.prefix[user_start..user_end];
    let index = site
        .pos
        .checked_sub(user_start)
        .ok_or("source precedes user span")?;
    let preceding = user
        .get(..index)
        .ok_or("source outside original user span")?;
    let cursor = tokenizer.decode_bytes(preceding).len();
    let source = tokenizer.encode(&format!(" {span}"));
    let first_bytes = tokenizer.decode_bytes(&[site.id]);
    let leading = first_bytes
        .iter()
        .take_while(|byte| byte.is_ascii_whitespace())
        .count();
    let source_end = site
        .pos
        .checked_add(source.len())
        .ok_or("source range overflow")?;
    require(
        !span.is_empty()
            && tokenizer.decode_bytes(user) == rendered.as_bytes()
            && cursor + leading == start + 1
            && source_end <= input.question_start
            && source.first() == Some(&site.id)
            && input.prefix.get(site.pos..source_end) == Some(source.as_slice())
            && tokenizer.decode_bytes(&source) == format!(" {span}").as_bytes()
            && cursor + tokenizer.decode_bytes(&source).len() == end + 1,
        "canonical answer IDs do not authenticate to the original byte span and address",
    )?;
    Ok(source)
}

#[derive(Serialize)]
struct Writer {
    decision: String,
    selected_address: Option<usize>,
    claimed_address: Option<usize>,
    committed_address: Option<usize>,
    period_release: Value,
}

fn ordinary_writer(arm: &str, client: &Client) -> Result<Writer> {
    let ids = &client.generated_ids;
    if arm == "contextual" {
        require(
            client.selected == client.paired["choice"]
                && client.copy_step_words.len() == ids.len()
                && client.contextual_writes_valid(),
            "contextual copy trace diverges from committed IDs or original bounds",
        )?;
        let [_, source, _, _, _, exact, copied, _] = client.copy_step_words[0];
        let selected = (source != u32::MAX).then_some(source as usize);
        let committed = (copied == 1).then_some(source as usize);
        require(
            selected.map(|pos| json!(pos)).unwrap_or(Value::Null) == client.selected["source"]
                && committed.is_none_or(|pos| {
                    client.prompt_ids.get(pos) == Some(&exact) && exact == ids[0]
                }),
            "contextual source selection or first committed copy claim differs",
        )?;
        Ok(Writer {
            decision: if selected.is_some() {
                "selected"
            } else {
                "no_source"
            }
            .into(),
            selected_address: selected,
            claimed_address: selected,
            committed_address: committed,
            period_release: Value::Null,
        })
    } else {
        require(
            client.authenticated_copy_steps.len() == ids.len()
                && client.native_occurrence_steps.len() == ids.len()
                && ids
                    .iter()
                    .zip(&client.authenticated_copy_steps)
                    .zip(&client.native_occurrence_steps)
                    .all(|((&id, copy), occurrence)| {
                        id == copy.committed_id
                            && id == occurrence.committed_id
                            && copy.copied == occurrence.copied
                            && copy.read_pos == occurrence.source_pos
                            && copy.claim_pos == occurrence.claim_pos
                    }),
            "authenticated writer trace diverges from committed IDs or occurrence trace",
        )?;
        let first = &client.authenticated_copy_steps[0];
        let occurrence = &client.native_occurrence_steps[0];
        require(
            !first.copied
                || first.claim_pos.is_some_and(|pos| {
                    client.prompt_ids.get(pos) == Some(&ids[0]) && first.source_id == Some(ids[0])
                }),
            "authenticated first committed copy claim differs from the original token",
        )?;
        Ok(Writer {
            decision: occurrence.decision.clone(),
            selected_address: first.read_pos,
            claimed_address: first.claim_pos,
            committed_address: first.claim_pos.filter(|_| first.copied),
            period_release: client.native_period_release.clone(),
        })
    }
}

impl Client {
    fn contextual_writes_valid(&self) -> bool {
        self.copy_step_words.iter().zip(&self.generated_ids).all(
            |(&[active, _, cursor, end, emitted, exact, copied, release], &id)| {
                active <= 1
                    && copied <= 1
                    && release <= 3
                    && emitted == id
                    && end as usize == self.prompt_ids.len()
                    && if copied == 1 {
                        cursor
                            .checked_sub(1)
                            .is_some_and(|pos| self.prompt_ids.get(pos as usize) == Some(&exact))
                            && exact == id
                    } else {
                        exact == u32::MAX
                    }
            },
        )
    }
}

fn score(
    tokenizer: &ByteBpeTokenizer,
    protocol: &DialogueProtocol,
    trace: Trace,
    input: &Input,
    text: &str,
    target: Option<Site>,
    arm: &str,
    digest: &str,
) -> Result<Value> {
    let client = trace.client;
    let ids = &client.generated_ids;
    let eos = protocol.eos_id;
    require(
        trace.arm == arm
            && trace.case_id == input.case_id
            && client.schema == "uor-r4.uor-chat-discrete-text/1"
            && client.protocol == protocol.schema
            && client.artifact_sha256 == digest
            && client.tokenizer_sha256 == TOKENIZER_SHA
            && client.prompt_ids == input.prefix
            && (1..=CAP).contains(&ids.len()),
        "client identity, artifact, raw prefix or length differs",
    )?;
    let eos_emitted = ids.last() == Some(&eos);
    let visible = ids.strip_suffix(&[eos]).unwrap_or(ids);
    require(
        !visible.contains(&eos)
            && client.text == tokenizer.decode(visible)
            && if eos_emitted {
                client.stop == "eos"
            } else {
                ids.len() == CAP && client.stop == "max_tokens"
            },
        "generated IDs, EOS, stop code and decoded text disagree",
    )?;
    let writer = ordinary_writer(arm, &client)?;
    let grade = if let Some(site) = target {
        let mut expected = expected_source(tokenizer, input, text, site)?;
        require(
            expected.len() + 1 <= CAP && !expected.contains(&eos),
            "target does not fit the cap through EOS",
        )?;
        expected.push(eos);
        let matched = ids
            .iter()
            .zip(&expected)
            .take_while(|(actual, want)| actual == want)
            .count();
        let wrong = (ids != &expected).then(|| {
            json!({
                "index": matched,
                "expected_id": expected.get(matched),
                "emitted_id": ids.get(matched),
            })
        });
        json!({
            "target":site,
            "expected_ids_through_eos":expected,
            "first_id_correct":ids[0] == expected[0],
            "first_address_correct":writer.committed_address == Some(site.pos),
            "matched_prefix_ids":matched,
            "exact_through_eos":ids == &expected,
            "first_wrong":wrong,
        })
    } else {
        Value::Null
    };
    Ok(json!({
        "case_id":input.case_id, "arm":arm, "prefix_ids":input.prefix.len(),
        "first_committed_id":ids[0], "generated_ids":ids,
        "eos_emitted":eos_emitted, "stop":client.stop, "writer":writer, "grade":grade,
    }))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [tokenizer, blind_input, prompts, key, traces] = args.as_slice() else {
        return Err(
            "usage: extractive-client-grade TOKENIZER BLIND_INPUT PROMPTS KEY TRACES_DIR".into(),
        );
    };
    let bytes = fs::read(tokenizer)?;
    require(
        hex::encode(Sha256::digest(&bytes)) == TOKENIZER_SHA,
        "canonical tokenizer digest differs",
    )?;
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes).ok_or("invalid canonical tokenizer")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let encoder = protocol.bind(&tokenizer)?;
    let inputs = load::<Input>(blind_input)?;
    let prompts = load::<Prompt>(prompts)?;
    let keys = load::<Key>(key)?;
    let key_count = keys.len();
    let keys: BTreeMap<_, _> = keys
        .into_iter()
        .map(|key| (key.case_id.clone(), key))
        .collect();
    require(
        inputs.len() == 18
            && prompts.len() == inputs.len()
            && key_count == inputs.len()
            && keys.len() == key_count
            && keys.values().filter(|key| key.target.is_some()).count() == 13,
        "unexpected recent panel sizes or duplicated case IDs",
    )?;
    for (arm, digest) in ARMS {
        let traces = load::<Trace>(Path::new(traces).join(format!("{arm}.jsonl")))?;
        require(
            traces.len() == inputs.len(),
            "client tape has an incorrect record count",
        )?;
        for ((trace, input), prompt) in traces.into_iter().zip(&inputs).zip(&prompts) {
            let key = keys.get(&input.case_id).ok_or("missing panel key")?;
            let [start, end] = input.user_bounds;
            let opening = encoder.encode_assistant_prefix(&[Message {
                role: "user",
                content: &prompt.text,
            }]);
            require(
                prompt.case_id == input.case_id
                    && opening.tokens == input.prefix
                    && start > 0
                    && start <= input.question_start
                    && input.question_start < end
                    && end < input.prefix.len()
                    && key.target.is_none_or(|site| key.sites.contains(&site)),
                "prompt, prefix, user span or grade address does not match the sealed panel",
            )?;
            let result = score(
                &tokenizer,
                &protocol,
                trace,
                input,
                &prompt.text,
                key.target,
                arm,
                digest,
            )
            .map_err(|error| format!("{arm} case {}: {error}", input.case_id))?;
            println!("{result}");
        }
    }
    Ok(())
}
