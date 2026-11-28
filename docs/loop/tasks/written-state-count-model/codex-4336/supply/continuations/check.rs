use std::{
    env, fs,
    io::{self, BufRead},
};

use serde_json::{json, Value};
use uor_r4_tokenizer::{
    dialogue::{DialogueEncoder, DialogueProtocol, Message},
    ByteBpeTokenizer,
};

const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const EOS: u32 = 1;
const SPACE: u32 = 223;
const SOURCE_CAP: usize = 8;
const PREFIX_LIMIT: usize = 180;
const WINDOW: usize = 192;

struct Span {
    start: usize,
    ids: Vec<u32>,
    prefix: usize,
    through_eos: usize,
    leading_space_ids: usize,
    opener_occurrences: usize,
}

impl Span {
    fn receipt(&self) -> Value {
        json!({
            "original_start": self.start,
            "original_end": self.start + self.ids.len(),
            "source_ids": self.ids.len(),
            "prefix_ids": self.prefix,
            "through_eos_ids": self.through_eos,
            "leading_space_ids": self.leading_space_ids,
            "opener_occurrences": self.opener_occurrences,
        })
    }
}

fn check(
    tokenizer: &ByteBpeTokenizer,
    encoder: &DialogueEncoder<'_>,
    context: &str,
    row: &Value,
) -> Result<Span, &'static str> {
    let question = row["question"].as_str().ok_or("question missing")?;
    let answer = row["text"].as_str().ok_or("annotation missing")?;
    let ordinal = row["start"].as_u64().ok_or("source offset missing")? as usize;
    let start = context
        .char_indices()
        .nth(ordinal)
        .ok_or("source offset outside context")?
        .0;
    let end = start + answer.len();
    if context.get(start..end) != Some(answer) {
        return Err("annotation and source bytes differ");
    }
    if [context, question, answer]
        .iter()
        .any(|text| text.trim() != *text || text.contains(['\n', '\r']))
    {
        return Err("input needs trim or line normalization");
    }

    let body = format!("{context} {question}");
    let user = Message {
        role: "user",
        content: &body,
    };
    let assistant = Message {
        role: "assistant",
        content: answer,
    };
    let prefix = encoder.encode_assistant_prefix(&[user.clone()]);
    let complete = encoder.encode_document(&[user, assistant]);
    let body_with_delimiter = format!(" {body}");
    let user_ids = tokenizer.encode(&body_with_delimiter);
    let user_head = [vec![0], tokenizer.encode("User:")].concat();
    if prefix.tokens != [&user_head[..], &user_ids, &MARKER].concat()
        || prefix.emitted_turns != 1
        || complete.emitted_turns != 2
        || prefix.special_token_occurrences != 0
        || complete.special_token_occurrences != 0
        || prefix.response_mask.iter().any(|&mask| mask != 0)
        || complete.response_mask[prefix.tokens.len()..]
            .iter()
            .any(|&mask| mask != 1)
        || tokenizer.decode_bytes(&user_ids) != body_with_delimiter.as_bytes()
    {
        return Err("protocol or user bytes differ");
    }
    let reply = complete
        .tokens
        .strip_prefix(prefix.tokens.as_slice())
        .and_then(|suffix| suffix.strip_suffix(&[EOS]))
        .ok_or("answer prefix or EOS missing")?;
    if tokenizer.decode_bytes(reply) != format!(" {answer}").as_bytes() {
        return Err("answer bytes differ");
    }
    let phase = reply
        .iter()
        .position(|&id| id != SPACE)
        .ok_or("answer is empty")?;

    let mut cursor = 0;
    let mut left = None;
    let mut right = None;
    for (index, &id) in user_ids.iter().enumerate() {
        let bytes = tokenizer.decode_bytes(&[id]);
        let whitespace = bytes
            .iter()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        // The source byte offsets exclude protocol 2's leading delimiter space.
        if cursor + whitespace == start + 1 && cursor + bytes.len() > start + 1 {
            left = Some(index);
        }
        cursor += bytes.len();
        if cursor == end + 1 {
            right = Some(index + 1);
        }
    }
    let first = left.ok_or("source start is not token-aligned")?;
    let stop = right.ok_or("source end is not token-aligned")?;
    let original = user_ids
        .get(first..stop)
        .ok_or("source interval is invalid")?;
    if original != &reply[phase..] {
        return Err("original and reply IDs differ");
    }
    let opener = original.first().ok_or("source interval is empty")?;
    Ok(Span {
        start: user_head.len() + first,
        ids: original.to_vec(),
        prefix: prefix.tokens.len(),
        through_eos: complete.tokens.len(),
        leading_space_ids: phase,
        opener_occurrences: user_ids.iter().filter(|id| *id == opener).count(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("tokenizer path required")?;
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(path)?).ok_or("invalid tokenizer")?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    for line in io::stdin().lock().lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let context = row["context"].as_str().ok_or("paragraph missing")?;
        let short = check(&tokenizer, &encoder, context, &row["short"]);
        let long = check(&tokenizer, &encoder, context, &row["long"]);
        let mut receipt = json!({
            "article": row["article"], "paragraph": row["paragraph"],
            "short_id": row["short"]["id"], "long_id": row["long"]["id"],
            "sealed_short": row["short"]["sealed"], "sealed_long": row["long"]["sealed"],
        });
        match (short, long) {
            (Ok(short), Ok(long)) => {
                receipt["same_original_start_and_id_prefix"] = json!(
                    short.start == long.start
                        && short.ids.len() < long.ids.len()
                        && long.ids.starts_with(&short.ids)
                );
                receipt["fits_192"] =
                    json!(short.through_eos <= WINDOW && long.through_eos <= WINDOW);
                receipt["fits_180_prefix"] =
                    json!(short.prefix <= PREFIX_LIMIT && long.prefix <= PREFIX_LIMIT);
                receipt["fits_8_source_ids"] =
                    json!(short.ids.len() <= SOURCE_CAP && long.ids.len() <= SOURCE_CAP);
                receipt["unique_source_opener"] =
                    json!(short.opener_occurrences == 1 && long.opener_occurrences == 1);
                receipt["short"] = short.receipt();
                receipt["long"] = long.receipt();
            }
            (short, long) => {
                receipt["short_error"] = json!(short.err());
                receipt["long_error"] = json!(long.err());
            }
        }
        println!("{receipt}");
    }
    Ok(())
}
