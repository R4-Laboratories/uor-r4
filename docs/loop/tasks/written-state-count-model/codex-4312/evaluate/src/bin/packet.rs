//! Private run custody and blinded delivery. The grading process takes only anonymous replies.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::ByteBpeTokenizer;

type R<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    id: String,
    arm: String,
    tokens: Vec<u32>,
}

#[derive(Deserialize)]
struct Prompt {
    id: String,
    cap: usize,
    domain: String,
}

fn lines<T: serde::de::DeserializeOwned>(path: &str) -> R<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|v| Ok(serde_json::from_str(&v?)?))
        .collect()
}

fn create(path: impl AsRef<Path>) -> R<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().create_new(true).write(true).open(path)?,
    ))
}

fn save(file: &mut BufWriter<File>, data: &impl Serialize) -> R<()> {
    serde_json::to_writer(&mut *file, data)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn hash(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    for part in parts {
        h.update(part.as_bytes());
        h.update([0]);
    }
    format!("{:x}", h.finalize())
}

fn valid_hex(id: &str) -> bool {
    id.len() == 24
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn seal(prompts: &str, a: &str, b: &str, output: &str) -> R<()> {
    let mut input = lines::<Reply>(a)?;
    input.extend(lines::<Reply>(b)?);
    let p: Vec<Prompt> = lines(prompts)?;
    let caps: BTreeMap<_, _> = p.iter().map(|p| (p.id.as_str(), p.cap)).collect();
    let mut ids = BTreeSet::new();
    let mut arms = BTreeSet::new();
    for row in &input {
        let cap = *caps.get(row.id.as_str()).ok_or("unexpected reply ID")?;
        if !valid_hex(&row.id)
            || !valid_hex(&row.arm)
            || !ids.insert((&row.id, &row.arm))
            || row.tokens.is_empty()
            || row.tokens.len() > cap
            || row.tokens.iter().any(|&v| v >= 4096)
            || row
                .tokens
                .iter()
                .position(|&v| v == 1)
                .map_or(row.tokens.len() != cap, |i| i + 1 != row.tokens.len())
        {
            return Err(
                "duplicate, invalid, unaccounted short reply, or continuation after EOS".into(),
            );
        }
        arms.insert(&row.arm);
    }
    if arms.len() != 2
        || caps.len() != p.len()
        || input.len() != p.len() * 2
        || p.iter().filter(|p| p.domain == "conversation").count() != 200
    {
        return Err("incomplete paired panel".into());
    }
    input.sort_by_cached_key(|r| hash(&["anonymous-grading-order/1", &r.id, &r.arm]));
    let mut file = create(output)?;
    for row in input {
        save(&mut file, &row)?;
    }
    file.flush()?;
    Ok(())
}

fn human(
    key: &str,
    vocab: &str,
    prompts: &str,
    source: &str,
    blind: &str,
    output: &str,
    private: &str,
) -> R<()> {
    let key: Value = serde_json::from_slice(&fs::read(key)?)?;
    let ordering = key["ordering"].as_str().ok_or("missing layout salt")?;
    let a = key["reasoning-first"].as_str().ok_or("missing arm")?;
    let b = key["answer-first"].as_str().ok_or("missing arm")?;
    let vocab = fs::read(vocab)?;
    if format!("{:x}", Sha256::digest(&vocab))
        != "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377"
    {
        return Err("incorrect vocabulary".into());
    }
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&vocab).ok_or("invalid tokenizer")?;
    let prompts: BTreeMap<_, _> = lines::<Prompt>(prompts)?
        .into_iter()
        .filter(|p| p.domain == "open")
        .map(|p| (p.id, p.cap))
        .collect();
    let mut sources = lines::<Value>(source)?;
    if sources.len() != 32 || prompts.len() != 32 {
        return Err("expected 32 open chats".into());
    }
    let replies: BTreeMap<_, _> = lines::<Reply>(blind)?
        .into_iter()
        .filter(|r| prompts.contains_key(&r.id))
        .map(|r| ((r.id, r.arm), r.tokens))
        .collect();
    if replies.len() != 64 {
        return Err("incomplete open reply pairs".into());
    }
    let mut seen = BTreeSet::new();
    let mut display = Vec::new();
    let mut decode = Vec::new();
    sources.sort_by_cached_key(|s| {
        hash(&["human-position/1", ordering, s["id"].as_str().unwrap_or("")])
    });
    for (i, src) in sources.iter().enumerate() {
        let id = src["id"].as_str().ok_or("missing source id")?;
        if !seen.insert(id) {
            return Err("duplicate open chat".into());
        }
        let cap = *prompts.get(id).ok_or("unknown open ID")?;
        let trial = hash(&["human-item/1", ordering, id])[..16].to_owned();
        let layout = if i % 2 == 0 { [a, b] } else { [b, a] };
        let mut choices = Vec::new();
        for arm in layout {
            let ids = replies
                .get(&(id.to_owned(), arm.to_owned()))
                .ok_or("missing open reply")?;
            let (content, end) = if ids.last() == Some(&1) {
                (&ids[..ids.len() - 1], "EOS")
            } else if ids.len() == cap {
                (&ids[..], "token limit")
            } else {
                return Err("unexplained termination".into());
            };
            let bos_offsets: Vec<_> = content
                .iter()
                .enumerate()
                .filter_map(|(i, &t)| (t == 0).then_some(i))
                .collect();
            choices.push(json!({"text":tokenizer.decode(content),"end":end,"token_count":ids.len(),"bos_offsets":bos_offsets}));
        }
        let messages = src["messages"].as_array().ok_or("missing open history")?;
        let mut history = Vec::new();
        for m in messages {
            let role = m["role"].as_str().ok_or("missing role")?;
            if !matches!(role, "assistant" | "user") {
                return Err("invalid open role".into());
            }
            history.push(
                json!({"role":role,"content":m["content"].as_str().ok_or("missing content")?}),
            );
        }
        if !history.iter().any(|v| v["role"] == "assistant")
            || history.last().map(|v| &v["role"]) != Some(&json!("user"))
        {
            return Err("open prompt requires human history and a latest request".into());
        }
        display.push(json!({"trial":trial,"history":history,"replies":choices}));
        decode.push(json!({"trial":trial,"id":id,"left":layout[0],"right":layout[1]}));
    }
    display
        .sort_by_cached_key(|v| hash(&["human-display/1", ordering, v["trial"].as_str().unwrap()]));
    let mut mapping = create(private)?;
    for row in decode {
        save(&mut mapping, &row)?;
    }
    mapping.flush()?;
    let data = serde_json::to_string(&display)?
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    let mut page = create(output)?;
    page.write_all(
        include_str!("packet.html")
            .replace("{{ROWS}}", &data)
            .as_bytes(),
    )?;
    page.flush()?;
    Ok(())
}

fn main() -> R<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [op, prompts, a, b, out] if op == "seal" => seal(prompts,a,b,out),
        [op, key, vocab, prompts, source, blind, html, private] if op == "human" =>
            human(key,vocab,prompts,source,blind,html,private),
        _ => Err("expected seal PROMPTS A B NEW_FILE or human KEY VOCAB PROMPTS OPEN_SOURCES SEALED NEW_HTML NEW_PRIVATE_MAP".into())
    }
}
