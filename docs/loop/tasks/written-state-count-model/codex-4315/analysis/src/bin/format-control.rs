//! Render an identically worded one-paragraph transfer panel from the sealed held text.
#![forbid(unsafe_code)]
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;
type R<T> = Result<T, Box<dyn Error>>;

fn read(path: &Path) -> R<Vec<Value>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn main() -> R<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [token_path, panel, out] = args.as_slice() else {
        return Err("usage: format-control TOKENIZER HELD_DIR NEW_OUTPUT".into());
    };
    let bytes = fs::read(token_path)?;
    if format!("{:x}", Sha256::digest(&bytes))
        != "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377"
    {
        return Err("tokenizer binding differs".into());
    }
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&bytes).ok_or("invalid canonical tokenizer")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let encoder = protocol.bind(&tokenizer)?;
    let src = Path::new(panel);
    let inputs = read(&src.join("rows.jsonl"))?;
    let keys = read(&src.join("keys.jsonl"))?;
    if keys.len() != inputs.len() || inputs.len() != 800 {
        return Err("expected exactly 800 crossed held rows".into());
    }
    let chosen: BTreeSet<_> = inputs
        .iter()
        .filter_map(|r| r["group_id"].as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .take(8)
        .map(str::to_owned)
        .collect();
    let dest = Path::new(out);
    fs::create_dir(dest)?;
    let mut records = BufWriter::new(File::create_new(dest.join("rows.jsonl"))?);
    let mut texts = BufWriter::new(File::create_new(dest.join("keys.jsonl"))?);
    let mut saved = 0;
    for (mut row, mut key) in inputs.into_iter().zip(keys) {
        if !chosen.contains(row["group_id"].as_str().ok_or("bad group")?) {
            continue;
        }
        if ["group_id", "pair_id", "assignment", "side"]
            .iter()
            .any(|f| row[*f] != key[*f])
        {
            return Err("key identities differ".into());
        }
        let original = key["user"].as_str().ok_or("bad user")?;
        let facts = key["context"].as_str().ok_or("bad context")?;
        let query = key["question"].as_str().ok_or("bad question")?;
        if original != format!("{facts}\n\n{query}") {
            return Err("original separator differs".into());
        }
        let render = |user: &str| {
            encoder
                .encode_assistant_prefix(&[Message {
                    role: "user",
                    content: user,
                }])
                .tokens
        };
        let before = render(original);
        if serde_json::to_value(&before)? != row["prefix"] {
            return Err("original row mismatches ordinary renderer".into());
        }
        let user = format!("{facts} {query}");
        let updated = render(&user);
        let start = row["context_start"]
            .as_u64()
            .ok_or("missing context start")? as usize;
        let end = row["context_end"].as_u64().ok_or("missing context end")? as usize;
        if end > before.len()
            || end > updated.len()
            || before[..end] != updated[..end]
            || start >= end
            || before == updated
            || updated.len() > 192
        {
            return Err("rendering changed the factual evidence".into());
        }
        for (pos_field, id_field) in [("gold_pos", "target"), ("rival_pos", "rival")] {
            let pos = row[pos_field].as_u64().ok_or("missing address")? as usize;
            if !(start..end).contains(&pos)
                || Some(u64::from(updated[pos])) != row[id_field].as_u64()
            {
                return Err("source moved".into());
            }
        }
        if updated.windows(2).any(|pair| pair == [201, 201]) {
            return Err("residual blank line".into());
        }
        row["prefix"] = json!(updated);
        row["cohort"] = json!("single_paragraph");
        let id = format!("{}-space", row["pair_id"].as_str().ok_or("bad pair")?);
        row["pair_id"] = json!(id);
        key["pair_id"] = json!(id);
        key["user"] = json!(user);
        serde_json::to_writer(&mut records, &row)?;
        writeln!(records)?;
        serde_json::to_writer(&mut texts, &key)?;
        writeln!(texts)?;
        saved += 1;
    }
    if saved != 32 {
        return Err("eight complete groups required".into());
    }
    records.flush()?;
    texts.flush()?;
    let receipt = json!({"schema":"uor-r4.occurrence-format-control/1","rows":saved,"selection":"lexically first eight unopened held group IDs","group_ids":chosen,"changed":"blank line to one ASCII space in raw ordinary user text","all_original_factual_prefixes_and_source_positions_identical":true,"digests":{"rows.jsonl":format!("{:x}",Sha256::digest(fs::read(dest.join("rows.jsonl"))?)),"keys.jsonl":format!("{:x}",Sha256::digest(fs::read(dest.join("keys.jsonl"))?))}});
    fs::write(
        dest.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
