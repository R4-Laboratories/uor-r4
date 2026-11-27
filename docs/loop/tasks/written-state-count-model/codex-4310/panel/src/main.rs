#![forbid(unsafe_code)]

mod grade;

use flate2::read::MultiGzDecoder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const CHAT_SHA: &str = "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b";
const PROSE_SHA: &str = "79ee06627f4ba2fec056fcd23c077a29238bbcbacc3187e8d1a0b47e663e9c7a";
const D87_BANK_DIGESTS_SHA: &str =
    "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541";
const D87_BANK_SOURCE: &str =
    include_str!("../../../../../../labs/m2-donor-corpus/instrument/src/bank.rs");
const SEED: &str = "codex-4310-natural-v1";
const CHAT_MIN_GOLD: usize = 8;
const CHAT_SHORT_CAP: usize = 48;
const CHAT_CAP: usize = 128;
const CHAT_MAX_PREFIX: usize = 1536;
const WIKI_FIRST_ROW: usize = 16_384;
const WIKI_END_ROW: usize = 32_768;
const WIKI_PREFIX: usize = 128;
const WIKI_CAP: usize = 64;
const BOS: u16 = 0;
const EOS: u16 = 1;

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Split {
    Fit,
    Held,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Domain {
    Conversation,
    Prose,
}

#[derive(Serialize, Deserialize)]
struct PanelRow {
    id: String,
    split: Split,
    domain: Domain,
    prefix: Vec<u16>,
    gold: Vec<u16>,
    new_gold_tokens: usize,
    new_teacher_tokens: Vec<bool>,
    free_run: bool,
    donor_opening_exposed: bool,
    user_turns: usize,
    record_position: usize,
    source_message_id: Option<String>,
    cap: usize,
}

#[derive(Deserialize)]
struct Tree {
    message_tree_id: String,
    tree_state: String,
    prompt: Node,
}

#[derive(Deserialize)]
struct Node {
    message_id: String,
    text: String,
    role: String,
    lang: String,
    review_result: Option<bool>,
    deleted: bool,
    rank: Option<i64>,
    synthetic: bool,
    #[serde(default)]
    replies: Vec<Node>,
}

#[derive(Default, Serialize)]
struct Census {
    inspected: usize,
    previous_held: usize,
    donor_prompt_hit: usize,
    d87_dev_bucket: usize,
    no_eligible_continuation: usize,
    duplicate_group: usize,
    fit_eligible: usize,
    held_eligible: usize,
    fit_opening_exposed: usize,
    held_opening_exposed: usize,
}

type Buckets = BTreeMap<(Domain, Split, bool), BTreeMap<(bool, [u8; 32]), PanelRow>>;

fn file_sha(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut digest)?;
    Ok(hex::encode(digest.finalize()))
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    let actual = file_sha(path)?;
    if actual != expected {
        return Err(format!("SHA mismatch for {}: {actual}", path.display()).into());
    }
    Ok(())
}

// Same normalization as the D87 prompt-bank original-ID prefix.
fn normalize_prompt(text: &str) -> String {
    let mapped: String = text
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect();
    mapped.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn prompt_key(text: &str) -> [u8; 32] {
    Sha256::digest(normalize_prompt(text)).into()
}

fn d87_dev_bucket(text: &str) -> bool {
    // Reproduce the published split-v1 prompt grouping without opening either
    // DEV split. Modulus 200 reserves test buckets 0–1 and tune bucket 2.
    let mut digest = Sha256::new();
    digest.update(b"claude-411/d87-split/v1\0");
    digest.update(normalize_prompt(text).as_bytes());
    digest_head(digest.finalize().into()) % 200 < 3
}

fn donor_follow_ups() -> Result<HashSet<[u8; 32]>> {
    let list = D87_BANK_SOURCE
        .split_once("pub const FOLLOW_UPS: [&str; 40] = [\n")
        .and_then(|(_, rest)| rest.split_once("\n];"))
        .ok_or("D87 bank source changed the follow-up list format")?;
    let keys: Option<HashSet<_>> = list
        .0
        .lines()
        .map(|line| {
            line.trim()
                .strip_prefix('"')?
                .strip_suffix("\",")
                .map(prompt_key)
        })
        .collect();
    keys.filter(|keys| keys.len() == 40)
        .ok_or_else(|| "D87 follow-up templates could not be recovered exactly".into())
}

fn read_bank_keys(path: &Path) -> Result<HashSet<String>> {
    let mut keys = HashSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let key = line?;
        if key.len() != 16 || !key.bytes().all(|ch| ch.is_ascii_hexdigit()) {
            return Err("bad D87 bank digest".into());
        }
        keys.insert(key);
    }
    if keys.is_empty() {
        return Err("empty D87 bank digest list".into());
    }
    Ok(keys)
}

fn digest_head(digest: [u8; 32]) -> u64 {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(bytes)
}

fn hash_rank(label: &str, key: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(SEED);
    digest.update([0]);
    digest.update(label);
    digest.update([0]);
    digest.update(key);
    digest.finalize().into()
}

fn split_for(key: &[u8]) -> Split {
    if hash_rank("split", key)[0] < 96 {
        Split::Held
    } else {
        Split::Fit
    }
}

fn new_gold_tokens(prefix: &[u16], gold: &[u16]) -> usize {
    let observed: HashSet<_> = prefix.iter().copied().collect();
    gold.iter()
        .filter(|&&token| token != EOS && !observed.contains(&token))
        .count()
}

fn new_teacher_tokens(prefix: &[u16], gold: &[u16]) -> Vec<bool> {
    let mut observed: HashSet<_> = prefix.iter().copied().collect();
    gold.iter()
        .map(|&token| token != EOS && observed.insert(token))
        .collect()
}

fn quota(domain: Domain, split: Split, short: bool) -> (usize, usize, usize) {
    // (max rows, min rows, free-run rows). The cap is independent of model output.
    match (domain, split, short) {
        (Domain::Conversation, Split::Fit, true) => (16, 16, 0),
        (Domain::Conversation, Split::Fit, false) => (48, 48, 0),
        (Domain::Conversation, Split::Held, true) => (32, 16, 8),
        (Domain::Conversation, Split::Held, false) => (96, 48, 16),
        (Domain::Prose, Split::Fit, _) => (64, 64, 0),
        (Domain::Prose, Split::Held, _) => (128, 64, 24),
    }
}

fn short_reply(row: &PanelRow) -> bool {
    row.domain == Domain::Conversation && row.gold.len() <= CHAT_SHORT_CAP
}

fn add_row(buckets: &mut Buckets, census: &mut Census, row: PanelRow) {
    match row.split {
        Split::Fit => {
            census.fit_eligible += 1;
            census.fit_opening_exposed += usize::from(row.donor_opening_exposed);
        }
        Split::Held => {
            census.held_eligible += 1;
            census.held_opening_exposed += usize::from(row.donor_opening_exposed);
        }
    }
    let rank = hash_rank("selection", row.id.as_bytes());
    let short = short_reply(&row);
    let (max_rows, _, _) = quota(row.domain, row.split, short);
    let bucket = buckets.entry((row.domain, row.split, short)).or_default();
    bucket.insert((row.donor_opening_exposed, rank), row);
    if bucket.len() > max_rows {
        bucket.pop_last();
    }
}

fn usable(node: &Node, role: &str) -> bool {
    node.role == role
        && node.lang == "en"
        && node.review_result == Some(true)
        && !node.deleted
        && !node.synthetic
        && !node.text.trim().is_empty()
        && !node.text.contains("<|")
}

fn bank_hit(bank: &HashSet<String>, text: &str) -> bool {
    bank.contains(&hex::encode(&prompt_key(text)[..8]))
}

fn token_ids(input: &[u32]) -> Result<Vec<u16>> {
    Ok(input
        .iter()
        .copied()
        .map(u16::try_from)
        .collect::<std::result::Result<_, _>>()?)
}

fn chat_row(
    tree: &Tree,
    path: &[&Node],
    encoder: &DialogueEncoder<'_>,
    key: &[u8],
    opening_exposed: bool,
) -> Result<Option<PanelRow>> {
    let messages: Vec<_> = path
        .iter()
        .map(|node| Message {
            role: if node.role == "prompter" {
                "user"
            } else {
                "assistant"
            },
            content: &node.text,
        })
        .collect();
    // A caller presents a complete real branch ending at an assistant response.
    let answer_index = messages.len() - 1;
    let prefix = encoder.encode_assistant_prefix(&messages[..answer_index]);
    let full = encoder.encode_document(&messages);
    if prefix.special_token_occurrences != 0 || full.special_token_occurrences != 0 {
        return Ok(None);
    }
    let user_turns = answer_index.div_ceil(2);
    let earlier_assistant_eos = prefix
        .tokens
        .iter()
        .zip(&prefix.response_mask)
        .filter(|&(&id, &mask)| id == u32::from(EOS) && mask == 1)
        .count();
    if earlier_assistant_eos != user_turns - 1
        || prefix.tokens.first() != Some(&u32::from(BOS))
        || prefix.emitted_turns != answer_index
    {
        return Err(format!(
            "missing original dialogue history in {}",
            tree.message_tree_id
        )
        .into());
    }
    let Some(target) = full.tokens.strip_prefix(prefix.tokens.as_slice()) else {
        return Err(format!("dialogue prefix mismatch in {}", tree.message_tree_id).into());
    };
    let target_mask = &full.response_mask[prefix.tokens.len()..];
    if target_mask.iter().any(|&mask| mask != 1) || target.last() != Some(&u32::from(EOS)) {
        return Err(format!(
            "assistant target boundary mismatch in {}",
            tree.message_tree_id
        )
        .into());
    }
    if prefix.tokens.len() > CHAT_MAX_PREFIX || !(CHAT_MIN_GOLD..=CHAT_CAP).contains(&target.len())
    {
        return Ok(None);
    }
    let prefix = token_ids(&prefix.tokens)?;
    let gold = token_ids(target)?;
    let novel = new_gold_tokens(&prefix, &gold);
    if novel == 0 {
        return Ok(None);
    }
    Ok(Some(PanelRow {
        id: format!("oasst2:{}", tree.message_tree_id),
        split: split_for(key),
        domain: Domain::Conversation,
        new_teacher_tokens: new_teacher_tokens(&prefix, &gold),
        free_run: false,
        donor_opening_exposed: opening_exposed,
        prefix,
        gold,
        new_gold_tokens: novel,
        user_turns,
        record_position: answer_index,
        source_message_id: Some(path[answer_index].message_id.clone()),
        cap: CHAT_CAP,
    }))
}

fn eligible_thread<'a>(
    tree: &Tree,
    path: &mut Vec<&'a Node>,
    current: &'a Node,
    find_turn: &impl Fn(&Tree, &[&Node]) -> Result<Option<PanelRow>>,
    bank: &HashSet<String>,
    donor_follow_ups: &HashSet<[u8; 32]>,
) -> Result<Option<PanelRow>> {
    if path.len() >= 4 && current.role == "assistant" {
        if let Some(row) = find_turn(tree, path)? {
            return Ok(Some(row));
        }
    }
    let next_role = if current.role == "assistant" {
        "prompter"
    } else {
        "assistant"
    };
    let mut children: Vec<_> = current
        .replies
        .iter()
        .filter(|node| {
            usable(node, next_role)
                && (next_role == "assistant"
                    || (!bank_hit(bank, &node.text)
                        && !donor_follow_ups.contains(&prompt_key(&node.text))))
        })
        .collect();
    children.sort_by(|a, b| {
        a.rank
            .is_none()
            .cmp(&b.rank.is_none())
            .then_with(|| a.rank.cmp(&b.rank))
            .then_with(|| a.message_id.cmp(&b.message_id))
    });
    for node in children {
        path.push(node);
        if let Some(row) = eligible_thread(tree, path, node, find_turn, bank, donor_follow_ups)? {
            return Ok(Some(row));
        }
        path.pop();
    }
    Ok(None)
}

fn chat(
    input: &Path,
    encoder: &DialogueEncoder<'_>,
    bank: &HashSet<String>,
    buckets: &mut Buckets,
) -> Result<Census> {
    let mut census = Census::default();
    let mut seen = HashSet::new();
    let donor_follow_ups = donor_follow_ups()?;
    let file = BufReader::new(MultiGzDecoder::new(File::open(input)?));
    for line in file.lines() {
        let tree: Tree = serde_json::from_str(&line?)?;
        census.inspected += 1;
        // claude-293's historical heldout split used this exact tree-ID rule.
        if digest_head(Sha256::digest(tree.message_tree_id.as_bytes()).into()) % 1000 < 50 {
            census.previous_held += 1;
            continue;
        }
        if tree.tree_state != "ready_for_export" || !usable(&tree.prompt, "prompter") {
            census.no_eligible_continuation += 1;
            continue;
        }
        let opening_exposed = bank_hit(bank, &tree.prompt.text);
        census.donor_prompt_hit += usize::from(opening_exposed);
        if opening_exposed && d87_dev_bucket(&tree.prompt.text) {
            census.d87_dev_bucket += 1;
            continue;
        }
        let key = prompt_key(&tree.prompt.text);
        let mut grouped_key = b"oasst2:".to_vec();
        grouped_key.extend_from_slice(&key);
        let find_turn = |tree: &Tree, path: &[&Node]| {
            chat_row(tree, path, encoder, &grouped_key, opening_exposed)
        };
        match eligible_thread(
            &tree,
            &mut vec![&tree.prompt],
            &tree.prompt,
            &find_turn,
            bank,
            &donor_follow_ups,
        )? {
            Some(row) if seen.insert(key) => add_row(buckets, &mut census, row),
            Some(_) => census.duplicate_group += 1,
            None => census.no_eligible_continuation += 1,
        }
    }
    Ok(census)
}

fn column<'a>(row: &'a Row, key: &str) -> Result<&'a str> {
    match row.get_column_iter().find(|(name, _)| name.as_str() == key) {
        Some((_, Field::Str(value))) => Ok(value),
        _ => Err(format!("missing string column {key}").into()),
    }
}

fn prose(input: &Path, tokenizer: &ByteBpeTokenizer, buckets: &mut Buckets) -> Result<Census> {
    let mut census = Census::default();
    let mut seen = HashSet::new();
    let file = SerializedFileReader::new(File::open(input)?)?;
    let rows = file.get_row_iter(None)?.enumerate();
    for (position, row) in rows.take(WIKI_END_ROW).skip(WIKI_FIRST_ROW) {
        let row = row?;
        census.inspected += 1;
        let article_id = column(&row, "id")?;
        let title = column(&row, "title")?.trim();
        let text = column(&row, "text")?.trim();
        // The bounded opening window is real adjacent prose with an unchanged
        // token boundary at the held continuation. It has no dialogue marker.
        let snippet: String = text.chars().take(4096).collect();
        let document = format!(" {title}\n\n{snippet}");
        let content = tokenizer.encode(&document);
        if content.len() < WIKI_PREFIX + WIKI_CAP
            || content[..WIKI_PREFIX + WIKI_CAP]
                .iter()
                .any(|&token| token < 3)
            || !title.chars().any(char::is_alphabetic)
        {
            census.no_eligible_continuation += 1;
            continue;
        }
        let mut prefix = vec![BOS];
        prefix.extend(token_ids(&content[..WIKI_PREFIX])?);
        let gold = token_ids(&content[WIKI_PREFIX..WIKI_PREFIX + WIKI_CAP])?;
        let novel = new_gold_tokens(&prefix, &gold);
        if novel == 0 {
            census.no_eligible_continuation += 1;
            continue;
        }
        if !seen.insert(article_id.to_owned()) {
            census.duplicate_group += 1;
            continue;
        }
        let id = format!("wikipedia-en:{article_id}");
        let row = PanelRow {
            split: split_for(id.as_bytes()),
            id,
            domain: Domain::Prose,
            new_teacher_tokens: new_teacher_tokens(&prefix, &gold),
            free_run: false,
            donor_opening_exposed: false,
            prefix,
            gold,
            new_gold_tokens: novel,
            user_turns: 0,
            record_position: position,
            source_message_id: None,
            cap: WIKI_CAP,
        };
        add_row(buckets, &mut census, row);
    }
    Ok(census)
}

fn write_panel(
    out: &Path,
    mut buckets: Buckets,
    chat: Census,
    prose: Census,
    bank_path: &Path,
) -> Result<()> {
    let mut totals = BTreeMap::new();
    let mut fit = Vec::new();
    let mut held = Vec::new();
    for (domain, short) in [
        (Domain::Conversation, true),
        (Domain::Conversation, false),
        (Domain::Prose, false),
    ] {
        for split in [Split::Fit, Split::Held] {
            let (_, min_rows, free_rows) = quota(domain, split, short);
            let mut rows: Vec<_> = buckets
                .remove(&(domain, split, short))
                .into_iter()
                .flat_map(BTreeMap::into_values)
                .collect();
            let band = match (domain, short) {
                (Domain::Conversation, true) => "short_chat",
                (Domain::Conversation, false) => "long_chat",
                (Domain::Prose, _) => "prose",
            };
            let label = format!(
                "{band}_{}",
                serde_json::to_value(split)?.as_str().ok_or("split")?
            );
            if rows.len() < min_rows {
                return Err(format!(
                    "{label} underfilled at {} of required {min_rows} rows",
                    rows.len()
                )
                .into());
            }
            for row in rows.iter_mut().take(free_rows) {
                row.free_run = true;
            }
            totals.insert(label, json!({
                "rows": rows.len(),
                "free_run_rows": free_rows,
                "donor_opening_exposed_rows": rows.iter().filter(|row| row.donor_opening_exposed).count(),
                "free_run_donor_opening_exposed_rows": rows.iter().filter(|row| row.free_run && row.donor_opening_exposed).count(),
                "prompt_tokens": rows.iter().map(|row| row.prefix.len()).sum::<usize>(),
                "gold_tokens": rows.iter().map(|row| row.gold.len()).sum::<usize>(),
                "new_gold_tokens": rows.iter().map(|row| row.new_gold_tokens).sum::<usize>(),
                "new_teacher_tokens": rows.iter().flat_map(|row| &row.new_teacher_tokens).filter(|&&new| new).count(),
            }));
            match split {
                Split::Fit => fit.extend(rows),
                Split::Held => held.extend(rows),
            }
        }
    }
    fs::create_dir(out)?;
    for (label, rows) in [("fit", fit), ("held", held)] {
        let path = out.join(format!("{label}.jsonl"));
        let mut writer = BufWriter::new(File::create(path)?);
        for row in rows {
            serde_json::to_writer(&mut writer, &row)?;
            writeln!(writer)?;
        }
        writer.flush()?;
    }
    let receipt = json!({
        "schema": "uor-r4.s4-source-panel/1",
        "seed": SEED,
        "tokenizer_sha256": TOKENIZER_SHA,
        "oasst2_sha256": CHAT_SHA,
        "wiki_00020_sha256": PROSE_SHA,
        "d87_bank_digests_sha256": file_sha(bank_path)?,
        "fit_sha256": file_sha(&out.join("fit.jsonl"))?,
        "held_sha256": file_sha(&out.join("held.jsonl"))?,
        "chat_census": chat,
        "prose_census": prose,
        "cells": totals,
    });
    let mut writer = BufWriter::new(File::create(out.join("receipt.json"))?);
    serde_json::to_writer_pretty(&mut writer, &receipt)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if let [verb, tokenizer, held, generations, out] = args.as_slice() {
        if verb == "grade" {
            return grade::run(
                Path::new(tokenizer),
                Path::new(held),
                Path::new(generations),
                Path::new(out),
            );
        }
    }
    let [tok_path, chat_path, prose_path, bank_path, out] = args.as_slice() else {
        return Err("usage: s4-source-panel TOKENIZER OASST2_GZ WIKI_00020_PARQUET BANK_DIGESTS NEW_OUTPUT_DIR".into());
    };
    for (path, pin) in [
        (tok_path, TOKENIZER_SHA),
        (chat_path, CHAT_SHA),
        (prose_path, PROSE_SHA),
        (bank_path, D87_BANK_DIGESTS_SHA),
    ] {
        verify(Path::new(path), pin)?;
    }
    let token_bytes = fs::read(tok_path)?;
    let tokenizer =
        ByteBpeTokenizer::from_tokenizer_json_bytes(&token_bytes).ok_or("invalid tokenizer")?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    if protocol.bos_id != u32::from(BOS) || protocol.eos_id != u32::from(EOS) {
        return Err("unexpected tokenizer BOS/EOS IDs".into());
    }
    let mut buckets = BTreeMap::new();
    let bank_path = Path::new(bank_path);
    let bank = read_bank_keys(bank_path)?;
    let chat_census = chat(
        Path::new(chat_path),
        &protocol.bind(&tokenizer)?,
        &bank,
        &mut buckets,
    )?;
    let prose_census = prose(Path::new(prose_path), &tokenizer, &mut buckets)?;
    eprintln!("chat {}", serde_json::to_string(&chat_census)?);
    eprintln!("prose {}", serde_json::to_string(&prose_census)?);
    write_panel(
        Path::new(out),
        buckets,
        chat_census,
        prose_census,
        bank_path,
    )
}
