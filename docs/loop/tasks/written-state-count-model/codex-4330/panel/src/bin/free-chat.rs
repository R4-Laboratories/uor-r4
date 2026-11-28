#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use uor_r4_tokenizer::dialogue::{DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const PANEL: &str = "20261127-codex-visible-answer/panel";
const OASST: &str = "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b";
const PREVIOUS: &str = "8fc9be4086f86b35187b019fb1e22581574168ea9fb6326627fb45658d99f035";
const BANK: &str = "d95090332157731b3d02bace4bd3e3467a3a6ee80e5934a6fed3cf89cb3e1541";
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";

#[derive(Deserialize)]
struct Tree {
    message_tree_id: String,
    tree_state: String,
    prompt: Prompt,
}

#[derive(Deserialize)]
struct Prompt {
    #[serde(flatten)]
    message: Node,
    text: String,
    message_id: String,
    replies: Vec<Node>,
}

#[derive(Deserialize)]
struct Node {
    role: String,
    lang: String,
    deleted: bool,
    synthetic: bool,
    review_result: Option<bool>,
}

#[derive(Deserialize)]
struct Prior {
    source_tree_id: String,
}

impl Node {
    fn human(&self, role: &str) -> bool {
        self.role == role
            && self.lang == "en"
            && self.review_result == Some(true)
            && !self.deleted
            && !self.synthetic
    }
}

fn checked(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if hex::encode(Sha256::digest(&bytes)) != expected {
        return Err(format!("input digest changed: {}", path.display()).into());
    }
    Ok(bytes)
}

fn normalized(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn census() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [runs, output] = args.as_slice() else {
        return Err("usage: free-chat WICK_RUNS NEW_OUTPUT".into());
    };
    let output = Path::new(output);
    let panel = Path::new(runs).join(PANEL);
    let corpus = panel.join("inputs/oasst2-ready.trees.jsonl.gz");
    let checked_source = checked(&corpus, OASST)?;
    drop(checked_source);
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked(
        &panel.join("inputs/tokenizer.json"),
        TOKENIZER,
    )?)
    .ok_or("invalid tokenizer")?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let prior_bytes = checked(&panel.join("frozen-v1/open_sources.jsonl"), PREVIOUS)?;
    let prior: BTreeSet<_> = BufReader::new(prior_bytes.as_slice())
        .lines()
        .map(|line| -> Result<_> { Ok(serde_json::from_str::<Prior>(&line?)?.source_tree_id) })
        .collect::<Result<_>>()?;
    let bank_bytes = checked(&panel.join("inputs/d87-bank-prompt-digests.txt"), BANK)?;
    let bank: BTreeSet<_> = std::str::from_utf8(&bank_bytes)?.lines().collect();
    let mut gzip = Command::new("gzip")
        .arg("-dc")
        .arg(&corpus)
        .stdout(Stdio::piped())
        .spawn()?;
    let mut candidates: BTreeMap<_, Vec<_>> = BTreeMap::new();
    let mut trees = 0;
    let mut admitted = 0;
    for line in BufReader::new(gzip.stdout.take().ok_or("gzip lost output")?).lines() {
        let tree: Tree = serde_json::from_str(&line?)?;
        trees += 1;
        if tree.tree_state != "ready_for_export"
            || prior.contains(&tree.message_tree_id)
            || !tree.prompt.message.human("prompter")
            || !tree
                .prompt
                .replies
                .iter()
                .any(|node| node.human("assistant"))
            || tree.prompt.text.contains(['@', '\0', '<', '>'])
            || tree.prompt.text.contains("http")
        {
            continue;
        }
        let words = normalized(&tree.prompt.text);
        let count = words.split_whitespace().count();
        if !(10..=48).contains(&count)
            || bank.contains(hex::encode(&Sha256::digest(&words)[..8]).as_str())
        {
            continue;
        }
        let kind = match words.split_whitespace().next() {
            Some("explain") => "explain",
            Some("write") => "write",
            Some("how" | "why") => "how_why",
            Some("can" | "could") => "request",
            _ => continue,
        };
        let render = encoder.encode_assistant_prefix(&[Message {
            role: "user",
            content: &tree.prompt.text,
        }]);
        if render.emitted_turns != 1
            || render.special_token_occurrences != 0
            || render.tokens.len() > 176
        {
            continue;
        }
        let rank = Sha256::digest(format!("codex-4330/free-chat/1/{}", tree.message_tree_id));
        candidates
            .entry(kind)
            .or_default()
            .push((rank, words, tree, render.tokens));
        admitted += 1;
    }
    if !gzip.wait()?.success() {
        return Err("gzip failed".into());
    }
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for (kind, pool) in &mut candidates {
        pool.sort_unstable_by_key(|candidate| candidate.0);
        rows.extend(
            pool.iter()
                .filter(|(_, text, _, _)| seen.insert(text.as_str()))
                .take(2)
                .map(|(rank, _, tree, prefix)| (kind, rank, tree, prefix)),
        );
    }
    if rows.len() != 8 {
        return Err("expected two fresh roots in each of four request groups".into());
    }
    fs::create_dir(output)?;
    let mut prompts = BufWriter::new(File::create(output.join("prompts.jsonl"))?);
    let mut sources = BufWriter::new(File::create(output.join("sources.jsonl"))?);
    for (kind, rank, tree, prefix) in rows {
        let case_id = format!("free-{}", hex::encode(&rank[..8]));
        serde_json::to_writer(
            &mut prompts,
            &json!({"case_id":case_id,"prompt":tree.prompt.text}),
        )?;
        writeln!(prompts)?;
        serde_json::to_writer(
            &mut sources,
            &json!({"case_id":case_id,"tree_id":tree.message_tree_id,
            "message_id":tree.prompt.message_id,"kind":kind,"prompt_ids":prefix}),
        )?;
        writeln!(sources)?;
    }
    prompts.flush()?;
    sources.flush()?;
    let receipt = json!({"schema":"uor-r4.real-chat-roots/1","scanned_trees":trees,"eligible_roots":admitted,
        "selected":8,"source":"OpenAssistant/oasst2","source_revision":"179dd21fc55192153d94adb0e0ce8f69e222bf75",
        "corpus_sha256":OASST,"blocked_prior_open_panel_sha256":PREVIOUS,"prompt_bank_sha256":BANK,"tokenizer_sha256":TOKENIZER});
    fs::write(
        output.join("receipt.json"),
        format!("{}\n", serde_json::to_string_pretty(&receipt)?),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    census()
}
