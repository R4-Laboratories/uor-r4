#![forbid(unsafe_code)]

mod audit;
mod controlled;
mod human;
mod inputs;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SEED: &str = "codex-4315-shared-occurrence-panel-v1";
const LIMIT: usize = 192;
const SPACE: u32 = 223;
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const SPLITS: [&str; 5] = ["train", "held", "human", "absent", "train-absent"];

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct Row {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    cohort: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    raw_target: u32,
    raw_rival: u32,
    lexical_offset: usize,
    leading_whitespace: Vec<u32>,
    space_token: u32,
    equal_length: bool,
    context_start: usize,
    context_end: usize,
    gold_pos: Option<usize>,
    rival_pos: Option<usize>,
    source_status: Status,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Unique,
    Ambiguous,
    Absent,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct Key {
    pair_id: String,
    group_id: String,
    assignment: usize,
    side: usize,
    context: String,
    user: String,
    question: String,
    reply: Option<String>,
    gold: Vec<u32>,
    topic: String,
    source: serde_json::Value,
}

#[derive(Clone)]
struct Case {
    context: String,
    question: String,
    user: String,
    reply: Option<String>,
    prefix: Vec<u32>,
    context_start: usize,
    context_end: usize,
    gold: Vec<u32>,
    lexical_offset: usize,
}

impl Case {
    fn target(&self) -> Option<u32> {
        self.gold.get(self.lexical_offset).copied()
    }

    fn context_positions(&self, id: u32) -> Vec<usize> {
        (self.context_start..self.context_end)
            .filter(|&pos| self.prefix[pos] == id)
            .collect()
    }

    fn unique_prefix_position(&self, id: u32) -> Option<usize> {
        let mut positions = self
            .prefix
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == id);
        let (pos, _) = positions.next()?;
        ((self.context_start..self.context_end).contains(&pos) && positions.next().is_none())
            .then_some(pos)
    }

    // Decode individual pieces as bytes. UTF-8 may cross a BPE boundary.
    fn context_token_ranges(&self, tokenizer: &ByteBpeTokenizer) -> Vec<(usize, usize, usize)> {
        let mut cursor = 0;
        (self.context_start..self.context_end)
            .map(|pos| {
                let start = cursor;
                cursor += tokenizer.decode_bytes(&[self.prefix[pos]]).len();
                (pos, start, cursor)
            })
            .collect()
    }
}

struct Builder<'a> {
    tokenizer: &'a ByteBpeTokenizer,
    encoder: DialogueEncoder<'a>,
    user_head: Vec<u32>,
}

impl<'a> Builder<'a> {
    fn new(tokenizer: &'a ByteBpeTokenizer, protocol: &DialogueProtocol) -> Result<Self> {
        let encoder = protocol.bind(tokenizer)?;
        let user_head = [vec![0], tokenizer.encode("User:")].concat();
        let suffix = [
            tokenizer.encode("\n"),
            tokenizer.encode(protocol.assistant_marker()),
        ]
        .concat();
        require(suffix == MARKER, "canonical assistant marker changed")?;
        Ok(Self {
            tokenizer,
            encoder,
            user_head,
        })
    }

    fn case(&self, context: &str, question: &str, reply: Option<&str>) -> Result<Option<Case>> {
        let user = format!("{context}\n\n{question}");
        let message = Message {
            role: "user",
            content: &user,
        };
        let open = self.encoder.encode_assistant_prefix(&[message]);
        if open.tokens.len() > LIMIT
            || open.special_token_occurrences != 0
            || open.emitted_turns != 1
        {
            return Ok(None);
        }
        require(
            open.tokens.ends_with(&MARKER),
            "canonical prompt lost its assistant marker",
        )?;
        let context_bytes = format!(" {context}");
        let context_ids = self.tokenizer.encode(&context_bytes);
        let context_start = self.user_head.len();
        let context_end = context_start + context_ids.len();
        let body_end = open.tokens.len() - MARKER.len();
        let delimiters: Vec<_> = open.tokens[context_start..body_end]
            .windows(2)
            .enumerate()
            .filter_map(|(offset, ids)| (ids == [201, 201]).then_some(context_start + offset))
            .collect();
        if delimiters != [context_end]
            || context_end + 2 >= body_end
            || !open
                .tokens
                .starts_with(&[self.user_head.clone(), context_ids.clone()].concat())
            || self.tokenizer.decode_bytes(&context_ids) != context_bytes.as_bytes()
        {
            return Ok(None);
        }
        let (gold, lexical_offset) = if let Some(reply) = reply {
            let complete = self.encoder.encode_document(&[
                Message {
                    role: "user",
                    content: &user,
                },
                Message {
                    role: "assistant",
                    content: reply,
                },
            ]);
            let tail = complete
                .tokens
                .strip_prefix(open.tokens.as_slice())
                .ok_or("canonical reply changes the open prefix")?;
            if complete.emitted_turns != 2
                || complete.special_token_occurrences != 0
                || tail.last() != Some(&1)
                || complete.response_mask[open.tokens.len()..]
                    .iter()
                    .any(|&mask| mask != 1)
            {
                return Ok(None);
            }
            let gold = tail[..tail.len() - 1].to_vec();
            if self.tokenizer.decode_bytes(&gold) != format!(" {}", reply.trim()).as_bytes() {
                return Ok(None);
            }
            let Some(offset) = gold.iter().position(|&id| {
                let bytes = self.tokenizer.decode_bytes(&[id]);
                !bytes.is_empty() && !bytes.iter().all(u8::is_ascii_whitespace)
            }) else {
                return Ok(None);
            };
            if open.tokens.len() + offset > LIMIT || gold[..offset].iter().any(|&id| id != SPACE) {
                return Ok(None);
            }
            (gold, offset)
        } else {
            (Vec::new(), 0)
        };
        Ok(Some(Case {
            context: context.to_owned(),
            question: question.to_owned(),
            user,
            reply: reply.map(str::to_owned),
            prefix: open.tokens,
            context_start,
            context_end,
            gold,
            lexical_offset,
        }))
    }
}

#[derive(Default)]
struct Tape {
    rows: Vec<Row>,
    keys: Vec<Key>,
}

struct Annotation {
    gold_pos: Option<usize>,
    rival_pos: Option<usize>,
    status: Status,
}

struct Entry<'a> {
    group: &'a str,
    pair: &'a str,
    cohort: &'a str,
    topic: &'a str,
    assignment: usize,
    side: usize,
    equal_length: bool,
}

impl Tape {
    fn push(
        &mut self,
        entry: Entry<'_>,
        case: Case,
        label: (u32, u32),
        rival: (u32, u32),
        annotation: Annotation,
        source: serde_json::Value,
    ) {
        self.rows.push(Row {
            pair_id: entry.pair.to_owned(),
            group_id: entry.group.to_owned(),
            assignment: entry.assignment,
            side: entry.side,
            cohort: entry.cohort.to_owned(),
            prefix: case.prefix,
            target: label.0,
            rival: rival.0,
            raw_target: label.1,
            raw_rival: rival.1,
            lexical_offset: case.lexical_offset,
            leading_whitespace: case.gold[..case.lexical_offset].to_vec(),
            space_token: SPACE,
            equal_length: entry.equal_length,
            context_start: case.context_start,
            context_end: case.context_end,
            gold_pos: annotation.gold_pos,
            rival_pos: annotation.rival_pos,
            source_status: annotation.status,
        });
        self.keys.push(Key {
            pair_id: entry.pair.to_owned(),
            group_id: entry.group.to_owned(),
            assignment: entry.assignment,
            side: entry.side,
            context: case.context,
            user: case.user,
            question: case.question,
            reply: case.reply,
            gold: case.gold,
            topic: entry.topic.to_owned(),
            source,
        });
    }

    fn write(&self, path: &Path) -> Result<()> {
        fs::create_dir(path)?;
        write_jsonl(&path.join("rows.jsonl"), &self.rows)?;
        write_jsonl(&path.join("keys.jsonl"), &self.keys)?;
        Ok(())
    }

    fn read(path: &Path) -> Result<Self> {
        Ok(Self {
            rows: read_jsonl(&path.join("rows.jsonl"))?,
            keys: read_jsonl(&path.join("keys.jsonl"))?,
        })
    }
}

fn sha(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

fn hash(domain: &str, key: &str) -> [u8; 32] {
    Sha256::digest(format!("{SEED}\0{domain}\0{key}")).into()
}

fn id(domain: &str, key: &str) -> String {
    hex::encode(&hash(domain, key)[..12])
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

fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|s| !s.is_empty())
}

fn prompt_digest(text: &str) -> String {
    hex::encode(&Sha256::digest(normalized(text))[..8])
}

fn require(value: bool, message: impl Into<String>) -> Result<()> {
    if !value {
        return Err(message.into().into());
    }
    Ok(())
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn write_jsonl<T: Serialize>(path: &Path, values: &[T]) -> Result<()> {
    let mut output = writer(path)?;
    for value in values {
        serde_json::to_writer(&mut output, value)?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut output = writer(path)?;
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn run() -> Result<()> {
    let values: Vec<_> = std::env::args().skip(1).collect();
    let args: Vec<_> = values.iter().map(String::as_str).collect();
    let (operation, root, rest) = match args.as_slice() {
        [operation @ ("preview" | "build" | "verify"), root, rest @ ..] => (*operation, Path::new(root), rest),
        _ => return Err("usage: shared-source-panel preview INPUT_ROOT | build INPUT_ROOT REVIEW NEW_OUTPUT | verify INPUT_ROOT REVIEW SEALED_OUTPUT".into()),
    };
    let inputs = inputs::Inputs::load(root)?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(&inputs.tokenizer)?)
        .ok_or("invalid tokenizer")?;
    require(
        tokenizer.vocab_size() == 4096 && tokenizer.encode(" ") == [SPACE],
        "tokenizer identity drift",
    )?;
    let protocol = DialogueProtocol::literal_roles_v2(&tokenizer)?;
    let builder = Builder::new(&tokenizer, &protocol)?;
    let donor = inputs.load_donor(&tokenizer)?;
    let prior = inputs.load_prior()?;
    match (operation, rest) {
        ("preview", []) => human::preview(&builder, &inputs, &prior),
        ("build", [review, output]) => {
            let output = Path::new(output);
            let review = Path::new(review);
            let (train, train_absent) = controlled::train(&builder, &prior)?;
            let (held, absent) = controlled::held(&builder, &prior, &donor, &train)?;
            let human = human::build(&builder, &inputs, &prior, review)?;
            let tapes = [train, held, human, absent, train_absent];
            let audit = audit::check(&builder, &prior, &donor, tapes.each_ref())?;
            fs::create_dir(output)?;
            for (split, tape) in SPLITS.into_iter().zip(tapes) {
                tape.write(&output.join(split))?;
            }
            let mut hashes = BTreeMap::new();
            for split in SPLITS {
                for kind in ["keys", "rows"] {
                    let name = format!("{split}/{kind}.jsonl");
                    hashes.insert(name.clone(), sha(&output.join(name))?);
                }
            }
            let receipt = serde_json::json!({
                "schema": "uor-r4.shared-source-occurrence-panel/1",
                "seed": SEED, "limit": LIMIT, "space_token": SPACE, "assistant_suffix": MARKER,
                "protocol": protocol.identity()?, "inputs": inputs.pins, "review_sha256": sha(review)?,
                "audit": audit, "outputs": hashes,
            });
            write_json(&output.join("receipt.json"), &receipt)?;
            hashes.insert(
                "receipt.json".to_owned(),
                sha(&output.join("receipt.json"))?,
            );
            let mut seal = writer(&output.join("SHA256SUMS"))?;
            for (name, hash) in hashes {
                writeln!(seal, "{hash}  {name}")?;
            }
            seal.flush()?;
            Ok(())
        }
        ("verify", [review, path]) => {
            let output = Path::new(path);
            let receipt: serde_json::Value =
                serde_json::from_reader(File::open(output.join("receipt.json"))?)?;
            require(
                receipt["inputs"] == serde_json::to_value(&inputs.pins)?
                    && receipt["protocol"] == protocol.identity()?
                    && receipt["review_sha256"] == sha(Path::new(review))?,
                "receipt input identity changed",
            )?;
            audit::verify_seal(output, &receipt)?;
            let [train, held, human, absent, train_absent] =
                SPLITS.map(|split| Tape::read(&output.join(split)));
            let tapes = [train?, held?, human?, absent?, train_absent?];
            let census = audit::check(&builder, &prior, &donor, tapes.each_ref())?;
            require(
                census == receipt["audit"],
                "persisted audit differs from fresh readback",
            )?;
            human::verify(&builder, &inputs, &tapes[2], Path::new(review))?;
            println!("{}", serde_json::to_string_pretty(&census)?);
            Ok(())
        }
        _ => Err("unexpected operands for requested panel operation".into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("shared-source-panel: {error}");
        std::process::exit(1);
    }
}
