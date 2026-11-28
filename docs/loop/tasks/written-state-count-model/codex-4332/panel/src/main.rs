#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const CENSUS: &str = "20261127-codex-whole-turn-census";
const SQUAD_SHA: &str = "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002";
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const CLASSES_SHA: &str = "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23";
const SUPPLY_SHA: &str = "b8cc4c6ff0e65f731cf909df48c72e1497d72217d431d1d75c23ae50cdd5201e";
const FIT_PANEL_SHA: &str = "78a3a3196c8099c01fd191fa73573b0846cd3cf7b867825f08ddeae9b27e2151";
const HEAD: [u32; 4] = [0, 55, 2728, 28];
const TAIL: [u32; 6] = [201, 35, 560, 652, 714, 28];

#[derive(Deserialize)]
struct Dataset {
    data: Vec<Article>,
}

#[derive(Deserialize)]
struct Article {
    title: String,
    paragraphs: Vec<Paragraph>,
}

#[derive(Deserialize)]
struct Paragraph {
    context: String,
}

#[derive(Deserialize, Serialize)]
struct Origin {
    article: String,
    paragraph: usize,
}

#[derive(Deserialize)]
struct Pair {
    kind: String,
    passage: Option<String>,
    origin: Option<Origin>,
    sites: [Anchor; 2],
    turns: [Turn; 2],
}

#[derive(Deserialize)]
struct Anchor {
    span: String,
    context: String,
}

#[derive(Deserialize)]
struct Turn {
    question: String,
    target: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct Site {
    pos: usize,
    id: u32,
    bytes: [usize; 2],
}

#[derive(Serialize)]
struct Input {
    case_id: String,
    prefix: Vec<u32>,
    user_bounds: [usize; 2],
    question_start: usize,
}

#[derive(Serialize)]
struct Prompt {
    case_id: String,
    text: String,
}

#[derive(Serialize)]
struct Key<'a> {
    case_id: String,
    pair_id: String,
    kind: &'a str,
    target: Option<Site>,
    sites: [Site; 2],
    source_origin: Option<&'a Origin>,
}

struct Encoded<'a> {
    input: Input,
    prompt: Prompt,
    key: Key<'a>,
}

fn require(condition: bool, reason: &str) -> Result<()> {
    if !condition {
        return Err(reason.into());
    }
    Ok(())
}

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn checked(path: &Path, pin: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        sha(&bytes) == pin,
        &format!("digest mismatch: {}", path.display()),
    )?;
    Ok(bytes)
}

fn sealed_file(root: &Path, manifest_hash: &str, name: &str) -> Result<Vec<u8>> {
    let manifest = checked(&root.join("SEALED.sha256"), manifest_hash)?;
    let pin = std::str::from_utf8(&manifest)?
        .lines()
        .filter_map(|line| line.split_once("  "))
        .find_map(|(pin, file)| (file == name).then_some(pin))
        .ok_or("file absent from sealed manifest")?;
    checked(&root.join(name), pin)
}

fn normalized(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[derive(Default)]
struct Prior {
    articles: BTreeSet<String>,
    passages: BTreeSet<String>,
    questions: BTreeSet<String>,
    prompts: BTreeSet<String>,
}

impl Prior {
    fn load(runs: &Path) -> Result<Self> {
        let mut prior = Self::default();
        for split in ["train", "reserve", "dev"] {
            let tape = sealed_file(
                &runs.join(CENSUS).join("supply-v1"),
                SUPPLY_SHA,
                &format!("{split}-truth.jsonl"),
            )?;
            for line in std::str::from_utf8(&tape)?.lines() {
                let row: Value = serde_json::from_str(line)?;
                prior.articles.insert(normalized(
                    row["article"]
                        .as_str()
                        .ok_or("missing historical article")?,
                ));
                prior.passages.insert(
                    row["passage_sha256"]
                        .as_str()
                        .ok_or("missing historical passage hash")?
                        .to_owned(),
                );
                prior.questions.insert(normalized(
                    row["question"]
                        .as_str()
                        .ok_or("missing historical question")?,
                ));
            }
        }
        let prompts = sealed_file(
            &runs.join("20261127-codex-whole-turn-fit/panel-v1"),
            FIT_PANEL_SHA,
            "prompts.jsonl",
        )?;
        for line in std::str::from_utf8(&prompts)?.lines() {
            let row: Value = serde_json::from_str(line)?;
            prior.prompts.insert(normalized(
                row["prompt"].as_str().ok_or("missing historical prompt")?,
            ));
        }
        Ok(prior)
    }
}

fn passage<'a>(pair: &'a Pair, supply: &'a Dataset, prior: &Prior) -> Result<&'a str> {
    match (&pair.passage, &pair.origin) {
        (Some(text), None) => Ok(text),
        (None, Some(origin)) => {
            require(
                !prior.articles.contains(&normalized(&origin.article)),
                "natural article already appeared in #4329",
            )?;
            let article = supply
                .data
                .iter()
                .find(|article| article.title == origin.article)
                .ok_or("natural article not in pinned supply")?;
            let paragraph = article
                .paragraphs
                .get(origin.paragraph)
                .ok_or("invalid paragraph locator")?;
            require(
                !prior.passages.contains(&sha(paragraph.context.as_bytes())),
                "natural paragraph already appeared in #4329",
            )?;
            Ok(&paragraph.context)
        }
        _ => Err("supply exactly one authored passage or natural origin".into()),
    }
}

fn unique_start(haystack: &str, needle: &str) -> Result<usize> {
    let mut matches = haystack.match_indices(needle);
    let start = matches.next().ok_or("missing source anchor")?.0;
    require(
        !needle.is_empty() && matches.next().is_none(),
        "ambiguous source anchor",
    )?;
    Ok(start)
}

fn source(
    tokenizer: &ByteBpeTokenizer,
    ids: &[u32],
    passage: &str,
    anchor: &Anchor,
    classes: &[u8],
) -> Result<Site> {
    let start =
        unique_start(passage, &anchor.context)? + unique_start(&anchor.context, &anchor.span)?;
    let expected = tokenizer.encode(&format!(" {}", anchor.span));
    let first = *expected.first().ok_or("missing canonical answer token")?;
    require(
        first != 223 && expected.len() < 12,
        "answer does not open at the first lexical ID",
    )?;
    let mut cursor = 0;
    for (index, &id) in ids.iter().enumerate() {
        let bytes = tokenizer.decode_bytes(&[id]);
        let next = cursor + bytes.len();
        let spaces = bytes
            .iter()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        if cursor + spaces == start + 1 && id == first && next <= start + 1 + anchor.span.len() {
            require(
                classes
                    .get(id as usize)
                    .is_some_and(|class| ![0, 9].contains(class)),
                "source token class is ineligible",
            )?;
            require(
                ids[index..].starts_with(&expected),
                "source does not preserve the full canonical answer IDs",
            )?;
            return Ok(Site {
                pos: HEAD.len() + index,
                id,
                bytes: [start, start + anchor.span.len()],
            });
        }
        cursor = next;
    }
    Err("no exact original token at the source byte address".into())
}

fn encode_pair<'a>(
    pair: &'a Pair,
    passage: &str,
    tokenizer: &ByteBpeTokenizer,
    encoder: &DialogueEncoder<'_>,
    classes: &[u8],
    prior: &Prior,
) -> Result<Vec<Encoded<'a>>> {
    let pair_id = sha(format!("uor-4332-pair-v1\0{passage}").as_bytes());
    let mut rows = Vec::new();
    for turn in &pair.turns {
        let question = normalized(&turn.question);
        require(
            !prior.questions.contains(&question),
            "reused #4329 question",
        )?;
        require(
            !question.is_empty() && !turn.question.contains('\n'),
            "invalid question",
        )?;
        let text = format!("{passage}\n\n{}", turn.question);
        require(
            !prior.prompts.contains(&normalized(&text)),
            "reused #4330 prompt",
        )?;
        let rendered = format!(" {text}");
        let ids = tokenizer.encode(&rendered);
        let prefix = encoder.encode_assistant_prefix(&[Message {
            role: "user",
            content: &text,
        }]);
        require(
            prefix.emitted_turns == 1
                && prefix.special_token_occurrences == 0
                && prefix.tokens == [HEAD.as_slice(), &ids, &TAIL].concat()
                && tokenizer.decode_bytes(&ids) == rendered.as_bytes()
                && prefix.tokens.len() + 12 <= 192,
            "invalid ordinary prefix",
        )?;
        let user_bounds = [HEAD.len(), HEAD.len() + ids.len()];
        let separator = tokenizer.encode(&format!(" {passage}\n\n"));
        require(
            ids.starts_with(&separator),
            "question token crosses passage boundary",
        )?;
        let sites = [
            source(tokenizer, &ids, passage, &pair.sites[0], classes)?,
            source(tokenizer, &ids, passage, &pair.sites[1], classes)?,
        ];
        require(
            sites[0].pos < sites[1].pos,
            "sites must be distinct and in passage order",
        )?;
        if pair.kind == "equal-id" {
            require(
                sites[0].id == sites[1].id,
                "equal-ID reversal has different token IDs",
            )?;
        } else if pair.kind.ends_with("rank") {
            require(
                sites[0].id != sites[1].id,
                "distinct-ID reversal has equal token IDs",
            )?;
        }
        let target = turn
            .target
            .map(|index| sites.get(index).copied().ok_or("invalid target index"))
            .transpose()?;
        let case_id = sha(format!("uor-4332-case-v1\0{text}").as_bytes());
        rows.push(Encoded {
            input: Input {
                case_id: case_id.clone(),
                prefix: prefix.tokens,
                user_bounds,
                question_start: HEAD.len() + separator.len(),
            },
            prompt: Prompt {
                case_id: case_id.clone(),
                text,
            },
            key: Key {
                case_id,
                pair_id: pair_id.clone(),
                kind: &pair.kind,
                target,
                sites,
                source_origin: pair.origin.as_ref(),
            },
        });
    }
    let [left, right] = rows.as_slice() else {
        return Err("wrong pair count".into());
    };
    require(
        left.key.sites == right.key.sites
            && left.input.question_start == right.input.question_start
            && left.input.prefix[..left.input.question_start]
                == right.input.prefix[..right.input.question_start],
        "source address or prequestion tokens differ within pair",
    )?;
    let targets = (left.key.target, right.key.target);
    let valid = match pair.kind.as_str() {
        "rank" | "equal-id" | "natural-rank" => matches!(targets, (Some(a), Some(b)) if a != b),
        "support" | "natural-support" => matches!(targets, (Some(_), None) | (None, Some(_))),
        _ => false,
    };
    require(
        valid && pair.kind.starts_with("natural-") == pair.origin.is_some(),
        "invalid pair type or targets",
    )?;
    Ok(rows)
}

fn tape<T: Serialize>(root: &Path, name: &str, rows: impl IntoIterator<Item = T>) -> Result<()> {
    let mut file = BufWriter::new(File::create(root.join(name))?);
    for row in rows {
        serde_json::to_writer(&mut file, &row)?;
        writeln!(file)?;
    }
    file.flush()?;
    let mut permissions = file.get_ref().metadata()?.permissions();
    permissions.set_readonly(true);
    file.get_ref().set_permissions(permissions)?;
    Ok(())
}

fn seal(root: &Path, names: &[&str]) -> Result<()> {
    let mut manifest = String::new();
    for name in names {
        use std::fmt::Write;
        writeln!(manifest, "{}  {name}", sha(&fs::read(root.join(name))?))?;
    }
    let path = root.join("SEALED.sha256");
    fs::write(&path, manifest)?;
    let mut permissions = fs::metadata(&path)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&path, permissions)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [runs, spec, output] = args.as_slice() else {
        return Err(
            "usage: whole-turn-recurrence-panel WICK_RUNS PRIVATE_PAIR_SPEC NEW_OUTPUT_DIR".into(),
        );
    };
    let runs = Path::new(runs);
    let output = Path::new(output);
    let supply = runs
        .join(CENSUS)
        .join("supply-inputs/squad-train-v2.0.json");
    let dataset: Dataset = serde_json::from_slice(&checked(&supply, SQUAD_SHA)?)?;
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked(
        &runs.join("20261127-codex-visible-answer/panel/inputs/tokenizer.json"),
        TOKENIZER_SHA,
    )?)
    .ok_or("invalid pinned tokenizer")?;
    let classes = checked(
        &runs.join("20261127-codex-shared-source/product-inputs-v1/token-classes.u8"),
        CLASSES_SHA,
    )?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let prior = Prior::load(runs)?;
    let pairs: Vec<Pair> = serde_json::from_slice(&fs::read(spec)?)?;
    let mut rows = Vec::new();
    let mut passages = BTreeSet::new();
    for (index, pair) in pairs.iter().enumerate() {
        let text = passage(pair, &dataset, &prior)?;
        require(passages.insert(normalized(text)), "duplicate panel passage")?;
        let mut encoded = encode_pair(pair, text, &tokenizer, &encoder, &classes, &prior)
            .map_err(|error| format!("pair {}: {error}", index + 1))?;
        rows.append(&mut encoded);
    }
    rows.sort_unstable_by(|left, right| left.input.case_id.cmp(&right.input.case_id));
    require(
        rows.windows(2)
            .all(|pair| pair[0].input.case_id != pair[1].input.case_id),
        "duplicate case ID",
    )?;
    require(
        [
            "rank",
            "equal-id",
            "support",
            "natural-rank",
            "natural-support",
        ]
        .iter()
        .all(|kind| pairs.iter().any(|pair| pair.kind == *kind)),
        "missing required pair type",
    )?;
    fs::create_dir(output)?;
    let blind = output.join("blind");
    let truth = output.join("truth");
    fs::create_dir(&blind)?;
    fs::create_dir(&truth)?;
    tape(&blind, "input.jsonl", rows.iter().map(|row| &row.input))?;
    tape(&blind, "prompts.jsonl", rows.iter().map(|row| &row.prompt))?;
    seal(&blind, &["input.jsonl", "prompts.jsonl"])?;
    tape(&truth, "key.jsonl", rows.iter().map(|row| &row.key))?;
    seal(&truth, &["key.jsonl"])?;
    Ok(())
}
