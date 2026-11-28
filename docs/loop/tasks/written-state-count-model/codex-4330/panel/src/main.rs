#![forbid(unsafe_code)]

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::{DialogueEncoder, DialogueProtocol, Message};
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
const CLASSES: &str = "20261127-codex-shared-source/product-inputs-v1/token-classes.u8";
const SUPPLY_ROOT: &str = "20261127-codex-whole-turn-census/supply-v1";
const AUTHORED_ROOT: &str = "20261127-codex-whole-turn-census/contrasts-v1";
const PROTECTED_ROOT: &str = "20261127-codex-ordinary-source/panel-v1";
const TOKENIZER_SHA: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const CLASSES_SHA: &str = "a0afae48560e5a8112d682fb067f24c57a2f432935b2405acfcc8b7dd7d31a23";
const SUPPLY_SHA: &str = "b8cc4c6ff0e65f731cf909df48c72e1497d72217d431d1d75c23ae50cdd5201e";
const AUTHORED: [(&str, &str); 2] = [
    (
        "train",
        "a54f3f41c21fc81fc7d3b4e7d88fd5784ec2ccc4529775a69c63fdbaad0e58dd",
    ),
    (
        "dev",
        "3aef039f21c070792629376c8e822fc7d1362cf38973563841bbd98ecf0838db",
    ),
];
const PROTECTED_PROMPTS_SHA: &str =
    "475d11303e92b9f9a2e03d2499ef877be74220170d5ed75bb18d8ccdc318ff43";
const PROTECTED_TRUTH_SHA: &str =
    "9787903baae7d402ec5b7ce4e1b65008efc437773a8e2e76b53e7b606171eba0";
const USER: [u32; 4] = [0, 55, 2728, 28];
const ASSISTANT: [u32; 6] = [201, 35, 560, 652, 714, 28];
const SPACE: u32 = 223;
const EOS: u32 = 1;
const CAP: usize = 12;

#[derive(Deserialize, Serialize)]
struct Prompt {
    case_id: String,
    prompt: String,
}

#[derive(Serialize)]
struct Eval<'a> {
    case_id: &'a str,
    prefix: &'a [u32],
    user_bounds: [usize; 2],
}

#[derive(Serialize)]
struct Train<'a> {
    #[serde(flatten)]
    eval: Eval<'a>,
    target: Option<usize>,
}

#[derive(Clone, Deserialize, Serialize)]
struct Source {
    pos: usize,
    id: u32,
}

#[derive(Serialize)]
struct Gold {
    case_id: String,
    cohort: String,
    split: String,
    train: bool,
    source_id: String,
    family: String,
    contrast_group: String,
    contrast_type: String,
    paired_ids: Vec<String>,
    supported: bool,
    source: Option<Source>,
    source_delay: usize,
    expected_ids: Option<Vec<u32>>,
    same_id_rival: Option<Source>,
}

struct Row {
    prompt: Prompt,
    prefix: Vec<u32>,
    bounds: [usize; 2],
    gold: Gold,
}

#[derive(Deserialize)]
struct NaturalLabel {
    first_position: usize,
    first_id: u32,
    response_ids: Vec<u32>,
    response_lexical_offset: usize,
}

#[derive(Deserialize)]
struct Natural {
    case_id: String,
    paired_with: String,
    article: String,
    passage_sha256: String,
    is_impossible: bool,
    prefix: Vec<u32>,
    user_bounds: [usize; 2],
    gold_source: Option<NaturalLabel>,
    plausible_only: Option<NaturalLabel>,
    same_first_position: bool,
}

#[derive(Deserialize)]
struct Authored {
    prefix: Vec<u32>,
    prompt: String,
    split: String,
    family: String,
    expected_ids: Vec<u32>,
    supported: bool,
    source_pos: Option<usize>,
    site_positions: [usize; 2],
    site_ids: [u32; 2],
    statement_type: String,
    changed_side: Option<usize>,
    fact_order: usize,
    value_assignment: [usize; 2],
    asked: usize,
}

#[derive(Deserialize)]
struct Protected {
    case_id: String,
    expected_ids: Vec<u32>,
    family: String,
    group: String,
    prompt_ids: Vec<u32>,
    supported: bool,
    sources: Vec<Source>,
    same_id_rival: Option<Source>,
    fact_order: usize,
    question_target: String,
}

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.to_owned().into());
    }
    Ok(())
}

fn checked(path: &Path, expected: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    require(
        hex::encode(Sha256::digest(&bytes)) == expected,
        &format!("input changed at {}", path.display()),
    )?;
    Ok(bytes)
}

fn rows<T: DeserializeOwned>(path: &Path, expected: &str) -> Result<Vec<T>> {
    let bytes = checked(path, expected)?;
    std::str::from_utf8(&bytes)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

fn paired_prompts(path: &Path, expected: &str) -> Result<BTreeMap<String, String>> {
    let mut prompts = BTreeMap::new();
    for row in rows::<Prompt>(path, expected)? {
        require(
            prompts.insert(row.case_id, row.prompt).is_none(),
            "duplicate input prompt ID",
        )?;
    }
    Ok(prompts)
}

fn bounds(prefix: &[u32]) -> Result<[usize; 2]> {
    require(
        prefix.starts_with(&USER)
            && prefix.ends_with(&ASSISTANT)
            && prefix.len() > USER.len() + ASSISTANT.len()
            && prefix.len() <= 192,
        "prefix outside ordinary protocol-2 bounds",
    )?;
    Ok([USER.len(), prefix.len() - ASSISTANT.len()])
}

fn row(prompt: String, prefix: Vec<u32>, gold: Gold) -> Result<Row> {
    let bounds = bounds(&prefix)?;
    Ok(Row {
        prompt: Prompt {
            case_id: gold.case_id.clone(),
            prompt,
        },
        prefix,
        bounds,
        gold,
    })
}

fn natural(runs: &Path, output: &mut Vec<Row>) -> Result<()> {
    let root = runs.join(SUPPLY_ROOT);
    let manifest = checked(&root.join("SEALED.sha256"), SUPPLY_SHA)?;
    let mut pins = BTreeMap::new();
    for line in std::str::from_utf8(&manifest)?.lines() {
        let (hash, name) = line.split_once("  ").ok_or("malformed supply pin")?;
        pins.insert(name.to_owned(), hash.to_owned());
    }
    for split in ["train", "reserve", "dev"] {
        let prompt_name = format!("{split}-prompts.jsonl");
        let truth_name = format!("{split}-truth.jsonl");
        let prompts = paired_prompts(&root.join(&prompt_name), &pins[&prompt_name])?;
        let truth = rows::<Natural>(&root.join(&truth_name), &pins[&truth_name])?;
        let by_id: BTreeMap<_, _> = truth
            .iter()
            .map(|entry| (entry.case_id.as_str(), entry))
            .collect();
        require(
            prompts.len() == by_id.len() && by_id.len() == truth.len(),
            "misaligned natural prompt and key lists",
        )?;
        for entry in &truth {
            let partner = by_id
                .get(entry.paired_with.as_str())
                .ok_or("missing natural partner")?;
            require(
                entry.case_id == partner.paired_with
                    && entry.is_impossible != partner.is_impossible
                    && entry.passage_sha256 == partner.passage_sha256,
                "invalid natural pair",
            )?;
            let (positive, negative) = if entry.is_impossible {
                (*partner, entry)
            } else {
                (entry, *partner)
            };
            let positive_label = positive
                .gold_source
                .as_ref()
                .ok_or("missing natural positive label")?;
            require(
                positive.plausible_only.is_none()
                    && negative.gold_source.is_none()
                    && negative.plausible_only.is_some(),
                "inconsistent natural label",
            )?;
            if split == "train" && positive_label.response_lexical_offset != 0 {
                continue;
            }
            require(
                entry.user_bounds == bounds(&entry.prefix)?,
                "natural user bounds changed",
            )?;
            let source_id = entry.case_id.clone();
            let case_id = format!("squad-{split}-{source_id}");
            let label = entry.gold_source.as_ref();
            let source = label.map(|value| Source {
                pos: value.first_position,
                id: value.first_id,
            });
            let source_delay = label.map_or(0, |value| value.response_lexical_offset);
            require(
                source_delay <= 1
                    && label.is_none_or(|value| {
                        value.response_ids.get(source_delay) == Some(&value.first_id)
                            && value.response_ids[..source_delay]
                                .iter()
                                .all(|&id| id == SPACE)
                    }),
                "natural answer token shifted",
            )?;
            let expected_ids = label.map(|value| [value.response_ids.as_slice(), &[EOS]].concat());
            output.push(row(
                prompts
                    .get(&source_id)
                    .ok_or("missing natural prompt")?
                    .clone(),
                entry.prefix.clone(),
                Gold {
                    case_id,
                    cohort: "squad".to_owned(),
                    split: split.to_owned(),
                    train: split == "train",
                    source_id,
                    family: entry.article.clone(),
                    contrast_group: format!("squad-{split}-{}", entry.passage_sha256),
                    contrast_type: if entry.same_first_position {
                        "same_source_position"
                    } else {
                        "same_paragraph"
                    }
                    .to_owned(),
                    paired_ids: vec![format!("squad-{split}-{}", entry.paired_with)],
                    supported: !entry.is_impossible,
                    source,
                    source_delay,
                    expected_ids,
                    same_id_rival: None,
                },
            )?);
        }
    }
    Ok(())
}

fn authored(runs: &Path, output: &mut Vec<Row>) -> Result<()> {
    let root = runs.join(AUTHORED_ROOT);
    for (split, pin) in AUTHORED {
        let input = rows::<Authored>(&root.join(format!("{split}.jsonl")), pin)?;
        let groups: Vec<_> = input
            .iter()
            .map(|entry| {
                format!(
                    "authored-{split}-{}-{}-{}{}-{}-{}",
                    entry.family,
                    entry.fact_order,
                    entry.value_assignment[0],
                    entry.value_assignment[1],
                    entry.statement_type,
                    entry.changed_side.map_or(2, |side| side),
                )
            })
            .collect();
        let case_id = |index: usize| format!("authored-{split}-{:03}", index + 1);
        for (index, entry) in input.iter().enumerate() {
            require(
                entry.split == split
                    && entry.asked < 2
                    && entry.supported == entry.source_pos.is_some(),
                "inconsistent authored label",
            )?;
            let partners: Vec<_> = input
                .iter()
                .enumerate()
                .filter(|(peer, other)| {
                    groups[*peer] == groups[index] && other.asked != entry.asked
                })
                .map(|(peer, _)| case_id(peer))
                .collect();
            require(
                partners.len() == 1,
                "authored row needs one question contrast",
            )?;
            let source = entry.source_pos.map(|pos| Source {
                pos,
                id: entry.site_ids[entry.asked],
            });
            let rival = 1 - entry.asked;
            let same_id_rival = source.as_ref().and_then(|gold| {
                (gold.id == entry.site_ids[rival]).then_some(Source {
                    pos: entry.site_positions[rival],
                    id: entry.site_ids[rival],
                })
            });
            output.push(row(
                entry.prompt.clone(),
                entry.prefix.clone(),
                Gold {
                    case_id: case_id(index),
                    cohort: "authored".to_owned(),
                    split: split.to_owned(),
                    train: split == "train",
                    source_id: (index + 1).to_string(),
                    family: entry.family.clone(),
                    contrast_group: groups[index].clone(),
                    contrast_type: entry.statement_type.clone(),
                    paired_ids: partners,
                    supported: entry.supported,
                    source,
                    source_delay: 0,
                    expected_ids: Some(entry.expected_ids.clone()),
                    same_id_rival,
                },
            )?);
        }
    }
    Ok(())
}

fn protected(runs: &Path, output: &mut Vec<Row>) -> Result<()> {
    let root = runs.join(PROTECTED_ROOT);
    let prompts = paired_prompts(&root.join("prompts.jsonl"), PROTECTED_PROMPTS_SHA)?;
    let truth = rows::<Protected>(&root.join("truth.jsonl"), PROTECTED_TRUTH_SHA)?;
    require(
        prompts.len() == truth.len(),
        "misaligned protected prompts and key lists",
    )?;
    for entry in &truth {
        let source = entry.sources.first().cloned();
        require(
            source.is_some() == entry.supported,
            "missing protected source",
        )?;
        let partners: Vec<_> = truth
            .iter()
            .filter(|other| {
                entry.case_id != other.case_id
                    && entry.family == other.family
                    && entry.fact_order == other.fact_order
                    && ((entry.group.starts_with("release_")
                        && other.group.starts_with("release_")
                        && entry.question_target == other.question_target)
                        || (!entry.group.starts_with("release_")
                            && entry.group == other.group
                            && entry.question_target != other.question_target))
            })
            .map(|peer| format!("protected-{}", peer.case_id))
            .collect();
        output.push(row(
            prompts
                .get(&entry.case_id)
                .ok_or("missing protected prompt")?
                .clone(),
            entry.prompt_ids.clone(),
            Gold {
                case_id: format!("protected-{}", entry.case_id),
                cohort: "protected".to_owned(),
                split: "protected".to_owned(),
                train: false,
                source_id: entry.case_id.clone(),
                family: entry.family.clone(),
                contrast_group: format!(
                    "protected-{}-{}-{}",
                    entry.family, entry.group, entry.fact_order
                ),
                contrast_type: entry.group.clone(),
                paired_ids: partners,
                supported: entry.supported,
                source,
                source_delay: 0,
                expected_ids: Some(entry.expected_ids.clone()),
                same_id_rival: entry.same_id_rival.clone(),
            },
        )?);
    }
    Ok(())
}

fn verify(encoder: &DialogueEncoder<'_>, classes: &[u8], rows: &[Row]) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut prefixes = BTreeSet::new();
    let mut counts = BTreeMap::new();
    for entry in rows {
        let gold = &entry.gold;
        let fail = |reason: &str| format!("{reason} for {}", gold.case_id);
        require(
            ids.insert(&gold.case_id) && prefixes.insert(&entry.prefix),
            &fail("duplicate case or prefix"),
        )?;
        let rendered = encoder.encode_assistant_prefix(&[Message {
            role: "user",
            content: &entry.prompt.prompt,
        }]);
        require(
            rendered.tokens == entry.prefix
                && rendered.emitted_turns == 1
                && rendered.special_token_occurrences == 0
                && !entry.prompt.prompt.contains('\0'),
            &fail("ordinary raw prompt differs from sealed prefix"),
        )?;
        let [start, end] = entry.bounds;
        for source in gold.source.iter().chain(gold.same_id_rival.iter()) {
            require(
                (start..end).contains(&source.pos)
                    && entry.prefix[source.pos] == source.id
                    && classes
                        .get(source.id as usize)
                        .is_some_and(|class| ![0, 9].contains(class)),
                &fail("source outside the eligible user role or wrong original ID"),
            )?;
        }
        require(
            gold.source.is_some() == gold.supported && (!gold.train || gold.source_delay == 0),
            &fail("invalid source timing or support"),
        )?;
        if let Some(expected) = &gold.expected_ids {
            require(
                expected.len() <= CAP
                    && expected.last() == Some(&EOS)
                    && !expected[..expected.len() - 1].contains(&EOS)
                    && entry.prefix.len() + CAP <= 192
                    && gold
                        .source
                        .as_ref()
                        .is_none_or(|source| expected.get(gold.source_delay) == Some(&source.id)),
                &fail("invalid expected reply or causal reserve"),
            )?;
        }
        *counts
            .entry((gold.cohort.as_str(), gold.split.as_str()))
            .or_insert(0) += 1;
    }
    require(
        counts
            == BTreeMap::from([
                (("squad", "train"), 228),
                (("squad", "reserve"), 400),
                (("squad", "dev"), 50),
                (("authored", "train"), 240),
                (("authored", "dev"), 120),
                (("protected", "protected"), 48),
            ]),
        &format!("unexpected panel counts {counts:?}"),
    )?;
    for entry in rows {
        require(
            entry.gold.paired_ids.iter().all(|id| ids.contains(id)),
            "dangling paired case",
        )?;
    }
    Ok(())
}

fn write(stream: &mut BufWriter<File>, row: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *stream, row)?;
    writeln!(stream)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [runs, output] = args.as_slice() else {
        return Err("usage: whole-turn-fit-panel WICK_RUNS NEW_OUTPUT_DIR".into());
    };
    let runs = Path::new(runs);
    let output = Path::new(output);
    let tokenizer = ByteBpeTokenizer::from_tokenizer_json_bytes(&checked(
        &runs.join(TOKENIZER),
        TOKENIZER_SHA,
    )?)
    .ok_or("invalid pinned tokenizer")?;
    let classes = checked(&runs.join(CLASSES), CLASSES_SHA)?;
    require(classes.len() == 4096, "invalid class table")?;
    let encoder = DialogueProtocol::literal_roles_v2(&tokenizer)?.bind(&tokenizer)?;
    let mut gathered = Vec::new();
    natural(runs, &mut gathered)?;
    authored(runs, &mut gathered)?;
    protected(runs, &mut gathered)?;
    gathered.sort_unstable_by(|left, right| left.gold.case_id.cmp(&right.gold.case_id));
    verify(&encoder, &classes, &gathered)?;
    fs::create_dir(output)?;
    let mut train = BufWriter::new(File::create(output.join("train.jsonl"))?);
    let mut eval = BufWriter::new(File::create(output.join("eval.jsonl"))?);
    let mut gold = BufWriter::new(File::create(output.join("gold.jsonl"))?);
    let mut prompts = BufWriter::new(File::create(output.join("prompts.jsonl"))?);
    for entry in &gathered {
        let eval_row = Eval {
            case_id: &entry.gold.case_id,
            prefix: &entry.prefix,
            user_bounds: entry.bounds,
        };
        write(&mut eval, &eval_row)?;
        if entry.gold.train {
            write(
                &mut train,
                &Train {
                    eval: eval_row,
                    target: entry.gold.source.as_ref().map(|source| source.pos),
                },
            )?;
        }
        write(&mut gold, &entry.gold)?;
        write(&mut prompts, &entry.prompt)?;
    }
    for mut stream in [train, eval, gold, prompts] {
        stream.flush()?;
    }
    Ok(())
}
