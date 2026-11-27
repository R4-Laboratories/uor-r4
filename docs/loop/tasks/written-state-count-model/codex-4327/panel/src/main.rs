#![forbid(unsafe_code)]

#[allow(dead_code)]
#[path = "../../../codex-4324/panel/src/archive.rs"]
mod archive;
mod design;
#[path = "../../../codex-4325/panel/src/generate.rs"]
mod generate;
mod independent;
#[allow(dead_code)]
#[path = "../../../codex-4324/panel/src/design.rs"]
mod old_design;
#[allow(dead_code)]
#[path = "../../../codex-4325/panel/src/design.rs"]
mod prior_design;

use archive::{normalized, template, Archive};
use design::HELD;
use generate::{Generator, Record, Wire};
use prior_design::Design;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use uor_r4_tokenizer::dialogue::DialogueProtocol;
use uor_r4_tokenizer::ByteBpeTokenizer;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TOKENIZER: &str = "20261127-codex-visible-answer/panel/inputs/tokenizer.json";
const TOKENIZER_SHA256: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const OLD_FACTORIAL: &str = "20261127-codex-release-factorial";
const OLD_FACTORIAL_SHA256: &str =
    "ed25f9d7daa1087fbd7c42423acb6a3a8ec75d5bd123f6fe49804fe75c8fdbd5";
const JOINT: &str = "20261127-codex-joint-release";
const JOINT_SHA256: &str = "700acd0859ed42e08603457ea784e4065ca7c4944986a314acf72c83a0672d6d";
const EOS: u32 = 1;
const PERIOD: u32 = 16;
const QUESTION: u32 = 33;
const FACT_FENCE: [u32; 2] = [201, 201];
const MARKER: [u32; 6] = [201, 35, 560, 652, 714, 28];
const CAP: usize = 5;
const WRITER_LIMIT: usize = 192;

#[derive(Deserialize, Serialize)]
struct Row {
    #[serde(default)]
    row_id: String,
    pair_id: String,
    side: u8,
    contrast: u8,
    positions: [u32; 4],
    ids: [u32; 4],
    prefix: Vec<u32>,
    expected: Vec<u32>,
    family: String,
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if !condition {
        return Err(message.into().into());
    }
    Ok(())
}

fn hash(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    io::copy(&mut File::open(path)?, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(&line?)
                .map_err(|error| format!("{}:{}: {error}", path.display(), index + 1).into())
        })
        .collect()
}

fn string<'a>(key: &'a Value, field: &str) -> Result<&'a str> {
    key[field]
        .as_str()
        .ok_or_else(|| format!("missing string in {field}").into())
}

fn read_split(dir: &Path) -> Result<Vec<Record>> {
    verify_manifest(dir)?;
    let rows: Vec<Row> = read_jsonl(&dir.join("rows.jsonl"))?;
    let keys: Vec<Value> = read_jsonl(&dir.join("keys.jsonl"))?;
    require(rows.len() == keys.len(), "row/key count differs")?;
    rows.into_iter()
        .zip(keys)
        .map(|(row, key)| -> Result<Record> {
            require(
                key["pair_id"] == row.pair_id
                    && key["side"] == row.side
                    && key["contrast"] == row.contrast,
                format!("row/key identity differs at {}", row.pair_id),
            )?;
            Ok(Record { row, key })
        })
        .collect()
}

fn writer(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn jsonl(output: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    writeln!(output)?;
    Ok(())
}

fn verify_manifest(dir: &Path) -> Result<()> {
    let manifest = dir.join("SHA256SUMS");
    for line in fs::read_to_string(&manifest)?.lines() {
        let (expected, relative) = line
            .split_once("  ")
            .ok_or("invalid checksum manifest entry")?;
        require(
            hash(&dir.join(relative))? == expected,
            format!("{} checksum changed: {relative}", manifest.display()),
        )?;
    }
    Ok(())
}

fn write_manifest(dir: &Path, members: &[&str]) -> Result<()> {
    let mut manifest = writer(&dir.join("SHA256SUMS"))?;
    for member in members {
        writeln!(manifest, "{}  {member}", hash(&dir.join(member))?)?;
    }
    manifest.flush()?;
    verify_manifest(dir)
}

fn archives(runs: &Path, vocab: &ByteBpeTokenizer) -> Result<Archive> {
    let mut archive = Archive::read(runs, vocab)?;
    for (run, digest, splits) in [
        (OLD_FACTORIAL, OLD_FACTORIAL_SHA256, &["fresh"][..]),
        (JOINT, JOINT_SHA256, &["train", "held"][..]),
    ] {
        let root = runs.join(run).join("panel-v1");
        require(
            hash(&root.join("SHA256SUMS"))? == digest,
            format!("{run} manifest drift"),
        )?;
        verify_manifest(&root)?;
        archive.roots.insert(run, digest);
        for split in splits {
            for Record { row, key } in read_split(&root.join(split))? {
                archive.rows += 1;
                archive.labels.extend(row.ids);
                archive.pair_ids.insert(row.pair_id);
                archive.prefixes.insert(row.prefix);
                archive.prompts.insert(normalized(string(&key, "user")?));
                archive
                    .contexts
                    .insert(normalized(string(&key, "context")?));
                archive.templates.insert(template(&row.family));
                archive
                    .templates
                    .insert(template(string(&key, "directive")?));
            }
        }
    }
    archive
        .templates
        .extend([old_design::FACT, old_design::QUESTION, old_design::FORMAT].map(template));
    for family in old_design::SCOPED {
        archive.templates.extend(family.clauses.map(template));
        archive.templates.extend(family.selectors.map(template));
    }
    for design in [&prior_design::TRAIN, &prior_design::HELD] {
        archive
            .templates
            .extend(template_strings(design)?.iter().map(|text| template(text)));
    }
    Ok(archive)
}

fn template_strings(design: &Design) -> Result<BTreeSet<String>> {
    let mut raw = BTreeSet::from([
        design.format.to_owned(),
        design.fact.to_owned(),
        design.fact_header.to_owned(),
        design.question.to_owned(),
    ]);
    for scoped in design.scoped {
        require(
            scoped.selectors.iter().all(|selector| {
                selector.matches("{a}").count() == 1
                    && selector.matches("{b}").count() == 1
                    && selector.find("{a}") < selector.find("{b}")
            }),
            format!("bad selector slots for {}", scoped.name),
        )?;
        for operative in 0..2 {
            for short in 0..2 {
                require(
                    raw.insert(scoped.directive(operative, short)),
                    format!("duplicate rendered directive for {}", scoped.name),
                )?;
            }
        }
        raw.insert(scoped.name.to_owned());
        raw.extend(scoped.clauses.map(str::to_owned));
        raw.extend(scoped.selectors.map(str::to_owned));
    }
    for direct in design.direct {
        raw.extend([direct.name, direct.listing, direct.question].map(str::to_owned));
        raw.extend(direct.directives.map(str::to_owned));
    }
    Ok(raw)
}

fn author(family: &str) -> &'static str {
    match family {
        "paired-type-case"
        | "paired-bell"
        | "paired-translation-booth"
        | "paired-receipt"
        | "paired-direct-clerk"
        | "paired-direct-rehearsal" => "independent",
        _ => "panel",
    }
}

fn prepare(runs: &Path, root: &Path) -> Result<()> {
    let tokenizer = runs.join(TOKENIZER);
    require(
        hash(&tokenizer)? == TOKENIZER_SHA256,
        "tokenizer identity drift",
    )?;
    let vocab = ByteBpeTokenizer::from_tokenizer_json_bytes(&fs::read(tokenizer)?)
        .ok_or("invalid tokenizer")?;
    require(
        vocab.vocab_size() == 4096
            && vocab.encode(".") == [PERIOD]
            && vocab.encode("?") == [QUESTION]
            && vocab.encode("\n\n") == FACT_FENCE,
        "canonical text token identity drift",
    )?;
    let protocol = DialogueProtocol::literal_roles_v2(&vocab)?;
    let wire = Wire {
        vocab: &vocab,
        dialogue: protocol.bind(&vocab)?,
    };
    let archive = archives(runs, &vocab)?;
    let templates = template_strings(&HELD)?;
    let mut reserved = archive.labels.clone();
    for text in &templates {
        archive.check_template(text)?;
        reserved.extend(vocab.encode(text));
        reserved.extend(vocab.encode(&format!(" {text}")));
    }
    let (held, census) = Generator::new(&wire, &archive).split(&HELD, &mut reserved)?;
    require(held.len() == 512, "held row count drift")?;
    verify_matched_axes(&held)?;

    fs::create_dir(root)?;
    let split = root.join("held");
    fs::create_dir(&split)?;
    let mut rows = writer(&split.join("rows.jsonl"))?;
    let mut keys = writer(&split.join("keys.jsonl"))?;
    let mut families = BTreeMap::new();
    for Record { row, mut key } in held {
        let wording_author = author(&row.family);
        families.insert(row.family.clone(), wording_author);
        key["split"] = json!("held");
        key["source_issue"] = json!(4327);
        key["wording_author"] = json!(wording_author);
        jsonl(&mut rows, &row)?;
        jsonl(&mut keys, &key)?;
    }
    rows.flush()?;
    keys.flush()?;
    let mut receipt = writer(&split.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema": "codex-4327.paired-release-held/1",
            "tokenizer_sha256": TOKENIZER_SHA256,
            "cap": CAP,
            "original_train": {
                "tape": format!("{JOINT}/panel-v1/train"),
                "rows": 768,
                "sha256": hash(&runs.join(JOINT).join("panel-v1/train/SHA256SUMS"))?,
            },
            "archive_roots": archive.roots,
            "archive_rows_checked": archive.rows,
            "archive_template_signatures_checked": archive.templates.len(),
            "fresh_template_strings_checked": templates.len(),
            "exact_archive_overlaps": {
                "source_token_ids": 0,
                "template_signatures": 0,
                "normalized_prompts": 0,
                "normalized_contexts": 0,
                "encoded_prefixes": 0,
                "pair_ids": 0,
            },
            "family_authors": families,
            "fresh": census,
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    write_manifest(&split, &["rows.jsonl", "keys.jsonl", "receipt.json"])?;
    write_manifest(root, &["held/SHA256SUMS"])
}

fn verify_matched_axes(held: &[Record]) -> Result<()> {
    let mut contexts = BTreeMap::<_, Vec<_>>::new();
    for record in held {
        contexts
            .entry(string(&record.key, "context_id")?)
            .or_default()
            .push(record);
    }
    let mut scoped = 0;
    let mut direct = [0, 0];
    for (context, records) in contexts {
        let first = &records[0];
        require(
            records.iter().all(|r| {
                r.row.positions == first.row.positions
                    && r.row.ids == first.row.ids
                    && r.key["context"] == first.key["context"]
                    && r.key["question"] == first.key["question"]
            }),
            format!("{context}: facts, addresses or question differ"),
        )?;
        if records.len() == 4 {
            let mut cells = BTreeSet::new();
            for record in records {
                let selector = record.key["operative_clause_ordinal"]
                    .as_u64()
                    .ok_or("missing selector")?;
                let short = record.key["one_word_clause_ordinal"]
                    .as_u64()
                    .ok_or("missing short clause")?;
                require(
                    record.row.contrast == 0
                        && record.row.side == u8::from(selector != short)
                        && cells.insert((selector, short)),
                    format!("{context}: invalid or duplicate selector/clause cell"),
                )?;
            }
            require(
                cells == BTreeSet::from([(1, 1), (1, 2), (2, 1), (2, 2)]),
                format!("{context}: invalid crossed orientation"),
            )?;
            scoped += 1;
        } else {
            require(
                records.len() == 2
                    && first.row.contrast < 2
                    && records[0].row.side == 0
                    && records[1].row.side == 1
                    && records[1].row.contrast == first.row.contrast
                    && records
                        .iter()
                        .all(|r| r.key["one_word_clause_ordinal"].is_null()),
                format!("{context}: invalid direct pair"),
            )?;
            direct[usize::from(first.row.contrast)] += 1;
        }
    }
    require(scoped == 64 && direct == [64, 64], "held view counts drift")
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [mode, runs, out] if mode == "prepare" => prepare(Path::new(runs), Path::new(out)),
        _ => Err("usage: paired-release-held prepare RUNS NEW_PANEL".into()),
    }
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
