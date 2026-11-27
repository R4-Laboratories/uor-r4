#![forbid(unsafe_code)]

#[allow(dead_code)]
#[path = "../../../codex-4324/panel/src/archive.rs"]
mod archive;
mod design;
mod generate;
#[allow(dead_code)]
#[path = "../../../codex-4324/panel/src/design.rs"]
mod old_design;
mod rehearsal;

use archive::{normalized, template, Archive};
use design::{Design, HELD, TRAIN};
use generate::{Census, Generator, Record, Wire};
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
    let root = runs.join(OLD_FACTORIAL).join("panel-v1");
    require(
        hash(&root.join("SHA256SUMS"))? == OLD_FACTORIAL_SHA256,
        "factorial archive manifest drift",
    )?;
    verify_manifest(&root)?;
    archive.roots.insert(OLD_FACTORIAL, OLD_FACTORIAL_SHA256);
    for Record { row, key } in read_split(&root.join("fresh"))? {
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
    archive
        .templates
        .extend([old_design::FACT, old_design::QUESTION, old_design::FORMAT].map(template));
    for family in old_design::SCOPED {
        archive.templates.extend(family.clauses.map(template));
        archive.templates.extend(family.selectors.map(template));
    }
    Ok(archive)
}

fn split_templates(design: &Design, archive: &Archive) -> Result<BTreeSet<String>> {
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
    for text in &raw {
        archive.check_template(text)?;
    }
    Ok(raw)
}

fn split_file(
    dir: &Path,
    records: &[Record],
    census: Census,
    design: &Design,
    templates: &BTreeSet<String>,
    archive: &Archive,
    sources: Value,
) -> Result<()> {
    fs::create_dir(dir)?;
    let mut rows = writer(&dir.join("rows.jsonl"))?;
    let mut keys = writer(&dir.join("keys.jsonl"))?;
    for Record { row, key } in records {
        jsonl(&mut rows, row)?;
        jsonl(&mut keys, key)?;
    }
    rows.flush()?;
    keys.flush()?;
    let mut receipt = writer(&dir.join("receipt.json"))?;
    serde_json::to_writer_pretty(
        &mut receipt,
        &json!({
            "schema": "codex-4325.joint-release-panel/1",
            "split": design.split,
            "authorship": {
                "fresh_scoped_and_train_direct": "Codex panel agent-authored templates and deterministic instances",
                "held_direct": "Separate Codex product replay agent-authored count directives; panel agent adapted the suggested fact frames into matched four-entry lists",
                "rehearsal": sources,
            },
            "tokenizer_sha256": TOKENIZER_SHA256,
            "cap": CAP,
            "archives": archive.roots,
            "archive_rows_checked": archive.rows,
            "archive_template_signatures_checked": archive.templates.len(),
            "fresh_template_strings_checked": templates.len(),
            "fresh_exact_overlap": {
                "prior_or_other_split_source_token_ids": 0,
                "prior_or_other_split_template_signatures": 0,
                "prior_or_other_split_prompts": 0,
                "prior_or_other_split_contexts": 0,
                "prior_or_other_split_encoded_prompts": 0,
            },
            "total_rows": records.len(),
            "total_pairs": records.len() / 2,
            "total_post_b_pairs": records.iter().filter(|r| r.row.contrast == 1).count() / 2,
            "fresh": census,
        }),
    )?;
    writeln!(receipt)?;
    receipt.flush()?;
    write_manifest(dir, &["rows.jsonl", "keys.jsonl", "receipt.json"])
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
    let train_templates = split_templates(&TRAIN, &archive)?;
    let held_templates = split_templates(&HELD, &archive)?;
    let train_signatures: BTreeSet<_> = train_templates.iter().map(|text| template(text)).collect();
    require(
        held_templates
            .iter()
            .all(|text| !train_signatures.contains(&template(text))),
        "train/held template overlap",
    )?;
    let templates: BTreeSet<_> = train_templates.union(&held_templates).cloned().collect();
    let mut reserved = archive.labels.clone();
    for text in &templates {
        reserved.extend(vocab.encode(text));
        reserved.extend(vocab.encode(&format!(" {text}")));
    }
    let mut generator = Generator::new(&wire, &archive);
    let (mut train, train_census) = generator.split(&TRAIN, &mut reserved)?;
    let (held, held_census) = generator.split(&HELD, &mut reserved)?;
    let (rehearsal, sources) = rehearsal::read(runs, &archive)?;
    train.extend(rehearsal);
    fs::create_dir(root)?;
    split_file(
        &root.join("train"),
        &train,
        train_census,
        &TRAIN,
        &train_templates,
        &archive,
        sources,
    )?;
    split_file(
        &root.join("held"),
        &held,
        held_census,
        &HELD,
        &held_templates,
        &archive,
        Value::Null,
    )?;
    write_manifest(root, &["train/SHA256SUMS", "held/SHA256SUMS"])
}

fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [mode, runs, out] if mode == "prepare" => prepare(Path::new(runs), Path::new(out)),
        _ => Err("usage: joint-release-panel prepare RUNS NEW_PANEL".into()),
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
