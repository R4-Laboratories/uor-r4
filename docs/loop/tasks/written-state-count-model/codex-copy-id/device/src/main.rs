use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

mod gpu;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const CONTEXT: usize = 192;
const VOCAB: u32 = 4096;
const NONE: u32 = u32::MAX;
const EOS: u32 = 1;
const COLON: u32 = 28;
const ARMS: usize = 8;
const ARM_NAMES: [&str; ARMS] = [
    "generate",
    "learned",
    "correct",
    "swapped",
    "same_id_wrong_occurrence",
    "invalid",
    "missing",
    "marker",
];

#[derive(Deserialize)]
struct Row {
    pair_id: String,
    prefix: Vec<u32>,
    target: u32,
    rival: u32,
    gold_pos: Option<u32>,
    rival_pos: Option<u32>,
    wrong_same_id_pos: Option<u32>,
}

impl Row {
    fn validate(&self) -> Result<()> {
        if self.pair_id.is_empty()
            || self.prefix.is_empty()
            || self.prefix.len() > CONTEXT
            || self.prefix.iter().any(|&id| id >= VOCAB)
            || self.target >= VOCAB
            || self.rival >= VOCAB
        {
            return Err(
                format!("invalid causal prefix or token domain for {}", self.pair_id).into(),
            );
        }
        for (position, token) in [
            (self.gold_pos, self.target),
            (self.rival_pos, self.rival),
            (self.wrong_same_id_pos, self.target),
        ] {
            if let Some(position) = position {
                if self.prefix.get(position as usize) != Some(&token) {
                    return Err(format!("position/ID witness mismatch for {}", self.pair_id).into());
                }
            }
        }
        if self.wrong_same_id_pos.is_some()
            && (self.gold_pos.is_none() || self.wrong_same_id_pos == self.gold_pos)
        {
            return Err(format!(
                "same-ID control is missing an independent address for {}",
                self.pair_id
            )
            .into());
        }
        Ok(())
    }

    fn self_feed(&self, emitted: u32) -> Self {
        let mut prefix = self.prefix[usize::from(self.prefix.len() == CONTEXT)..].to_vec();
        prefix.push(emitted);
        Self {
            pair_id: self.pair_id.clone(),
            prefix,
            target: self.target,
            rival: self.rival,
            gold_pos: None,
            rival_pos: None,
            wrong_same_id_pos: None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct WriterEvent {
    read_pos: Option<u32>,
    claim_pos: Option<u32>,
    gate_id: u32,
    source_id: Option<u32>,
    source_distance: Option<u32>,
    source_admitted: bool,
    minimum_distance: u32,
    first_admitted_cell: Option<u32>,
    native_proposal: u32,
    committed_id: u32,
    copied: bool,
    read_code: u32,
    recurrence_code: u32,
    causal_successor: Option<(u32, u32)>,
    read_vector: [i32; 32],
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Arm {
    first: WriterEvent,
    successor: Option<WriterEvent>,
    terminal: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Observation<'a> {
    pair_id: &'a str,
    prefix: &'a [u32],
    arms: std::collections::BTreeMap<&'static str, Arm>,
}

fn rows(path: &str) -> Result<Vec<Row>> {
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let row: Row = serde_json::from_str(&line?)?;
        row.validate()?;
        if !seen.insert((row.pair_id.clone(), row.prefix.clone())) {
            return Err(format!("duplicate pair ID and prefix {}", row.pair_id).into());
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("panel contains zero rows".into());
    }
    Ok(rows)
}

fn write_rows(path: &Path, observations: &[Observation<'_>]) -> Result<()> {
    let f = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut file = BufWriter::new(f);
    for row in observations {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    Ok(())
}

fn score(files: &gpu::Files<'_>, path: &str, output: &str) -> Result<()> {
    let rows = rows(path)?;
    let output = Path::new(output);
    fs::create_dir(output)?;
    let start = Instant::now();
    let forward = gpu::replay(files, rows.iter())?;
    write_rows(&output.join("observations.jsonl"), &forward)?;
    let reverse = gpu::replay(files, rows.iter().rev())?;
    write_rows(&output.join("reverse.jsonl"), &reverse)?;
    for (first, second) in forward.iter().zip(reverse.iter().rev()) {
        if first != second {
            return Err(format!("reverse replay mismatch in {}", first.pair_id).into());
        }
    }
    let receipt = serde_json::json!({
        "schema": "uor-r4.copy-id-device/1",
        "rows": forward.len(),
        "arms_per_row": ARMS,
        "raw_steps": 2,
        "reverse_reloaded": true,
        "reverse_replay": "identical",
        "elapsed_ms": start.elapsed().as_millis(),
        "input_sha256": gpu::digest(path)?,
        "copy_cubin_sha256": gpu::digest(files.copy)?,
    });
    fs::write(
        output.join("replay.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["check-rows", panel] => {
            println!("{}", rows(panel)?.len());
            Ok(())
        }
        ["score", copy, occurrence, baseline, base, classes, weights, marker, panel, output] => {
            score(
                &gpu::Files {
                    copy,
                    occurrence,
                    baseline,
                    base,
                    classes,
                    weights,
                    marker,
                },
                panel,
                output,
            )
        }
        _ => Err(concat!(
            "usage: copy-id-device check-rows PANEL.jsonl\n",
            "       copy-id-device score COPY.cubin OCCURRENCE.cubin BASELINE.cubin ",
            "BASE_FULL_DIR CLASSES.u8 WEIGHTS.i16le MARKER.tsv PANEL.jsonl NEW_OUTPUT_DIR"
        )
        .into()),
    }
}
