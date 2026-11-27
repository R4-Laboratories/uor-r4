use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use d11::artifact::ArtifactTables;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

mod gpu;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const BASE_SHA: &str = "47214eda18119b1e555cec858f9356c93246a23c9f34686962bc19ba05716049";
const VOCAB: u32 = 4096;
const CONTEXT: usize = 192;
const NONE: u32 = u32::MAX;
const COLON: u32 = 28;
const SPACE: u32 = 223;
const FEATURES: usize = 16;
const STATE_WORDS: usize = FEATURES * 32 + 1;
const STRIDE: usize = CONTEXT + 8;
const DONOR_ROWS: usize = 118_915;
const TRAIN_ROWS: usize = 800;
const BATCHES: usize = 20;

#[derive(Clone, Deserialize)]
struct Donor {
    prefix: Vec<u32>,
    target: u32,
    genuine_assistant: bool,
}

#[derive(Clone, Deserialize, Serialize)]
struct Row {
    prefix: Vec<u32>,
    group_id: String,
    pair_id: String,
    cohort: String,
    assignment: u32,
    side: u32,
    target: u32,
    rival: u32,
    #[serde(default)]
    raw_target: Option<u32>,
    #[serde(default)]
    raw_rival: Option<u32>,
    gold_pos: Option<u32>,
    rival_pos: Option<u32>,
    context_start: usize,
    context_end: usize,
    source_status: String,
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn write_words(path: &Path, words: impl Iterator<Item = u32>) -> Result<()> {
    let mut out = create(path)?;
    for word in words {
        out.write_all(&word.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

fn extract(path: &str, out: &str) -> Result<()> {
    let (base, client) = ArtifactTables::from_bytes(&fs::read(path)?)?;
    if base.artifact_sha256 != BASE_SHA || base.shape[0..3] != [4096, 4, 192] {
        return Err("wrong frozen D11 model".into());
    }
    let dir = Path::new(out);
    fs::create_dir(dir)?;
    for (name, values) in [
        ("shape", base.shape.as_slice()),
        ("table", &base.table),
        ("gates", &base.gates),
        ("plans", &base.plans),
    ] {
        write_words(&dir.join(name), values.iter().map(|v| *v as u32))?;
    }
    for (name, values) in [
        ("cells", base.cells.as_slice()),
        ("routes", &base.routes),
        ("roots", &client.packed_codebook_roots()),
        ("trunk", &base.trunk),
    ] {
        write_words(&dir.join(name), values.iter().copied())?;
    }
    let mut identity = create(&dir.join("base.sha256"))?;
    writeln!(identity, "{}", base.artifact_sha256)?;
    identity.flush()?;
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(&line?).map_err(|e| format!("{path}:{}: {e}", index + 1).into())
        })
        .collect()
}

fn valid_prefix(prefix: &[u32], target: u32) -> bool {
    !prefix.is_empty()
        && prefix.len() <= CONTEXT
        && prefix.last() == Some(&COLON)
        && prefix.iter().all(|&id| id < VOCAB)
        && target < VOCAB
}

fn read_donor(path: &str) -> Result<Vec<Donor>> {
    let rows = read_json::<Donor>(path)?;
    if rows.len() != DONOR_ROWS {
        return Err(format!("incorrect canonical donor census: {}", rows.len()).into());
    }
    for (index, row) in rows.iter().enumerate() {
        if !valid_prefix(&row.prefix, row.target) {
            return Err(format!("invalid donor row {index}").into());
        }
    }
    Ok(rows)
}

fn read_panel(path: &str) -> Result<Vec<Row>> {
    let rows = read_json::<Row>(path)?;
    if rows.is_empty() {
        return Err("empty panel".into());
    }
    for (index, row) in rows.iter().enumerate() {
        if !valid_prefix(&row.prefix, row.target)
            || row.rival >= VOCAB
            || row.raw_target.is_some_and(|id| id >= VOCAB)
            || row.raw_rival.is_some_and(|id| id >= VOCAB)
            || row.context_start >= row.context_end
            || row.context_end > row.prefix.len()
            || !row.position_is(row.gold_pos, row.target)
            || !row.position_is(row.rival_pos, row.rival)
        {
            return Err(format!("invalid panel row {index}").into());
        }
    }
    Ok(rows)
}

impl Row {
    fn position_is(&self, pos: Option<u32>, id: u32) -> bool {
        pos.is_none_or(|position| {
            let pos = position as usize;
            (self.context_start..self.context_end).contains(&pos) && self.prefix[pos] == id
        })
    }

    fn row_words(&self, fallback: u32) -> [u32; STRIDE] {
        let mut row = [0; STRIDE];
        row[..6].copy_from_slice(&[
            self.prefix.len() as u32,
            self.target,
            self.rival,
            self.gold_pos.unwrap_or(NONE),
            self.rival_pos.unwrap_or(NONE),
            fallback,
        ]);
        row[6] = u32::from(
            self.prefix.last() == Some(&COLON)
                && self.raw_target == Some(SPACE)
                && self.target != SPACE,
        );
        row[8..8 + self.prefix.len()].copy_from_slice(&self.prefix);
        row
    }
}

impl Donor {
    fn row_words(&self) -> [u32; STRIDE] {
        let mut row = [0; STRIDE];
        row[0] = self.prefix.len() as u32;
        row[1] = self.target;
        row[7] = u32::from(self.genuine_assistant);
        row[8..8 + self.prefix.len()].copy_from_slice(&self.prefix);
        row
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut out = create(path)?;
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn write_rows<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    let mut out = create(path)?;
    for row in rows {
        serde_json::to_writer(&mut out, row)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["extract", base, out] => extract(base, out),
        ["fit", image, base, classes, donor, contrasts, absent, out] =>
            gpu::fit(image, base, classes, donor, contrasts, absent, out),
        ["score", image, base, classes, held, marker, weights, out] =>
            gpu::score(image, base, classes, held, marker, weights, out),
        ["fixture", image, base, classes, out] => gpu::fixture(image, base, classes, out),
        _ => Err(concat!(
            "usage:\n  extract ORIGINAL_D11 NEW_BASE_DIR\n",
            "  fit CUBIN BASE_DIR TOKEN_CLASSES DONOR.jsonl TRAIN.jsonl TRAIN_ABSENT.jsonl NEW_WEIGHTS_DIR\n",
            "  score CUBIN BASE_DIR TOKEN_CLASSES HELD.jsonl MARKER.jsonl WEIGHTS_DIR NEW_HELD_DIR\n",
            "  fixture CUBIN BASE_DIR TOKEN_CLASSES NEW_FIXTURE_DIR",
        ).into()),
    }
}
