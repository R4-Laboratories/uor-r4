use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use d11::artifact::ArtifactTables;
use serde::{Deserialize, Serialize};
use uor_r4_integer::discrete_d11::native_opening as contract;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const VOCAB: u32 = 4096;
const STRIDE: usize = contract::CONTEXT as usize + 3;
const BASE_SHA: &str = "47214eda18119b1e555cec858f9356c93246a23c9f34686962bc19ba05716049";
#[cfg(feature = "cuda")]
mod gpu;

#[derive(Deserialize, Serialize, Clone)]
struct Row {
    prefix: Vec<u32>,
    target: u32,
    #[serde(default)]
    raw_target: Option<u32>,
    #[serde(default)]
    rival: Option<u32>,
    #[serde(default)]
    pair_id: String,
    #[serde(default)]
    group_id: String,
    #[serde(default)]
    side: u32,
    #[serde(default)]
    assignment: u32,
    #[serde(default)]
    cohort: String,
}

fn new_file(path: impl AsRef<Path>) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn write_words(path: &Path, words: impl Iterator<Item = u32>) -> Result<()> {
    let mut out = new_file(path)?;
    for word in words {
        out.write_all(&word.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

fn extract(path: &str, out: &str) -> Result<()> {
    let (base, _client) = ArtifactTables::from_bytes(&fs::read(path)?)?;
    if base.artifact_sha256 != BASE_SHA
        || base.shape[0] != VOCAB as i32
        || base.shape[2] != contract::CONTEXT as i32
    {
        return Err("wrong base model".into());
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
        ("trunk", &base.trunk[..70 + 2 * VOCAB as usize]),
    ] {
        write_words(&dir.join(name), values.iter().copied())?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn read_rows(path: &str, contrasts: bool) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let row: Row = serde_json::from_str(&line?)?;
        if row.prefix.is_empty()
            || row.prefix.len() > contract::CONTEXT as usize
            || row.prefix.last() != Some(&contract::COLON)
            || row.prefix.iter().any(|&id| id >= VOCAB)
            || row.target >= VOCAB
            || row.rival.is_some() != contrasts
            || row.rival.is_some_and(|id| id >= VOCAB)
        {
            return Err(format!("invalid opening row {index} in {path}").into());
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("empty input".into());
    }
    Ok(rows)
}

#[cfg(feature = "cuda")]
fn config() -> Vec<u32> {
    let mut v = contract::ASSISTANT_MARKER.to_vec();
    v.extend(contract::QUESTION_MASKS);
    v.extend([
        contract::MIN_SUPPORT,
        contract::MARKER_BUCKET,
        contract::LEVEL_SHIFT,
        contract::COLON,
        contract::SPACE,
    ]);
    v
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["extract",base,out]=>extract(base,out),
        #[cfg(feature="cuda")]
        ["fit",image,base,donor,pairs,out]=>gpu::fit(image,base,donor,pairs,out),
        #[cfg(feature="cuda")]
        ["score",image,base,held,weights,out]=>gpu::score(image,base,held,weights,out),
        #[cfg(feature="cuda")]
        ["fixture",image,base,out]=>gpu::fixture(image,base,out),
        _=>Err("usage: extract D11 NEWDIR | fit CUBIN BASEDIR DONOR.jsonl TRAIN.jsonl NEWDIR | score CUBIN BASEDIR PANEL.jsonl WEIGHTS_DIR NEWDIR | fixture CUBIN BASEDIR NEWDIR".into()),
    }
}
