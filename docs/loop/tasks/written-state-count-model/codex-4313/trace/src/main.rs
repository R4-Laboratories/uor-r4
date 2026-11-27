#![forbid(unsafe_code)]

use d11::artifact::ArtifactTables;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
};
use uor_r4_discrete::corpus::sha256;
use uor_r4_integer::discrete_d11::IntegerDiscreteModel;

#[cfg(feature = "cuda")]
mod gpu;
mod parity;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const VOCAB: u32 = 4096;
const CONTEXT: usize = 192;
const SPACE: u32 = 223;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Input {
    pair_id: String,
    side: usize,
    cohort: String,
    prefix: Vec<u32>,
    target: u32,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ScoreTerms {
    c_q: u32,
    c_qr: u32,
    c_r: u32,
    c_qc: u32,
    c_rr: u32,
    pointer_bonus: u32,
}

impl ScoreTerms {
    fn mass(&self) -> u64 {
        let wide = u64::from;
        wide(self.c_q)
            + (wide(self.c_qr) << 6)
            + (wide(self.c_r) << 1)
            + (wide(self.c_qc) << 5)
            + (wide(self.c_rr) << 3)
            + wide(self.pointer_bonus)
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct TokenProbe {
    id: u32,
    mass: u64,
    gate_score_terms: ScoreTerms,
    pool_member: bool,
    positive_mass: bool,
    eligible_positive: bool,
    gate_rank: Option<u32>,
    gate_full_rank: u32,
    mass_ties: u32,
    eligible_mass_ties: u32,
    prior_count: u32,
    prior_rank: u32,
    prior_ties: u32,
    writer_cell: u32,
    writer_distance: u32,
    writer_admitted: bool,
    writer_rank: Option<u32>,
    writer_full_rank: u32,
    writer_distance_ties: u32,
    writer_admitted_ties: u32,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct Probes {
    gold: TokenProbe,
    rival: TokenProbe,
    space: TokenProbe,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct Route {
    block: usize,
    stage: usize,
    level_gate: i32,
    selected_root: Option<u32>,
    node_indices: Vec<u32>,
    frame_indices: Vec<u32>,
    carried_roots: Vec<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Observation {
    #[serde(flatten)]
    input: Input,
    rival: u32,
    space: u32,
    artifact_sha256: String,
    prefix_len: usize,
    prefix_sha256: String,
    query_token: u32,
    read_vector: [i32; 32],
    read_code: u32,
    recurrence_code: u32,
    recurrent_frames: [i32; 4],
    recurrent_vector: [i32; 32],
    recurrent_sum: [i32; 32],
    writer_vector: Vec<i32>,
    admitted_cells: Vec<u32>,
    routes: Vec<Route>,
    source_position: Option<usize>,
    source_token: Option<u32>,
    gate_selected: u32,
    gate_mass: u64,
    emitted: u32,
    top8: Vec<u32>,
    prior_selected: u32,
    active_candidates: u32,
    positive_candidates: u32,
    writer_candidates: u32,
    probes: Probes,
    duplicate_replay: bool,
}

fn prefix_sha256(prefix: &[u32]) -> String {
    sha256(
        &prefix
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn open_artifact(path: &str) -> Result<(ArtifactTables, IntegerDiscreteModel)> {
    let (tables, client) = ArtifactTables::from_bytes(&fs::read(path)?)?;
    if !client.is_trunk_d11()
        || client.shape().blocks != 4
        || client.shape().context as usize != CONTEXT
        || client.shape().vocabulary != VOCAB
        || !matches!(tables.shape[5], 1 | 2)
    {
        return Err("expected the registered four-block Flock/Relation D11 trunk".into());
    }
    Ok((tables, client))
}

fn read_rows<T: serde::de::DeserializeOwned>(path: &str) -> Result<Vec<T>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line?;
            serde_json::from_str(&line).map_err(|error| format!("{path}:{} {error}", i + 1).into())
        })
        .collect()
}

fn write_rows<T: Serialize>(path: &str, rows: &[T]) -> Result<()> {
    let mut file = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    file.flush()?;
    file.get_ref().sync_all()?;
    Ok(())
}

fn validate_pairs(rows: &[Input]) -> Result<Vec<u32>> {
    let mut pairs = BTreeMap::new();
    for row in rows {
        if row.pair_id.is_empty()
            || row.cohort.is_empty()
            || row.side > 1
            || row.prefix.is_empty()
            || row.prefix.len() > CONTEXT
            || row.target >= VOCAB
            || row.target == SPACE
            || row.prefix.iter().any(|&id| id >= VOCAB)
        {
            return Err(format!("invalid row {} side {}", row.pair_id, row.side).into());
        }
        let sides = pairs.entry(&row.pair_id).or_insert([None, None]);
        if sides[row.side].replace(row).is_some() {
            return Err(format!("duplicate input {} side {}", row.pair_id, row.side).into());
        }
    }
    if pairs.is_empty() {
        return Err("empty pair set".into());
    }
    for (id, sides) in &pairs {
        let [Some(a), Some(b)] = *sides else {
            return Err(format!("incomplete pair {id}").into());
        };
        if a.target == b.target
            || a.cohort != b.cohort
            || a.prefix == b.prefix
            || a.prefix.last() != b.prefix.last()
        {
            return Err(format!("invalid swap contrast {id}").into());
        }
    }
    rows.iter()
        .map(|row| {
            pairs[&row.pair_id][1 - row.side]
                .map(|r| r.target)
                .ok_or_else(|| format!("missing rival for {}", row.pair_id).into())
        })
        .collect()
}

fn run(args: &[String]) -> Result<()> {
    match args {
        [op, artifact, input, output] if op == "gpu" => {
            #[cfg(feature = "cuda")]
            return gpu::run(artifact, input, output);
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (artifact, input, output);
                Err("gpu requires the cuda feature".into())
            }
        }
        [op, artifact, observed] if op == "client-parity" => {
            let (tables, client) = open_artifact(artifact)?;
            parity::run(&tables, &client, &read_rows(observed)?)
        }
        #[cfg(feature = "cuda")]
        [op] if op == "finite" => gpu::finite(),
        #[cfg(feature = "cuda")]
        [op] if op == "info" => gpu::info(),
        _ => Err("usage: question-swap-trace gpu ARTIFACT INPUT OUTPUT | client-parity ARTIFACT GPU_JSONL | finite | info".into()),
    }
}

fn main() -> std::process::ExitCode {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
