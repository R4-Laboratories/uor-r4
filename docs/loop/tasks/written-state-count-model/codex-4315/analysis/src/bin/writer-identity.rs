//! Audit the frozen writer's identity loss from saved observations and artifact words.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const VOCAB: usize = 4096;
const OBSERVATIONS: &str = "eba57d864bc5bd04a4933e8bdbb34bfed93c4a5898c57c16569b874e0cf0ddbb";
const ROOTS: &str = "875a0857dfef840d88a64c90a5777a6154a1674cde08a71429e2cca93fb7bb5a";
const CELLS: &str = "fae89184d30bacc4ef16af6339cb458e92891dd91a43cf237cdbc473deb60383";
const TRUNK: &str = "c69c748209abad27c3be2bff7ed438fa12df54bdd83b546d9dda9331112ba577";

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(message.into());
    }
    Ok(())
}

fn verified(path: &Path, digest: &str) -> Result<File> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    io::copy(&mut file, &mut hash)?;
    if format!("{:x}", hash.finalize()) != digest {
        return Err(format!("SHA-256 mismatch: {}", path.display()).into());
    }
    file.rewind()?;
    Ok(file)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[derive(Deserialize)]
struct Observation {
    cohort: String,
    group_id: String,
    assignment: usize,
    side: usize,
    prefix: Vec<usize>,
    target: usize,
    raw_target: usize,
    gold_pos: Option<usize>,
    context_start: usize,
    context_end: usize,
    parsed_start: Option<usize>,
    parsed_end: Option<usize>,
    candidates: Vec<Candidate>,
    arms: Arms,
}

#[derive(Deserialize)]
struct Candidate {
    pos: usize,
    id: usize,
    eligible: bool,
    geometry: u32,
}

#[derive(Deserialize)]
struct Arms {
    correct: WriterEvent,
}

#[derive(Deserialize)]
struct WriterEvent {
    source_pos: Option<usize>,
    gate_id: usize,
    emitted_id: usize,
    first_admitted_cell: Option<u32>,
}

#[derive(Serialize)]
struct SuppliedId {
    rows: usize,
    root_code: u32,
    cell: u32,
    emitted_id: usize,
    zero_distance_candidates: usize,
    rank_in_exact_code: usize,
}

#[derive(Default, Serialize)]
struct Census {
    rows: usize,
    target_cell_admitted_first: usize,
    emitted_cell_matches_supplied_cell: usize,
    emitted_exact_id: usize,
    wrong_emission: usize,
    wrong_same_code: usize,
    wrong_different_code: usize,
    wrong_lower_id: usize,
    wrong_eos: usize,
    emitted_smallest_id_at_supplied_code: usize,
    supplied_rank_beyond_top_eight: usize,
    zero_distance_rank_by_row: BTreeMap<usize, usize>,
    zero_distance_tie_size_by_row: BTreeMap<usize, usize>,
    zero_distance_rank_by_supplied_id: BTreeMap<usize, usize>,
    zero_distance_tie_size_by_supplied_id: BTreeMap<usize, usize>,
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [observations, base] = args.as_slice() else {
        return Err("usage: writer-identity OBSERVATIONS.jsonl BASE_FULL_DIR".into());
    };
    let base = Path::new(base);
    let observations = verified(Path::new(observations), OBSERVATIONS)?;
    let mut roots_bytes = Vec::new();
    verified(&base.join("roots"), ROOTS)?.read_to_end(&mut roots_bytes)?;
    let mut cells_bytes = Vec::new();
    verified(&base.join("cells"), CELLS)?.read_to_end(&mut cells_bytes)?;
    require(
        roots_bytes.len() == VOCAB * 4 && cells_bytes.len() == VOCAB * 4,
        "unexpected root or cell table length",
    )?;
    let mut trunk = verified(&base.join("trunk"), TRUNK)?;
    let mut trunk_roots = vec![0; VOCAB * 4];
    // This is the packed-root range used by d11_rotate_to_token.
    trunk.seek(SeekFrom::Start((4 * (70 + VOCAB)) as u64))?;
    trunk.read_exact(&mut trunk_roots)?;
    require(
        trunk_roots == roots_bytes,
        "writer and feature roots differ",
    )?;

    let roots = words(&roots_bytes);
    let cells = words(&cells_bytes);
    let mut by_code = BTreeMap::<u32, Vec<usize>>::new();
    let mut rank_at_code = Vec::with_capacity(VOCAB);
    for (id, (&root, &cell)) in roots.iter().zip(&cells).enumerate() {
        require(
            root <= u32::from(u16::MAX) && cell == (root & 15) * 16 + (root >> 12),
            "token cell disagrees with packed endpoint roots",
        )?;
        let matches = by_code.entry(root).or_default();
        matches.push(id);
        rank_at_code.push(matches.len());
    }

    let mut census = Census::default();
    let mut by_group = BTreeMap::<String, BTreeSet<(usize, usize)>>::new();
    let mut by_id = BTreeMap::<usize, SuppliedId>::new();
    let mut emitted_ids = BTreeSet::new();
    for (index, line) in BufReader::new(observations).lines().enumerate() {
        let row: Observation = serde_json::from_str(&line?)
            .map_err(|error| format!("observation {}: {error}", index + 1))?;
        if row.cohort != "controlled" {
            continue;
        }
        let gold = row.gold_pos.ok_or("controlled gold position missing")?;
        let event = row.arms.correct;
        let gate = event.gate_id;
        let emitted = event.emitted_id;
        require(
            gate < VOCAB
                && emitted < VOCAB
                && gate == row.target
                && gate == row.raw_target
                && event.source_pos == Some(gold)
                && row.prefix.get(gold) == Some(&gate)
                && row.parsed_start == Some(row.context_start)
                && row.parsed_end == Some(row.context_end)
                && (row.context_start..row.context_end).contains(&gold),
            "correct arm is not bound to its original factual token",
        )?;
        let code = roots[gate];
        require(
            row.candidates.iter().any(|candidate| {
                candidate.pos == gold
                    && candidate.id == gate
                    && candidate.geometry == code
                    && candidate.eligible
            }),
            "correct arm candidate provenance differs from the frozen roots",
        )?;
        require(
            row.assignment < 2
                && row.side < 2
                && by_group
                    .entry(row.group_id)
                    .or_default()
                    .insert((row.assignment, row.side)),
            "duplicate or invalid controlled group cell",
        )?;

        let aliases = &by_code[&code];
        let exact_rank = rank_at_code[gate];
        census.rows += 1;
        census.target_cell_admitted_first +=
            usize::from(event.first_admitted_cell == Some(cells[gate]));
        census.emitted_cell_matches_supplied_cell += usize::from(cells[gate] == cells[emitted]);
        census.emitted_smallest_id_at_supplied_code += usize::from(emitted == aliases[0]);
        census.supplied_rank_beyond_top_eight += usize::from(exact_rank > 8);
        *census
            .zero_distance_rank_by_row
            .entry(exact_rank)
            .or_default() += 1;
        *census
            .zero_distance_tie_size_by_row
            .entry(aliases.len())
            .or_default() += 1;
        if emitted == gate {
            census.emitted_exact_id += 1;
        } else {
            let same_code = code == roots[emitted];
            census.wrong_emission += 1;
            census.wrong_same_code += usize::from(same_code);
            census.wrong_different_code += usize::from(!same_code);
            census.wrong_lower_id += usize::from(emitted < gate);
            census.wrong_eos += usize::from(emitted == 1);
        }
        let stats = by_id.entry(gate).or_insert(SuppliedId {
            rows: 0,
            root_code: code,
            cell: cells[gate],
            emitted_id: emitted,
            zero_distance_candidates: aliases.len(),
            rank_in_exact_code: exact_rank,
        });
        require(
            stats.emitted_id == emitted,
            "supplied ID changes emission across rows",
        )?;
        stats.rows += 1;
        emitted_ids.insert(emitted);
    }
    require(
        census.rows == 800
            && by_group.len() == 200
            && by_group.values().all(|cells| cells.len() == 4),
        "incomplete controlled observation panel",
    )?;
    require(
        census.target_cell_admitted_first == 800
            && census.emitted_cell_matches_supplied_cell == 800
            && census.emitted_smallest_id_at_supplied_code == 800
            && census.emitted_exact_id == 14
            && census.wrong_emission == 786
            && census.wrong_same_code == 786
            && census.wrong_different_code == 0
            && census.wrong_lower_id == 786
            && census.wrong_eos == 74,
        "registered writer identity census did not reproduce",
    )?;
    for stats in by_id.values() {
        *census
            .zero_distance_rank_by_supplied_id
            .entry(stats.rank_in_exact_code)
            .or_default() += 1;
        *census
            .zero_distance_tie_size_by_supplied_id
            .entry(stats.zero_distance_candidates)
            .or_default() += 1;
    }
    let receipt = serde_json::json!({
        "schema": "uor-r4.occurrence-writer-identity/1",
        "inputs_sha256": {"observations": OBSERVATIONS, "roots": ROOTS, "cells": CELLS, "trunk": TRUNK},
        "writer_root_words_equal_feature_roots": true,
        "controlled_groups": by_group.len(),
        "vocabulary_size": VOCAB,
        "vocabulary_distinct_exact_codes": by_code.len(),
        "distinct_supplied_ids": by_id.len(),
        "distinct_emitted_ids": emitted_ids.len(),
        "census": census,
        "by_supplied_id": by_id,
    });
    serde_json::to_writer_pretty(io::stdout().lock(), &receipt)?;
    println!();
    Ok(())
}
