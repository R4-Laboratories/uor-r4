use super::*;
use d11::{
    device::{DeviceScorer, ROW_PROBE_WORDS},
    kernel::{pack_rows_with_targets, RowScore},
};

mod fixture;

fn wide(lo: u32, hi: u32) -> u64 {
    u64::from(lo) | (u64::from(hi) << 32)
}

fn token(id: u32, words: &[u32; 23]) -> Result<TokenProbe> {
    let &[lo, hi, c_q, c_qr, c_r, c_qc, c_rr, pointer_bonus, pool, gate_rank, gate_full_rank, mass_ties, eligible_mass_ties, prior_count, prior_rank, prior_ties, writer_cell, writer_distance, writer_admitted, writer_rank, writer_full_rank, writer_distance_ties, writer_admitted_ties] =
        words;
    let gate_score_terms = ScoreTerms {
        c_q,
        c_qr,
        c_r,
        c_qc,
        c_rr,
        pointer_bonus,
    };
    let mass = wide(lo, hi);
    if mass != gate_score_terms.mass()
        || pool > 1
        || writer_admitted > 1
        || (writer_rank != 0) != (writer_admitted == 1)
        || gate_full_rank == 0
        || writer_full_rank == 0
        || prior_rank == 0
        || mass_ties == 0
        || prior_ties == 0
        || writer_distance_ties == 0
    {
        return Err(format!("invalid CUDA rank record for ID {id}").into());
    }
    Ok(TokenProbe {
        id,
        mass,
        gate_score_terms,
        pool_member: pool == 1,
        positive_mass: mass > 0,
        eligible_positive: pool == 1 && mass > 0,
        gate_rank: (gate_rank > 0).then_some(gate_rank),
        gate_full_rank,
        mass_ties,
        eligible_mass_ties,
        prior_count,
        prior_rank,
        prior_ties,
        writer_cell,
        writer_distance,
        writer_admitted: writer_admitted == 1,
        writer_rank: (writer_rank > 0).then_some(writer_rank),
        writer_full_rank,
        writer_distance_ties,
        writer_admitted_ties,
    })
}

fn decode(
    row: Input,
    ids: [u32; 3],
    tables: &ArtifactTables,
    score: RowScore,
    words: &[u32; ROW_PROBE_WORDS],
) -> Result<Observation> {
    let width = tables.shape[0] as u32;
    if score.status != 0
        || score.top8.first() != Some(&score.prediction)
        || score.prediction >= width
        || words[6] >= width
        || words[6] != words[7]
    {
        return Err(format!(
            "invalid GPU state at {} side {} status {}",
            row.pair_id, row.side, score.status
        )
        .into());
    }
    let [gold_id, rival, space] = ids;
    let probes = Probes {
        gold: token(gold_id, words[1013..1036].try_into()?)?,
        rival: token(rival, words[1036..1059].try_into()?)?,
        space: token(space, words[1059..1082].try_into()?)?,
    };
    if probes.gold.mass != wide(words[10], words[11])
        || u32::from(probes.gold.pool_member) != words[12]
    {
        return Err("rank tap disagrees with native gate target observation".into());
    }
    let gate_mass = wide(words[8], words[9]);
    for probe in [&probes.gold, &probes.rival, &probes.space] {
        if (probe.gate_rank == Some(1)) != (probe.id == words[6])
            || (probe.writer_rank == Some(1)) != (probe.id == score.prediction)
            || probe
                .gate_rank
                .is_some_and(|rank| rank != probe.gate_full_rank)
            || probe
                .writer_rank
                .is_some_and(|rank| rank != probe.writer_full_rank)
            || probe.prior_rank > width
            || probe.gate_full_rank > width
            || probe.writer_full_rank > width
        {
            return Err(format!(
                "rank tap disagrees with native decision for ID {}",
                probe.id
            )
            .into());
        }
    }
    let source_position = usize::try_from(words[3] as i32).ok();
    let source_token = source_position
        .map(|source| {
            row.prefix
                .get(source)
                .copied()
                .ok_or("source exceeds original prefix")
        })
        .transpose()?;
    if words[0] != words[3] {
        return Err("native gate source was changed".into());
    }
    let mut routes = Vec::new();
    let stages = tables.shape[4] as usize;
    for block in 0..4 {
        for stage in 0..stages {
            let at = 113 + (block * 8 + stage) * 28;
            let route = &words[at..at + 28];
            let level_gate = tables.gates[block * stages + stage];
            if level_gate == 0 {
                if route[1] != 0 {
                    return Err("disabled route reported contributing nodes".into());
                }
                continue;
            }
            let n = route[1] as usize;
            if n > 8 || (n > 0 && (route[0] as i32 != level_gate + 1 || route[2] >= 16)) {
                return Err("invalid GPU route trace".into());
            }
            routes.push(Route {
                block,
                stage,
                level_gate,
                selected_root: (n > 0).then_some(route[2]),
                node_indices: route[4..4 + n].to_vec(),
                frame_indices: route[12..12 + n].to_vec(),
                carried_roots: route[20..20 + n].to_vec(),
            });
        }
    }
    let signed = |a: u32| a as i32;
    let prefix_len = row.prefix.len();
    let prefix_sha256 = prefix_sha256(&row.prefix);
    let query_token = *row.prefix.last().ok_or("prefix missing query")?;
    Ok(Observation {
        input: row,
        rival,
        space,
        artifact_sha256: tables.artifact_sha256.clone(),
        prefix_len,
        prefix_sha256,
        query_token,
        read_vector: <[u32; 32]>::try_from(&words[13..45])?.map(signed),
        read_code: words[4],
        recurrence_code: words[5],
        recurrent_vector: <[u32; 32]>::try_from(&words[45..77])?.map(signed),
        recurrent_frames: <[u32; 4]>::try_from(&words[77..81])?.map(signed),
        recurrent_sum: <[u32; 32]>::try_from(&words[81..113])?.map(signed),
        writer_vector: score.read_vector,
        admitted_cells: score.admitted_cells,
        routes,
        source_position,
        source_token,
        gate_selected: words[6],
        gate_mass,
        emitted: score.prediction,
        top8: score.top8,
        prior_selected: words[1009],
        active_candidates: words[1010],
        positive_candidates: words[1011],
        writer_candidates: words[1012],
        probes,
        duplicate_replay: true,
    })
}

fn scorer(tables: &ArtifactTables) -> Result<DeviceScorer> {
    if std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() != Ok("1") {
        return Err("set CUDA_DISABLE_PTX_JIT=1 for ahead-of-time cubin loading".into());
    }
    Ok(DeviceScorer::new(0, tables)?)
}

fn score(
    tables: &ArtifactTables,
    rows: Vec<Input>,
    ids: Vec<[u32; 3]>,
) -> Result<Vec<Observation>> {
    let device = scorer(tables)?;
    let prefixes: Vec<_> = rows.iter().map(|r| r.prefix.as_slice()).collect();
    let targets: Vec<_> = rows.iter().map(|r| r.target).collect();
    let (first, words) =
        device.score_token_probes(&pack_rows_with_targets(&prefixes, &targets)?, &ids)?;
    let reversed_rows = pack_rows_with_targets(
        &prefixes.iter().rev().copied().collect::<Vec<_>>(),
        &targets.iter().rev().copied().collect::<Vec<_>>(),
    )?;
    let reversed_ids: Vec<_> = ids.iter().rev().copied().collect();
    let (second, repeated) = device.score_token_probes(&reversed_rows, &reversed_ids)?;
    if rows.len() != first.len()
        || rows.len() != words.len()
        || !first.iter().eq(second.iter().rev())
        || !words.iter().eq(repeated.iter().rev())
    {
        return Err("independent reverse-order replay changed CUDA row words".into());
    }
    rows.into_iter()
        .zip(ids)
        .zip(first.into_iter().zip(&words))
        .map(|((row, ids), (score, words))| decode(row, ids, tables, score, words))
        .collect()
}

pub(super) fn run(artifact: &str, input: &str, output: &str) -> Result<()> {
    let (tables, _) = open_artifact(artifact)?;
    let rows: Vec<Input> = read_rows(input)?;
    let ids = validate_pairs(&rows)?
        .into_iter()
        .zip(&rows)
        .map(|(rival, row)| [row.target, rival, SPACE])
        .collect();
    write_rows(output, &score(&tables, rows, ids)?)
}

pub(super) fn finite() -> Result<()> {
    fixture::run()
}

pub(super) fn info() -> Result<()> {
    use d11::device::{cubin_sha256, ptx_sha256, IMAGE_TARGET, RESOURCES};
    println!(
        "{}",
        serde_json::json!({
            "cubin_sha256": cubin_sha256(),
            "ptx_sha256": ptx_sha256(),
            "image_target": IMAGE_TARGET,
            "resources": RESOURCES,
        })
    );
    Ok(())
}
