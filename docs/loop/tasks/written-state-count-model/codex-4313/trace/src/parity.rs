use super::*;
use uor_r4_discrete::{
    normalization_mechanism::{
        normalize_block_with_frame, orbit_root_coords, transform_block_by_frame,
    },
    position_mechanism::{compose_frames, invert_frame},
};
use uor_r4_integer::discrete_d11::{
    DISCRETE_ACTIVE_TRACE_HEADER_WORDS, DISCRETE_ACTIVE_TRACE_LEVEL_WORDS,
};

pub(super) fn run(
    tables: &ArtifactTables,
    client: &IntegerDiscreteModel,
    rows: &[Observation],
) -> Result<()> {
    let inputs: Vec<_> = rows.iter().map(|row| row.input.clone()).collect();
    let rivals = validate_pairs(&inputs)?;
    for (row, rival) in rows.iter().zip(rivals) {
        if rival != row.rival || row.space != SPACE {
            return Err(format!("incorrect probe identity for {}", row.input.pair_id).into());
        }
        check(tables, client, row)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "artifact_sha256": tables.artifact_sha256,
            "rows": rows.len(),
            "pairs": rows.len() / 2,
            "matched": ["prefix", "pre_gate_read_vector", "read_code", "recurrent_frames",
                "recurrent_sum", "recurrent_vector", "recurrence_code", "active_routes",
                "successor_address_and_id", "gate_selected", "writer_vector", "admitted_cells",
                "writer_top8", "writer_emitted", "writer_full_and_admitted_ranks", "prior_ranks"]
        })
    );
    Ok(())
}

pub(super) fn check(
    tables: &ArtifactTables,
    client: &IntegerDiscreteModel,
    row: &Observation,
) -> Result<()> {
    let prefix = &row.input.prefix;
    let failure = |what: &str| {
        format!(
            "{} side {} portable parity mismatch in {what}",
            row.input.pair_id, row.input.side
        )
    };
    if row.artifact_sha256 != client.artifact_sha256()
        || row.prefix_len != prefix.len()
        || row.prefix_sha256 != prefix_sha256(prefix)
        || prefix.last() != Some(&row.query_token)
        || !row.duplicate_replay
    {
        return Err(failure("prefix, artifact or replay identity").into());
    }
    let trace = client.evaluate_prefix_traced(prefix)?;
    let pointer = trace.trunk_pointer.ok_or("client has no trunk gate")?;
    let eval = trace.evaluation;
    let successor = row.source_position.zip(row.source_token);
    if u32::from(pointer.read_code) != row.read_code
        || u32::from(pointer.recurrence_code) != row.recurrence_code
        || u32::from(pointer.selected_token) != row.gate_selected
        || pointer.successor.map(|(p, t)| (p, u32::from(t))) != successor
        || (row.source_position.is_some() != row.source_token.is_some())
    {
        return Err(failure("native gate codes/source/selected ID").into());
    }
    if eval.read_vector != row.writer_vector
        || eval.admitted_cells != row.admitted_cells
        || eval.top8 != row.top8
        || eval.prediction != row.emitted
    {
        return Err(failure("native writer vector, cells, ranking or emission").into());
    }

    let stage_words = DISCRETE_ACTIVE_TRACE_LEVEL_WORDS;
    let header_size = DISCRETE_ACTIVE_TRACE_HEADER_WORDS;
    let stages = tables.shape[4] as usize;
    let block_words = header_size + stages * stage_words;
    if trace.trace_words.len() != 4 * block_words {
        return Err(failure("route trace width").into());
    }
    for block in 0..4 {
        let base = block * block_words;
        let root = trace.trace_words[base + 1] as u8;
        if root != ((pointer.read_code >> (4 * block)) & 15) as u8
            || orbit_root_coords(root)? != row.read_vector[block * 8..(block + 1) * 8]
        {
            return Err(failure("pre-gate read vector").into());
        }
    }
    let mut expected_routes = Vec::new();
    for block in 0..4 {
        for stage in 0..stages {
            let gate = tables.gates[block * stages + stage];
            if gate == 0 {
                continue;
            }
            let start = block * block_words + header_size + stage * stage_words;
            let words = &trace.trace_words[start..start + stage_words];
            let n = usize::try_from(words[1])?;
            if n > 8 || words[0] as i32 != gate + 1 {
                return Err(failure("active route width or gate").into());
            }
            expected_routes.push(Route {
                block,
                stage,
                level_gate: gate,
                selected_root: (n > 0).then_some(words[2]),
                node_indices: words[4..4 + n].to_vec(),
                frame_indices: words[12..12 + n].to_vec(),
                carried_roots: words[20..20 + n].to_vec(),
            });
        }
    }
    if row.routes != expected_routes {
        return Err(failure("active route nodes/frames/roots").into());
    }

    let roots = client.packed_codebook_roots();
    let roots_at = |position: usize, block: usize| {
        ((roots[prefix[position] as usize] >> (4 * block)) & 15) as u8
    };
    let last = prefix.len() - 1;
    let prev = last.saturating_sub(1);
    let mid = last / 2;
    for block in 0..4 {
        let frames = (0..prefix.len())
            .scan(0u8, |frame, position| {
                let root = usize::from(roots_at(position, block));
                let step = tables.trunk[1 + 16 * block + root] as u8;
                let next = compose_frames(step, *frame);
                if let Ok(value) = next {
                    *frame = value;
                }
                Some(next)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let terminal = frames[last];
        let relative = |pos| compose_frames(terminal, invert_frame(frames[pos])?);
        let mut sum = orbit_root_coords(roots_at(last, block))?;
        for (pos, frame) in [
            (prev, relative(prev)?),
            (mid, relative(mid)?),
            (0, terminal),
        ] {
            let v = transform_block_by_frame(frame, &orbit_root_coords(roots_at(pos, block))?)?;
            for (coordinate, value) in sum.iter_mut().zip(v) {
                *coordinate += value;
            }
        }
        let norm = normalize_block_with_frame(&sum, 4, tables.trunk[65 + block] as u8)?;
        let indices = block * 8..(block + 1) * 8;
        if row.recurrent_frames[block] != i32::from(terminal)
            || sum != row.recurrent_sum[indices.clone()]
            || norm.normalized_block != row.recurrent_vector[indices]
            || u32::from(norm.aligned_root_index) != (row.recurrence_code >> (4 * block)) & 15
        {
            return Err(failure("raw recurrent frames, sum or normalized vector").into());
        }
    }

    let mut writer: Vec<_> = (0..tables.shape[0] as usize)
        .map(|token| {
            let distance = tables.table[32 * token..32 * (token + 1)]
                .iter()
                .zip(&eval.read_vector)
                .map(|(&a, &b)| a.abs_diff(b))
                .sum::<u32>();
            let admitted = eval.admitted_cells.contains(&tables.cells[token]);
            (!admitted, distance, token as u32)
        })
        .collect();
    writer.sort_unstable();
    let counts = &tables.trunk[70..70 + writer.len()];
    let mut prior: Vec<_> = (0..counts.len()).collect();
    prior.sort_unstable_by_key(|&token| (std::cmp::Reverse(counts[token]), token));
    if row.writer_candidates as usize != writer.iter().filter(|(a, _, _)| !a).count()
        || row.prior_selected != tables.trunk[69]
    {
        return Err(failure("writer candidate count or stored unigram fallback").into());
    }
    for (probe, id) in [
        (&row.probes.gold, row.input.target),
        (&row.probes.rival, row.rival),
        (&row.probes.space, row.space),
    ] {
        let index = id as usize;
        let prior_rank = prior
            .iter()
            .position(|&t| t == index)
            .ok_or("invalid prior ID")? as u32
            + 1;
        let writer_rank = writer
            .iter()
            .position(|&(_, _, t)| t == id)
            .ok_or("invalid writer ID")?;
        let (excluded, distance, _) = writer[writer_rank];
        if probe.id != id
            || probe.prior_count != counts[index]
            || probe.prior_rank != prior_rank
            || probe.prior_ties as usize
                != counts
                    .iter()
                    .filter(|&&count| count == counts[index])
                    .count()
            || probe.writer_cell != tables.cells[index]
            || probe.writer_distance != distance
            || probe.writer_admitted == excluded
            || probe.writer_full_rank != writer_rank as u32 + 1
            || probe.writer_rank != (!excluded).then_some(writer_rank as u32 + 1)
            || probe.writer_distance_ties as usize
                != writer.iter().filter(|(_, d, _)| *d == distance).count()
            || probe.writer_admitted_ties as usize
                != writer
                    .iter()
                    .filter(|(x, d, _)| !x && *d == distance)
                    .count()
            || probe.gate_score_terms.mass() != probe.mass
        {
            return Err(failure("probe, prior or writer rank").into());
        }
    }
    Ok(())
}
