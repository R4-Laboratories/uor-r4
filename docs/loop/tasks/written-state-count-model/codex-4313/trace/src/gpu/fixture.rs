use super::*;
use uor_r4_discrete::{
    metadata::BinderPlan,
    normalization_mechanism::{orbit_root_coords, NormalizationMode},
    q4_import::{COMPOSITION_KAPPA, OPERATOR_KAPPA},
    snapshot::Snapshot,
    trunk::{
        build_trunk_norm_frames, build_trunk_step_table, export_trunk_d11, TrunkTables,
        TrunkTrainRow,
    },
    ObjectivePolicy, ReadInitialization, ReadSpec, Spec, SCHEMA,
};

const GOLD: u32 = 31;
const RIVAL: u32 = 45;
const OFF_POOL: u32 = 60;
const ZERO_POOL: u32 = 61;
const QUERY: u32 = 9;
const ALIAS: u32 = 12;

fn snapshot() -> Result<Snapshot> {
    let spec = Spec {
        vocabulary: VOCAB,
        blocks: 4,
        context: CONTEXT as u32,
        kept_cells: 1,
    };
    let stages = BinderPlan::new(CONTEXT)?.stages() as u32;
    let read_spec = ReadSpec::Flock { kept: 2 };
    let table = (0..VOCAB)
        .map(|id| {
            let root = match id {
                0 => 12,
                6 => 6,
                8 => 8,
                QUERY => 15,
                11 => 11,
                GOLD => 2,
                RIVAL => 3,
                SPACE | ALIAS => 0,
                _ => 1,
            };
            Ok(orbit_root_coords(root)?.repeat(4))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let mut gates = vec![0; read_spec.gate_count(spec.blocks, stages)? as usize];
    for block in 0..spec.blocks as usize {
        gates[block * stages as usize] = if block == 3 { -1 } else { 1 };
    }
    Ok(Snapshot {
        schema: SCHEMA.to_owned(),
        spec,
        read_spec,
        objective_policy: ObjectivePolicy::NearestAddressMarginV1,
        read_initialization: ReadInitialization::ActiveLeafV1,
        normalization_mode: NormalizationMode::ScaleMatchedSnapV1,
        position: None,
        address_metric: Default::default(),
        seed: 1,
        epoch: 1,
        table,
        routes: vec![0; (spec.blocks * stages) as usize],
        gates,
        operator_kappa: OPERATOR_KAPPA.to_owned(),
        composition_kappa: COMPOSITION_KAPPA.to_owned(),
        kernel_sha256: "11".repeat(32),
        source_sha256: "22".repeat(32),
        geometry_sha256: "33".repeat(32),
    })
}

fn input(id: &str, side: usize, prefix: &[u32], target: u32) -> Input {
    Input {
        pair_id: id.to_owned(),
        side,
        cohort: "synthetic".to_owned(),
        prefix: prefix.to_owned(),
        target,
    }
}

pub(super) fn run() -> Result<()> {
    let snapshot = snapshot()?;
    let steps = build_trunk_step_table(4, 0)?;
    let norms = build_trunk_norm_frames(4, 0)?;
    let mut counts = TrunkTables::build(&[TrunkTrainRow {
        read_code: u16::MAX,
        rec_code: u16::MAX,
        rec_shuf_code: u16::MAX,
        rec_unit_code: u16::MAX,
        query_token: 4000,
        target_token: SPACE as u16,
    }])?;
    counts.unigram_counts[SPACE as usize] = 7;
    counts.unigram_counts[GOLD as usize] = 3;
    counts.unigram_counts[RIVAL as usize] = 3;
    counts.unigram_counts[OFF_POOL as usize] = 2;

    let a = [0, 6, 1, QUERY];
    let b = [0, 8, 1, QUERY];
    let preliminary =
        IntegerDiscreteModel::from_bytes(&export_trunk_d11(&snapshot, &steps, &norms, &counts)?)?;
    let gate = |prefix| -> Result<_> {
        Ok(preliminary
            .evaluate_prefix_traced(prefix)?
            .trunk_pointer
            .ok_or("fixture has no native gate")?)
    };
    let a_keys = gate(&a)?;
    let b_keys = gate(&b)?;
    if a_keys.recurrence_code == b_keys.recurrence_code {
        return Err("finite question contrast collapsed before planted count rows".into());
    }
    for query in [QUERY, 11] {
        for token in [GOLD, RIVAL, SPACE] {
            counts.query_table.add(query, token as u16);
        }
    }
    let key = |q, code: u16| (q << 16) | u32::from(code);
    counts
        .query_rec_table
        .add(key(QUERY, a_keys.recurrence_code), GOLD as u16);
    counts
        .query_rec_table
        .add(key(QUERY, b_keys.recurrence_code), RIVAL as u16);
    counts
        .query_read_table
        .add(key(QUERY, a_keys.read_code), OFF_POOL as u16);
    counts
        .rec_table
        .add(u32::from(a_keys.recurrence_code), ZERO_POOL as u16);
    let (tables, client) =
        ArtifactTables::from_bytes(&export_trunk_d11(&snapshot, &steps, &norms, &counts)?)?;

    let rows = vec![
        input("swap", 0, &a, GOLD),
        input("swap", 1, &b, RIVAL),
        input("zero-and-inactive", 0, &a, ZERO_POOL),
        input("positive-tie", 0, &[11], GOLD),
        input("fallback", 0, &[10], GOLD),
        input("pointer", 0, &[0, QUERY, GOLD, QUERY], GOLD),
    ];
    let ids = vec![
        [GOLD, RIVAL, SPACE],
        [RIVAL, GOLD, SPACE],
        [ZERO_POOL, OFF_POOL, SPACE],
        [GOLD, RIVAL, SPACE],
        [GOLD, RIVAL, SPACE],
        [GOLD, RIVAL, SPACE],
    ];
    let result = score(&tables, rows, ids)?;
    for row in &result {
        super::super::parity::check(&tables, &client, row)?;
    }
    let device = scorer(&tables)?;
    let prefixes: Vec<_> = result.iter().map(|r| r.input.prefix.as_slice()).collect();
    let targets: Vec<_> = result.iter().map(|r| r.input.target).collect();
    let native = device.score(&pack_rows_with_targets(&prefixes, &targets)?)?;
    if native.len() != result.len()
        || native.iter().zip(&result).any(|(output, trace)| {
            output.status != 0
                || output.prediction != trace.emitted
                || output.top8 != trace.top8
                || output.admitted_cells != trace.admitted_cells
                || output.read_vector != trace.writer_vector
        })
    {
        return Err("instrumentation changed the native geometric writer".into());
    }
    let [a, b, inactive, tie, fallback, pointer] = result.as_slice() else {
        return Err("missing finite fixture row".into());
    };
    if a.recurrent_vector == b.recurrent_vector
        || a.recurrent_sum == b.recurrent_sum
        || a.recurrent_frames == b.recurrent_frames
        || [a.gate_selected, b.gate_selected] != [GOLD, RIVAL]
        || [a.emitted, b.emitted] != [GOLD, RIVAL]
        || [a.probes.gold.mass, b.probes.gold.mass] != [33, 33]
        || [a.probes.rival.gate_rank, b.probes.rival.gate_rank] != [Some(3), Some(3)]
        || [a.probes.space.gate_rank, b.probes.space.gate_rank] != [Some(2), Some(2)]
        || [a.positive_candidates, b.positive_candidates] != [3, 3]
        || a.probes.rival.writer_rank.is_some()
        || b.probes.rival.writer_rank.is_some()
        || [
            a.probes.gold.writer_full_rank,
            b.probes.gold.writer_full_rank,
        ] != [1, 1]
    {
        return Err(
            "positive question-swap fixture lost a planted recurrence/rank/writer effect".into(),
        );
    }
    if inactive.probes.gold.mass != 0
        || !inactive.probes.gold.pool_member
        || inactive.probes.rival.mass != 64
        || inactive.probes.rival.pool_member
        || [
            inactive.probes.gold.gate_rank,
            inactive.probes.rival.gate_rank,
        ] != [None, None]
        || inactive.probes.gold.mass_ties != VOCAB - 4
        || inactive.probes.rival.mass_ties != 1
    {
        return Err("inactive positive or zero-mass fixture disagreed".into());
    }
    if tie.probes.space.eligible_mass_ties != 3
        || [
            tie.probes.space.gate_rank,
            tie.probes.gold.gate_rank,
            tie.probes.rival.gate_rank,
        ] != [Some(1), Some(2), Some(3)]
        || [
            tie.probes.space.prior_rank,
            tie.probes.gold.prior_rank,
            tie.probes.rival.prior_rank,
        ] != [1, 2, 3]
        || tie.gate_selected != SPACE
        || tie.emitted != ALIAS
        || tie.probes.space.writer_rank != Some(2)
        || tie.probes.space.writer_admitted_ties != 2
    {
        return Err(
            "unigram/ID positive tie or final geometric writer tie fixture disagreed".into(),
        );
    }
    if fallback.gate_mass != 0
        || fallback.gate_selected != SPACE
        || fallback.positive_candidates != 0
        || fallback.probes.space.gate_rank != Some(1)
        || fallback.probes.gold.gate_rank.is_some()
        || fallback.probes.gold.mass_ties != VOCAB
    {
        return Err("zero-mass fallback fixture disagreed".into());
    }
    if pointer.source_position != Some(2)
        || pointer.source_token != Some(GOLD)
        || pointer.probes.gold.gate_score_terms.pointer_bonus != 256
        || pointer.gate_selected != GOLD
        || pointer.emitted != GOLD
    {
        return Err("native source/pointer fixture disagreed".into());
    }
    println!(
        "{}",
        serde_json::json!({
            "status": "pass", "rows": result.len(), "artifact_sha256": tables.artifact_sha256,
            "capability": device.capability,
            "controls": ["nontrivial_recurrence_question_swap", "active_and_absent_positive_mass",
                "pool_member_with_zero_mass", "prior_and_id_positive_ties", "zero_mass_fallback",
                "writer_distance_and_id_tie", "nonadmitted_writer_id",
                "native_matched_source_with_256_bonus", "normal_scorer_path", "reverse_gpu_replay",
                "portable_client_state_and_rank_replay"]
        })
    );
    Ok(())
}
