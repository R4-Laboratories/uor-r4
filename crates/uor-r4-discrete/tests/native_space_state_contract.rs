//! Unified executable D85 native-space state contract, raw-word replay fixture,
//! and planted-fault suite (`native-space-state-contract`, Issue `#3973`).
//!
//! Compares raw integer words (`u32`, `i32`, `i64`) across training and serving paths
//! for all five D85 parity fields (`route choices`, `carry state`, `read choices`,
//! `memory addresses`, `token ids`) and verifies that planted operator, scale,
//! route, carry, read-choice, memory-address, and token-id faults fail through
//! the same shared replay comparators.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use uor_r4_core::spiralcore_operator::{
    cl06_finite_composition_table, CL06_FINITE_COMPOSITION_KAPPA_REFERENCE,
    SPIRALCORE_OPERATOR_KAPPA_REFERENCE,
};
use uor_r4_cpu_train::{
    CpuDiscretePick, CpuDiscreteProposalApplier, CpuDiscreteProposalKind,
    CpuDiscreteProposalSample, CpuDiscreteProposalScorer,
};
use uor_r4_discrete::d11_export::export_d11;
use uor_r4_discrete::metadata::BinderPlan;
use uor_r4_discrete::normalization_mechanism::{orbit_root_coords, NormalizationMode};
use uor_r4_discrete::snapshot::Snapshot;
use uor_r4_discrete::{ObjectivePolicy, ReadInitialization, ReadSpec, Spec, SCHEMA};
use uor_r4_integer::discrete_d11::{
    ExactCopyState, IntegerDiscreteModel, DISCRETE_ACTIVE_TRACE_HEADER_WORDS,
    DISCRETE_ACTIVE_TRACE_LEVEL_WORDS,
};
use uor_r4_integer::session_codec;
use uor_r4_integer::stack::synthetic::synthetic_fbs_trunk;
use uor_r4_integer::stack::{
    add_q31_read_uorlut01, export_fbs_trunk_uorlut01, export_fbs_trunk_uorlut01_with_lane_gates,
    export_fbs_trunk_uorlut01_with_q31, read_q31_read_uorlut01, FbsExportSnaps, IntegerStackModel,
    NativeQ31Carry, Q31ReadStep, Q31SessionRead,
};
use uor_r4_native_vector::contract::{
    canonical_boundary_inventory, canonical_native_operator_registry,
    canonical_native_state_schema, validate_contract, ParityField, ALL_PARITY_FIELDS,
};
use uor_r4_native_vector::{
    q31_hamilton_rotate, q31_read_address, Q31FirstPickRead, Q31HamiltonModel, Q31LaneGates,
    Q31Quaternion, Q31ReadHistory,
};

const Q31_ONE: i64 = 1_i64 << 31;
const Q31_IDENTITY: Q31Quaternion = [Q31_ONE, 0, 0, 0];

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReplayMismatch {
    field: ParityField,
    location: String,
    expected: String,
    actual: String,
}

#[derive(Debug, Default)]
struct DiscreteReplaySummary {
    rows: usize,
    trace_words_compared: usize,
    carry_words_compared: usize,
    admitted_cell_words_compared: usize,
    top8_words_compared: usize,
    prediction_words_compared: usize,
    kept_slots_compared: usize,
    action_slots_compared: usize,
    mismatches: Vec<ReplayMismatch>,
}

impl DiscreteReplaySummary {
    fn is_exact(&self) -> bool {
        self.mismatches.is_empty()
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root two levels above uor-r4-discrete")
        .to_path_buf()
}

fn all_prefixes(vocab: u32, max_len: u32) -> Vec<CpuDiscreteProposalSample> {
    let mut out = Vec::new();
    for len in 1..=max_len {
        for code in 0..vocab.pow(len) {
            let prefix: Vec<u32> = (0..len)
                .map(|i| (code / vocab.pow(len - 1 - i)) % vocab)
                .collect();
            let target = prefix.iter().sum::<u32>() % vocab;
            out.push(CpuDiscreteProposalSample { prefix, target });
        }
    }
    out
}

fn discrete_snapshot(
    roots: &[[u8; 2]],
    context: u32,
    kept_cells: u32,
    read_spec: ReadSpec,
    read_initialization: ReadInitialization,
    routes: Vec<u32>,
    gates: Vec<i32>,
) -> Snapshot {
    let vocab = roots.len() as u32;
    let blocks = 2u32;
    let table: Vec<i32> = roots
        .iter()
        .flat_map(|pair| {
            pair.iter()
                .flat_map(|&r| orbit_root_coords(r).expect("root"))
        })
        .collect();
    Snapshot {
        schema: SCHEMA.to_owned(),
        spec: Spec {
            vocabulary: vocab,
            blocks,
            context,
            kept_cells,
        },
        read_spec,
        objective_policy: ObjectivePolicy::NearestAddressMarginV1,
        read_initialization,
        normalization_mode: NormalizationMode::ScaleMatchedSnapV1,
        position: None,
        address_metric: Default::default(),
        seed: 0x2026_1124,
        epoch: 0,
        table,
        routes,
        gates,
        operator_kappa: SPIRALCORE_OPERATOR_KAPPA_REFERENCE.to_owned(),
        composition_kappa: CL06_FINITE_COMPOSITION_KAPPA_REFERENCE.to_owned(),
        kernel_sha256: "11".repeat(32),
        source_sha256: "22".repeat(32),
        geometry_sha256: "33".repeat(32),
    }
}

fn route_fixture_bytes(roots: &[[u8; 2]], routes: Vec<u32>) -> Vec<u8> {
    let snap = discrete_snapshot(
        roots,
        3,
        2,
        ReadSpec::Route,
        ReadInitialization::ZeroGatesV1,
        routes,
        vec![0; 4],
    );
    export_d11(&snap).expect("export route snapshot")
}

fn flock_fixture_bytes(roots: &[[u8; 2]], gates: Vec<i32>) -> Vec<u8> {
    let snap = discrete_snapshot(
        roots,
        3,
        2,
        ReadSpec::Flock { kept: 2 },
        ReadInitialization::ActiveLeafV1,
        vec![0; 4],
        gates,
    );
    export_d11(&snap).expect("export flock snapshot")
}

/// Shared raw-word discrete replay comparator between training oracle (`CpuDiscreteProposalScorer`)
/// and served integer runtime (`IntegerDiscreteModel`). Compares raw integer words across:
/// - Route/Read choice traces (`DISCRETE_ROUTE_TRACE_*` / `DISCRETE_ACTIVE_TRACE_*`)
/// - Carry vector words (`i32`)
/// - Admitted memory cells (`u32`)
/// - Emitted `top8` and `prediction` token IDs (`u32`)
fn compare_discrete_replay(
    train_bytes: &[u8],
    serve_bytes: &[u8],
    samples: &[CpuDiscreteProposalSample],
    trace_parity_field: ParityField,
    compare_trace_words: bool,
    is_active_leaf: bool,
) -> std::result::Result<DiscreteReplaySummary, String> {
    let scorer = CpuDiscreteProposalScorer::from_bytes(train_bytes)
        .map_err(|e| format!("training oracle load failed: {e}"))?;
    let served = IntegerDiscreteModel::from_bytes(serve_bytes)
        .map_err(|e| format!("serving loader refused artifact: {e}"))?;
    let export_reading = scorer
        .export_parity(serve_bytes, samples)
        .map_err(|e| format!("export parity check failed: {e}"))?;

    let mut summary = DiscreteReplaySummary {
        rows: samples.len(),
        ..Default::default()
    };

    if export_reading.admitted_cell_mismatches > 0 {
        summary.mismatches.push(ReplayMismatch {
            field: ParityField::MemoryAddresses,
            location: "admitted_cells".into(),
            expected: "0 mismatches".into(),
            actual: format!("{} mismatches", export_reading.admitted_cell_mismatches),
        });
    }

    let kept_cells = served.shape().kept_cells as usize;
    let stages = served.shape().stages as usize;
    summary.admitted_cell_words_compared = samples.len() * kept_cells;

    for (row, sample) in samples.iter().enumerate() {
        let cpu = scorer
            .query_traced(sample)
            .map_err(|e| format!("cpu query_traced row {row}: {e}"))?;
        let serving_traced = served
            .evaluate_prefix_traced(&sample.prefix)
            .map_err(|e| format!("serving evaluate_prefix_traced row {row}: {e}"))?;

        summary.carry_words_compared += cpu.query.len();
        if cpu.query != serving_traced.evaluation.read_vector {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::CarryState,
                location: format!("row {row} prefix {:?}", sample.prefix),
                expected: format!("{:?}", cpu.query),
                actual: format!("{:?}", serving_traced.evaluation.read_vector),
            });
        }

        if compare_trace_words {
            summary.trace_words_compared += cpu.trace_words.len();
            if cpu.trace_words != serving_traced.trace_words {
                summary.mismatches.push(ReplayMismatch {
                    field: trace_parity_field,
                    location: format!("trace row {row} prefix {:?}", sample.prefix),
                    expected: format!("{:?}", &cpu.trace_words[..cpu.trace_words.len().min(12)]),
                    actual: format!(
                        "{:?}",
                        &serving_traced.trace_words[..serving_traced.trace_words.len().min(12)]
                    ),
                });
            }
        }

        if is_active_leaf {
            let stride =
                DISCRETE_ACTIVE_TRACE_HEADER_WORDS + stages * DISCRETE_ACTIVE_TRACE_LEVEL_WORDS;
            for block in serving_traced.trace_words.chunks_exact(stride) {
                for level in block[DISCRETE_ACTIVE_TRACE_HEADER_WORDS..]
                    .chunks_exact(DISCRETE_ACTIVE_TRACE_LEVEL_WORDS)
                {
                    let found = level[1] as usize;
                    summary.kept_slots_compared += found;
                    summary.action_slots_compared += found;
                }
            }
        }

        let cpu_top8 = scorer
            .top8(sample)
            .map_err(|e| format!("cpu top8 row {row}: {e}"))?;
        let cpu_top8_slice: Vec<u32> = cpu_top8
            .iter()
            .copied()
            .take(serving_traced.evaluation.top8.len())
            .collect();
        summary.top8_words_compared += serving_traced.evaluation.top8.len();
        summary.prediction_words_compared += 1;
        if cpu_top8_slice != serving_traced.evaluation.top8
            || cpu_top8[0] != serving_traced.evaluation.prediction
        {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::TokenIds,
                location: format!("emission row {row} prefix {:?}", sample.prefix),
                expected: format!("top8={cpu_top8_slice:?} pred={}", cpu_top8[0]),
                actual: format!(
                    "top8={:?} pred={}",
                    serving_traced.evaluation.top8, serving_traced.evaluation.prediction
                ),
            });
        }
    }
    Ok(summary)
}

#[derive(Debug, Default)]
struct Q31ReplaySummary {
    steps: usize,
    gate_words_compared: usize,
    carry_words_compared: usize,
    read_picks_compared: usize,
    address_words_compared: usize,
    read_out_words_compared: usize,
    mismatches: Vec<ReplayMismatch>,
}

fn q31_rotate_with_operator(
    gate: Q31Quaternion,
    value: Q31Quaternion,
    fault_shift_30: bool,
) -> Q31Quaternion {
    if !fault_shift_30 {
        return q31_hamilton_rotate(gate, value).expect("admitted Q31 words");
    }
    // Intentionally injected operator shift fault (>> 30 instead of >> 31).
    let p = |a: i64, b: i64| a.wrapping_mul(b) as u64;
    let w = |raw: u64| (((raw.wrapping_add(1 << 29)) as i64) >> 30).clamp(-(1 << 31), 1 << 31);
    [
        w(p(gate[0], value[0])
            .wrapping_sub(p(gate[1], value[1]))
            .wrapping_sub(p(gate[2], value[2]))
            .wrapping_sub(p(gate[3], value[3]))),
        w(p(gate[0], value[1])
            .wrapping_add(p(gate[1], value[0]))
            .wrapping_add(p(gate[2], value[3]))
            .wrapping_sub(p(gate[3], value[2]))),
        w(p(gate[0], value[2])
            .wrapping_sub(p(gate[1], value[3]))
            .wrapping_add(p(gate[2], value[0]))
            .wrapping_add(p(gate[3], value[1]))),
        w(p(gate[0], value[3])
            .wrapping_add(p(gate[1], value[2]))
            .wrapping_sub(p(gate[2], value[1]))
            .wrapping_add(p(gate[3], value[0]))),
    ]
}

fn compare_q31_hamilton_replay(
    trained_model: &Q31HamiltonModel,
    serve_artifact: &[u8],
    prompt: &[u32],
    fault_shift_30: bool,
) -> Q31ReplaySummary {
    let mut summary = Q31ReplaySummary::default();
    let served = IntegerStackModel::parse(serve_artifact).expect("parse /12 artifact");
    let oracle_state = trained_model.to_bytes();
    match served.native_q31_state_bytes() {
        Some(bytes) if bytes == oracle_state => {
            summary.gate_words_compared += 4;
        }
        other => {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::CarryState,
                location: "q31.state bytes".into(),
                expected: format!("{oracle_state:?}"),
                actual: format!("{other:?}"),
            });
            return summary;
        }
    }

    let mut session = served.session();
    session.set_native_q31_trace(true).expect("enable trace");
    let mut prev: Option<Vec<NativeQ31Carry>> = None;
    for (step_idx, &tok) in prompt.iter().enumerate() {
        session.step_native(tok).expect("step_native");
        summary.steps += 1;
        let carries = session.take_native_q31_trace().expect("q31 carries");
        for carry in &carries {
            let expected_transport =
                q31_rotate_with_operator(trained_model.gate(), carry.before, fault_shift_30);
            summary.carry_words_compared += 4;
            if carry.transport != expected_transport {
                summary.mismatches.push(ReplayMismatch {
                    field: ParityField::CarryState,
                    location: format!(
                        "step {step_idx} layer {} lane {} transport",
                        carry.layer, carry.lane
                    ),
                    expected: format!("{expected_transport:?}"),
                    actual: format!("{:?}", carry.transport),
                });
            }
        }
        if let Some(prev_carries) = &prev {
            for (cur, old) in carries.iter().zip(prev_carries.iter()) {
                summary.carry_words_compared += 4;
                if cur.before != old.after {
                    summary.mismatches.push(ReplayMismatch {
                        field: ParityField::CarryState,
                        location: format!(
                            "step {step_idx} layer {} lane {} carry continuity",
                            cur.layer, cur.lane
                        ),
                        expected: format!("{:?}", old.after),
                        actual: format!("{:?}", cur.before),
                    });
                }
            }
        }
        prev = Some(carries);
    }
    summary
}

fn compare_q31_read_replay(
    train_read: &Q31FirstPickRead,
    serve_artifact: &[u8],
    prompt: &[u32],
    writer_table: &[i64],
) -> Q31ReplaySummary {
    let mut summary = Q31ReplaySummary::default();
    let served_read = read_q31_read_uorlut01(serve_artifact)
        .expect("parse q31.read")
        .expect("q31.read present")
        .read()
        .clone();
    let served = IntegerStackModel::parse(serve_artifact).expect("parse /14 artifact");
    let mut session = served.session();
    session
        .set_q31_read_trace(true)
        .expect("enable q31 read trace");

    let mut trace: Vec<Q31ReadStep> = Vec::with_capacity(prompt.len());
    for &tok in prompt {
        session.step_native(tok).expect("step_native");
        summary.steps += 1;
    }
    trace.extend_from_slice(session.q31_read_trace().expect("q31 read trace"));
    let stream: Vec<i64> = trace.iter().flat_map(|s| s.x_in.iter().copied()).collect();
    let batch_out = train_read
        .apply(&stream, prompt, writer_table, 1, prompt.len())
        .expect("Q31FirstPickRead::apply");

    let mut step_history = Q31ReadHistory::new();
    let width = train_read.width();
    for (t, step) in trace.iter().enumerate() {
        let row = &stream[t * width..(t + 1) * width];
        let train_query = q31_read_address(train_read.query_gates(), row).expect("train query");
        let train_key = q31_read_address(train_read.key_gates(), row).expect("train key");
        let served_query = q31_read_address(served_read.query_gates(), row).expect("served query");
        let served_key = q31_read_address(served_read.key_gates(), row).expect("served key");
        summary.address_words_compared += train_query.len() + train_key.len();
        if served_query != train_query {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::MemoryAddresses,
                location: format!("q31.read query address t={t}"),
                expected: format!("{train_query:?}"),
                actual: format!("{served_query:?}"),
            });
        }
        if served_key != train_key {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::MemoryAddresses,
                location: format!("q31.read key address t={t}"),
                expected: format!("{train_key:?}"),
                actual: format!("{served_key:?}"),
            });
        }

        let (step_pick, step_out) = train_read
            .step(&mut step_history, row, prompt[t], writer_table)
            .expect("Q31FirstPickRead::step");
        if step_pick != batch_out.picks[t] {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::ReadChoices,
                location: format!("train step vs apply t={t}"),
                expected: format!("{}", batch_out.picks[t]),
                actual: format!("{step_pick}"),
            });
        }
        if step_out.as_slice() != &batch_out.out[t * width..(t + 1) * width] {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::CarryState,
                location: format!("train step vs apply out t={t}"),
                expected: format!("{:?}", &batch_out.out[t * width..t * width + 4]),
                actual: format!("{:?}", &step_out[..4]),
            });
        }

        summary.read_picks_compared += 1;
        if step.pick != batch_out.picks[t] {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::ReadChoices,
                location: format!("served pick t={t}"),
                expected: format!("{}", batch_out.picks[t]),
                actual: format!("{}", step.pick),
            });
        }

        let expected_out = &batch_out.out[t * width..(t + 1) * width];
        summary.read_out_words_compared += expected_out.len();
        if step.out.as_slice() != expected_out {
            summary.mismatches.push(ReplayMismatch {
                field: ParityField::CarryState,
                location: format!("served out row t={t}"),
                expected: format!("{:?}", &expected_out[..8.min(width)]),
                actual: format!("{:?}", &step.out[..8.min(width)]),
            });
        }
    }
    summary
}

fn read_i16_table_as_i64(artifact: &[u8], table_name: &str) -> Vec<i64> {
    let header_len = u64::from_le_bytes(artifact[8..16].try_into().expect("header len")) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&artifact[16..16 + header_len]).expect("header json");
    let table_spec = header["tables"]
        .as_array()
        .expect("tables")
        .iter()
        .find(|t| t["name"].as_str() == Some(table_name))
        .expect("target table");
    let len = table_spec["len"].as_u64().expect("len") as usize;
    let span = &table_spec["span"];
    let data_start = (16 + header_len).div_ceil(64) * 64;
    let offset = data_start + span["offset"].as_u64().expect("offset") as usize;
    let bytes = &artifact[offset..offset + len * 2];
    bytes
        .chunks_exact(2)
        .map(|chunk| i64::from(i16::from_le_bytes([chunk[0], chunk[1]])))
        .collect()
}

fn plain_synthetic_fbs_and_table() -> (Vec<u8>, Vec<i64>) {
    let snap = synthetic_fbs_trunk(20_261_115);
    let plain = export_fbs_trunk_uorlut01(&snap).expect("export synthetic fbs");
    let writer_table = read_i16_table_as_i64(&plain, "writer.table");
    (plain, writer_table)
}

#[test]
fn contract_registry_files_and_symbols_exist_in_workspace() {
    validate_contract().expect("canonical native-space contract must validate");
    let root = repo_root();

    let schema = canonical_native_state_schema();
    assert_eq!(schema.owner_decision, "D85");
    let covered_schema_fields: BTreeSet<&str> = schema
        .fields
        .iter()
        .filter_map(|f| f.parity_field.map(ParityField::label))
        .collect();
    for parity in ALL_PARITY_FIELDS {
        assert!(
            covered_schema_fields.contains(parity.label()),
            "schema lacks parity field {}",
            parity.label()
        );
    }
    for field in schema.fields {
        let (file, sym_path) = field.source_anchor.split_once("::").unwrap_or_else(|| {
            panic!(
                "field {} anchor lacks `::`: {}",
                field.name, field.source_anchor
            )
        });
        let full = root.join(file);
        assert!(
            full.exists(),
            "field {} references missing anchor file {}",
            field.name,
            file
        );
        let contents =
            fs::read_to_string(&full).unwrap_or_else(|e| panic!("failed reading {}: {e}", file));
        let leaf = sym_path.rsplit("::").next().expect("leaf symbol");
        assert!(
            contents.contains(leaf),
            "field {} references missing symbol `{}` in {}",
            field.name,
            leaf,
            file
        );
    }

    let registry = canonical_native_operator_registry();
    for op in registry.operators {
        for binding in [
            op.cuda_training,
            op.integer_serving,
            op.cpu_oracle,
            op.wasm_serving,
        ]
        .into_iter()
        .flatten()
        {
            let full = root.join(binding.file);
            assert!(
                full.exists(),
                "operator {} references missing file {}",
                op.id,
                binding.file
            );
            let contents = fs::read_to_string(&full)
                .unwrap_or_else(|e| panic!("failed reading {}: {e}", binding.file));
            assert!(
                contents.contains(binding.symbol),
                "operator {} references missing symbol `{}` in {}",
                op.id,
                binding.symbol,
                binding.file
            );
        }
    }

    let boundary = canonical_boundary_inventory();
    for entry in boundary.paths {
        let full = root.join(entry.path);
        assert!(
            full.exists(),
            "boundary inventory references missing path {}",
            entry.path
        );
    }
}

#[test]
fn discrete_native_replay_matches_raw_words_for_all_five_parity_fields() {
    let roots = [[0u8, 1], [2, 3], [4, 5]];
    let samples = all_prefixes(3, 3);
    assert_eq!(samples.len(), 39);

    // 1. Route fixture: routes, carry, memory addresses, emitted token ids, and epoch move.
    let route_bytes = route_fixture_bytes(&roots, vec![1, 2, 3, 4]);
    let route_summary = compare_discrete_replay(
        &route_bytes,
        &route_bytes,
        &samples,
        ParityField::RouteChoices,
        true,
        false,
    )
    .expect("route replay");
    assert!(route_summary.is_exact(), "{:?}", route_summary.mismatches);
    assert!(route_summary.trace_words_compared > 0);
    assert!(route_summary.carry_words_compared > 0);
    assert!(route_summary.admitted_cell_words_compared > 0);
    assert_eq!(route_summary.top8_words_compared, 78);
    assert_eq!(route_summary.prediction_words_compared, 39);
    assert!(route_summary.token_words_ready() > 0);

    // Apply one native discrete Table move (epoch 0 -> 1) and verify exact replay on stepped state.
    let mut applier = CpuDiscreteProposalApplier::from_bytes(&route_bytes).expect("applier");
    let receipt = applier
        .apply(
            CpuDiscreteProposalKind::Table,
            0,
            uor_r4_cpu_train::CpuDiscreteProposal {
                lane: 1,
                current_loss: 10,
                selected_loss: 9,
                epoch: 0,
                sensitive: true,
            },
        )
        .expect("apply move");
    assert!(receipt.applied);
    let stepped_bytes = applier.artifact_bytes().expect("stepped bytes");
    let stepped_summary = compare_discrete_replay(
        &stepped_bytes,
        &stepped_bytes,
        &samples,
        ParityField::RouteChoices,
        true,
        false,
    )
    .expect("stepped route replay");
    assert!(
        stepped_summary.is_exact(),
        "{:?}",
        stepped_summary.mismatches
    );
    assert_eq!(
        IntegerDiscreteModel::from_bytes(&stepped_bytes)
            .expect("load stepped")
            .shape()
            .epoch,
        1
    );

    // 2. Flock fixture: active read choices (kept nodes + actions), carry, cells, token ids.
    let flock_bytes = flock_fixture_bytes(&roots, vec![1, 0, -1, 1]);
    let flock_summary = compare_discrete_replay(
        &flock_bytes,
        &flock_bytes,
        &samples,
        ParityField::ReadChoices,
        true,
        true,
    )
    .expect("flock replay");
    assert!(flock_summary.is_exact(), "{:?}", flock_summary.mismatches);
    assert_eq!(flock_summary.kept_slots_compared, 126);
    assert_eq!(flock_summary.action_slots_compared, 126);
    assert_eq!(flock_summary.top8_words_compared, 78);

    // 3. Pick fixture (`CpuDiscretePick` vs `IntegerDiscreteModel` serving evaluator).
    let pick_steps = uor_r4_discrete::position_mechanism::PositionStepTable::new(2, vec![1u8; 32])
        .expect("step table");
    let mut pick_snap = discrete_snapshot(
        &roots,
        3,
        2,
        ReadSpec::Pick,
        ReadInitialization::ActiveLeafV1,
        vec![0; 4],
        vec![1, -1, 0, 0],
    );
    pick_snap.position = Some(uor_r4_discrete::position_mechanism::PositionConfig::new(
        uor_r4_discrete::position_mechanism::PositionMode::PrefixHolonomyV1,
        pick_steps,
    ));
    let pick_bytes = export_d11(&pick_snap).expect("export pick snapshot");
    let pick_summary = compare_discrete_replay(
        &pick_bytes,
        &pick_bytes,
        &samples,
        ParityField::ReadChoices,
        false,
        false,
    )
    .expect("pick replay");
    assert!(pick_summary.is_exact(), "{:?}", pick_summary.mismatches);
    let pick_oracle = CpuDiscretePick::from_bytes(&pick_bytes).expect("cpu pick");
    let trace_p = pick_oracle.read_traced(&[0, 1, 2]).expect("pick trace");
    assert_eq!(trace_p.pick, Some(1));
    let pick_served = IntegerDiscreteModel::from_bytes(&pick_bytes)
        .expect("load pick")
        .evaluate_prefix(&[0, 1, 2])
        .expect("eval pick");
    let pick_flat: Vec<i32> = trace_p.blocks.iter().flatten().copied().collect();
    assert_eq!(pick_flat, pick_served.read_vector);
    assert!(cl06_finite_composition_table().is_ok());

    println!(
        "discrete_native_replay rows={} route_trace_words={} route_carry_words={} flock_trace_words={} flock_kept_slots={} flock_action_slots={} pick_carry_words={}",
        route_summary.rows,
        route_summary.trace_words_compared,
        route_summary.carry_words_compared,
        flock_summary.trace_words_compared,
        flock_summary.kept_slots_compared,
        flock_summary.action_slots_compared,
        pick_summary.carry_words_compared,
    );

    // 4. ExactCopyState lossless raw-word transition & occurrence copy verification.
    let served_route = IntegerDiscreteModel::from_bytes(&route_bytes).expect("served route");
    let copy_start = ExactCopyState::new(Some(0), 2);
    assert_eq!(ExactCopyState::from_words(copy_start.words()), copy_start);
    let (_eval, step) = served_route
        .evaluate_prefix_with_exact_copy(&[0, 1, 2], copy_start, [u32::MAX; 4])
        .expect("exact copy step");
    assert_eq!(step.copied, 1);
    assert_eq!(step.emitted, 0);
    assert_eq!(step.words(), [1, 0, 1, 2, 0, 0, 1, 0]);
    let _binder = BinderPlan::new(3).expect("binder plan");
}

trait SummaryTokens {
    fn token_words_ready(&self) -> usize;
}

impl SummaryTokens for DiscreteReplaySummary {
    fn token_words_ready(&self) -> usize {
        self.top8_words_compared + self.prediction_words_compared
    }
}

#[test]
fn q31_native_replay_matches_training_and_serving_raw_words() {
    let snap = synthetic_fbs_trunk(20_261_115);
    let prompt = [3u32, 5, 8, 13, 21, 34, 55, 89];

    // 1. Train one Q31HamiltonModel natively (`step_reference`), serialize to `/12`, replay raw words.
    let mut trained = Q31HamiltonModel::new(Q31_IDENTITY, 8).expect("initial model");
    let values = vec![[1 << 30, -(1 << 29), 1 << 28, -(1 << 27)]; 8];
    let targets = vec![[1_518_500_249, 1_518_500_249, 0, 0]; 8];
    for _ in 0..16 {
        trained.step_reference(&values, &targets).expect("q31 step");
    }
    assert!(trained.epoch() > 0);
    let artifact_12 =
        export_fbs_trunk_uorlut01_with_q31(&snap, FbsExportSnaps::SNAPPED, &trained.to_bytes())
            .expect("export /12");
    let summary_12 = compare_q31_hamilton_replay(&trained, &artifact_12, &prompt, false);
    assert!(
        summary_12.mismatches.is_empty(),
        "{:?}",
        summary_12.mismatches
    );
    assert_eq!(summary_12.steps, prompt.len());
    assert_eq!(summary_12.carry_words_compared, 960);
    println!(
        "q31_native_replay q31_12_carry_words={} q31_12_steps={}",
        summary_12.carry_words_compared, summary_12.steps
    );

    // Verify lossless session save/restore across UR4SESS bytes preserves token & carry continuation.
    let model_12 = IntegerStackModel::parse(&artifact_12).expect("parse /12");
    let mut sess_a = model_12.session();
    for &t in &prompt[..4] {
        sess_a.step_native(t).expect("prefix");
    }
    let saved_bytes = session_codec::encode(&sess_a.save_state());
    let restored_state = session_codec::decode(&saved_bytes).expect("decode session");
    let mut sess_b = model_12.session();
    sess_b.restore_state(&restored_state).expect("restore");
    sess_a.set_native_q31_trace(true).expect("trace a");
    sess_b.set_native_q31_trace(true).expect("trace b");
    for &t in &prompt[4..] {
        let sa = sess_a.step_native(t).expect("step a");
        let sb = sess_b.step_native(t).expect("step b");
        assert_eq!(sa.token, sb.token);
        assert_eq!(
            sess_a.take_native_q31_trace(),
            sess_b.take_native_q31_trace()
        );
    }

    // 2. Per-lane `/13` replay matches oracle `Q31LaneGates::transition` at every (layer, lane).
    let lane_gates: Vec<Q31Quaternion> = (0..8)
        .map(|lane| {
            let mut g = [0_i64; 4];
            g[lane % 4] = Q31_ONE;
            g[(lane + 1) % 4] = 65_536 + lane as i64;
            g
        })
        .collect();
    let mut lanes_state = Q31LaneGates::new(lane_gates, 8).expect("lane gates");
    lanes_state.set_cursor(7, 8).expect("cursor");
    let artifact_13 = export_fbs_trunk_uorlut01_with_lane_gates(
        &snap,
        FbsExportSnaps::SNAPPED,
        &lanes_state.to_bytes(),
    )
    .expect("export /13");
    let model_13 = IntegerStackModel::parse(&artifact_13).expect("parse /13");
    let mut sess_13 = model_13.session();
    sess_13.set_native_q31_trace(true).expect("trace /13");
    for &t in &prompt {
        sess_13.step_native(t).expect("step /13");
        for carry in sess_13.take_native_q31_trace().expect("carries /13") {
            assert_eq!(
                carry.transport,
                lanes_state
                    .transition(carry.lane, carry.before)
                    .expect("lane oracle")
            );
        }
    }

    // 3. First-pick read `/14` replay: exact raw picks, addresses, and out carry words.
    let (plain, writer_table) = plain_synthetic_fbs_and_table();
    let q_gates = vec![Q31_IDENTITY; 8];
    let k_gates: Vec<Q31Quaternion> = (0..8)
        .map(|j| [Q31_ONE - (j as i64) * 1024, (j as i64) * 2048, 0, 0])
        .collect();
    let weights = vec![1_i64 << 24; 32];
    let read = Q31FirstPickRead::new(q_gates, k_gates, weights, 4).expect("read");
    let artifact_14 =
        add_q31_read_uorlut01(&plain, &Q31SessionRead::new(1, read.clone())).expect("export /14");
    let read_summary = compare_q31_read_replay(&read, &artifact_14, &prompt, &writer_table);
    assert!(
        read_summary.mismatches.is_empty(),
        "{:?}",
        read_summary.mismatches
    );
    assert_eq!(read_summary.read_picks_compared, prompt.len());
    assert!(read_summary.address_words_compared >= prompt.len() * 32);
    assert_eq!(read_summary.read_out_words_compared, prompt.len() * 32);
}

fn patch_u32_table_word(artifact: &[u8], table_name: &str, index: usize, value: u32) -> Vec<u8> {
    let mut patched = artifact.to_vec();
    let header_len = u64::from_le_bytes(patched[8..16].try_into().expect("header len")) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&patched[16..16 + header_len]).expect("header json");
    let span = header["tables"]
        .as_array()
        .expect("tables")
        .iter()
        .find(|t| t["name"].as_str() == Some(table_name))
        .expect("target table")["span"]
        .clone();
    let data_start = (16 + header_len).div_ceil(64) * 64;
    let offset = data_start + span["offset"].as_u64().expect("offset") as usize + index * 4;
    patched[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    patched
}

#[test]
fn planted_faults_fail_through_shared_replay_comparator() {
    let roots = [[0u8, 1], [2, 3], [4, 5]];
    let samples = all_prefixes(3, 3);
    let route_bytes = route_fixture_bytes(&roots, vec![1, 2, 3, 4]);
    let flock_bytes = flock_fixture_bytes(&roots, vec![1, 0, -1, 1]);
    let snap = synthetic_fbs_trunk(20_261_115);
    let prompt = [3u32, 5, 8, 13, 21, 34, 55, 89];

    // F1: Operator shift fault (q31 rotate shifts >> 30 instead of >> 31).
    let mut trained = Q31HamiltonModel::new(Q31_IDENTITY, 8).expect("model");
    let values = vec![[1 << 30, -(1 << 29), 1 << 28, -(1 << 27)]; 8];
    let targets = vec![[1_518_500_249, 1_518_500_249, 0, 0]; 8];
    for _ in 0..8 {
        trained.step_reference(&values, &targets).expect("step");
    }
    let artifact_12 =
        export_fbs_trunk_uorlut01_with_q31(&snap, FbsExportSnaps::SNAPPED, &trained.to_bytes())
            .expect("export /12");
    let f1 = compare_q31_hamilton_replay(&trained, &artifact_12, &prompt, true);
    assert!(
        f1.mismatches
            .iter()
            .any(|m| m.field == ParityField::CarryState),
        "F1 operator fault went undetected"
    );

    // F2: Scale fault (gain_shift 4 -> 5 in serving Q31FirstPickRead).
    let (plain, writer_table) = plain_synthetic_fbs_and_table();
    let base_read = Q31FirstPickRead::new(
        vec![Q31_IDENTITY; 8],
        vec![Q31_IDENTITY; 8],
        vec![1_i64 << 25; 32],
        4,
    )
    .expect("base read");
    let fault_scale_read = Q31FirstPickRead::new(
        vec![Q31_IDENTITY; 8],
        vec![Q31_IDENTITY; 8],
        vec![1_i64 << 25; 32],
        5,
    )
    .expect("fault scale read");
    let fault_scale_artifact =
        add_q31_read_uorlut01(&plain, &Q31SessionRead::new(1, fault_scale_read))
            .expect("export fault /14");
    let f2 = compare_q31_read_replay(&base_read, &fault_scale_artifact, &prompt, &writer_table);
    assert!(
        f2.mismatches
            .iter()
            .any(|m| m.field == ParityField::CarryState),
        "F2 scale fault went undetected"
    );

    // F3: Route fault (single discrete.routes port changed on serving copy).
    let fault_route_bytes = route_fixture_bytes(&roots, vec![2, 2, 3, 4]);
    let f3 = compare_discrete_replay(
        &route_bytes,
        &fault_route_bytes,
        &samples,
        ParityField::RouteChoices,
        true,
        false,
    )
    .expect("f3 compare");
    assert!(
        f3.mismatches
            .iter()
            .any(|m| m.field == ParityField::RouteChoices),
        "F3 route fault went undetected"
    );

    // F4: Carry fault (a) Flock gate trit 1 -> -1, and (b) Q31 gate coordinate change.
    let fault_flock_bytes = flock_fixture_bytes(&roots, vec![-1, 0, -1, 1]);
    let f4a = compare_discrete_replay(
        &flock_bytes,
        &fault_flock_bytes,
        &samples,
        ParityField::ReadChoices,
        true,
        true,
    )
    .expect("f4a compare");
    assert!(
        f4a.mismatches
            .iter()
            .any(|m| m.field == ParityField::CarryState),
        "F4a flock gate carry fault went undetected"
    );
    let mut mutated_gate = trained.gate();
    mutated_gate[1] += 1 << 24;
    let mutated_model = Q31HamiltonModel::new(mutated_gate, trained.step_shift()).expect("mutated");
    let f4b = compare_q31_hamilton_replay(&mutated_model, &artifact_12, &prompt, false);
    assert!(
        f4b.mismatches
            .iter()
            .any(|m| m.field == ParityField::CarryState),
        "F4b Q31 carry gate fault went undetected"
    );

    // F5a: Read-choice & key-address fault through shared `compare_q31_read_replay` comparator,
    // plus a 1-LSB key-gate tie-sensitivity check on `Q31FirstPickRead`.
    let fault_key_gates: Vec<Q31Quaternion> = (0..8)
        .map(|j| {
            let mut g = [0_i64; 4];
            g[(j + 1) % 4] = Q31_ONE;
            g[j % 4] = -(1_i64 << 28);
            g
        })
        .collect();
    let fault_key_read = Q31FirstPickRead::new(
        vec![Q31_IDENTITY; 8],
        fault_key_gates,
        vec![1_i64 << 25; 32],
        4,
    )
    .expect("fault key read");
    let fault_pick_artifact =
        add_q31_read_uorlut01(&plain, &Q31SessionRead::new(1, fault_key_read))
            .expect("export F5a /14");
    let f5a = compare_q31_read_replay(&base_read, &fault_pick_artifact, &prompt, &writer_table);
    assert!(
        f5a.mismatches
            .iter()
            .any(|m| m.field == ParityField::ReadChoices),
        "F5a Q31 read-choice fault went undetected in shared comparator"
    );
    assert!(
        f5a.mismatches
            .iter()
            .any(|m| m.field == ParityField::MemoryAddresses),
        "F5a Q31 key-address fault went undetected in shared comparator"
    );

    let near = 3_i64 << 29;
    let mut row_a = vec![0_i64; 16];
    row_a[0] = near + 1000;
    let mut row_b = vec![0_i64; 16];
    row_b[0] = near - 1000;
    let mut row_q = vec![0_i64; 16];
    row_q[0] = near;
    let stream = [row_a, row_b, row_q].concat();
    let table = vec![100_i64; 4 * 16];
    let ids = [0u32, 1, 2];
    let clean_pick_read = Q31FirstPickRead::new(
        vec![Q31_IDENTITY; 4],
        vec![Q31_IDENTITY; 4],
        vec![Q31_ONE; 16],
        0,
    )
    .expect("clean tie read");
    let mut nudged_keys = vec![Q31_IDENTITY; 4];
    nudged_keys[0][0] = Q31_ONE - 1;
    let fault_pick_read =
        Q31FirstPickRead::new(vec![Q31_IDENTITY; 4], nudged_keys, vec![Q31_ONE; 16], 0)
            .expect("nudged tie read");
    let clean_picks = clean_pick_read
        .apply(&stream, &ids, &table, 1, 3)
        .expect("clean apply")
        .picks;
    let mut fault_history = Q31ReadHistory::new();
    let fault_picks: Vec<u32> = (0..3)
        .map(|t| {
            let (pick, _) = fault_pick_read
                .step(
                    &mut fault_history,
                    &stream[t * 16..(t + 1) * 16],
                    ids[t],
                    &table,
                )
                .expect("serving fault step");
            pick
        })
        .collect();
    assert_eq!(clean_picks, vec![0, 1, 2]);
    assert_eq!(fault_picks, vec![0, 1, 1]);
    assert_ne!(
        clean_picks, fault_picks,
        "F5a 1-LSB tie-sensitivity check must move pick"
    );

    // F5b: Read-choice fault on Flock active trace (distinct root alters kept/action read trace).
    let mut altered_flock_roots = roots;
    altered_flock_roots[0][0] = 6;
    let fault_flock_read_bytes = flock_fixture_bytes(&altered_flock_roots, vec![1, 0, -1, 1]);
    let f5b = compare_discrete_replay(
        &flock_bytes,
        &fault_flock_read_bytes,
        &samples,
        ParityField::ReadChoices,
        true,
        true,
    )
    .expect("f5b compare");
    assert!(
        f5b.mismatches
            .iter()
            .any(|m| m.field == ParityField::ReadChoices),
        "F5b Flock read-choice fault went undetected"
    );

    // F6: Memory-address fault (corrupt discrete.cells[0] fails closed on serving load).
    let tampered_cell_bytes = patch_u32_table_word(&route_bytes, "discrete.cells", 0, 99);
    let f6_err = compare_discrete_replay(
        &route_bytes,
        &tampered_cell_bytes,
        &samples,
        ParityField::RouteChoices,
        true,
        false,
    )
    .expect_err("F6 memory-address corruption must refuse at load");
    assert!(
        f6_err.contains("serving loader refused artifact"),
        "unexpected F6 error: {f6_err}"
    );

    // F7: Emitted token ID fault (swap tokens 0 and 1 in codebook + cells).
    let swapped_roots = [[2u8, 3], [0, 1], [4, 5]];
    let swapped_token_bytes = route_fixture_bytes(&swapped_roots, vec![1, 2, 3, 4]);
    let f7 = compare_discrete_replay(
        &route_bytes,
        &swapped_token_bytes,
        &samples,
        ParityField::RouteChoices,
        true,
        false,
    )
    .expect("f7 compare");
    assert!(
        f7.mismatches
            .iter()
            .any(|m| m.field == ParityField::TokenIds),
        "F7 emitted-token fault went undetected"
    );

    println!(
        "native_space_state_contract_faults F1_operator={} F2_scale={} F3_route={} F4a_flock={} F4b_q31={} F5a_q31_pick_mismatches={} F5a_q31_addr_mismatches={} F5a_tie={:?}->{:?} F5b_flock_read={} F6_cell_refused=true F7_token={}",
        f1.mismatches.len(),
        f2.mismatches.len(),
        f3.mismatches.len(),
        f4a.mismatches.len(),
        f4b.mismatches.len(),
        f5a.mismatches.iter().filter(|m| m.field == ParityField::ReadChoices).count(),
        f5a.mismatches.iter().filter(|m| m.field == ParityField::MemoryAddresses).count(),
        clean_picks,
        fault_picks,
        f5b.mismatches.iter().filter(|m| m.field == ParityField::ReadChoices).count(),
        f7.mismatches.iter().filter(|m| m.field == ParityField::TokenIds).count(),
    );
}
