use crate::{gpu, plan, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, num::NonZeroU64, path::Path, time::Instant};
use uor_r4_discrete::{
    corpus::{sha256, CorpusSource, FilePin, PreparedCorpus, Split},
    d11_export::{export_d11, validate_and_extract_cells},
    device::Device,
    snapshot::Snapshot,
    training::{
        visible_answer_admission, visible_answer_finite_optimizer, PreparedDriver,
        VisibleAnswerAdmission,
    },
    trunk::{build_trunk_norm_frames, build_trunk_step_table, export_and_verify_trunk_artifact},
};
use uor_r4_integer::discrete_d11::IntegerDiscreteModel;

pub fn fit(plan_path: &Path, go_path: &Path, go_sha: &str, out: &Path) -> Result<()> {
    let go_bytes = fs::read(go_path)?;
    if sha256(&go_bytes) != go_sha {
        return Err("GO digest differs from independently supplied approval".into());
    }
    let go: Value = serde_json::from_slice(&go_bytes)?;
    let plan_bytes = fs::read(plan_path)?;
    let plan_sha = sha256(&plan_bytes);
    let plan: plan::Plan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != "uor-r4.codex-4312.train-plan/1"
        || plan.group_count != 4060
        || go["plan_sha256"].as_str() != Some(plan_sha.as_str())
        || go["donor_manifest_sha256"].as_str() != Some(plan.donor_manifest.sha256.as_str())
        || go["reasoning_first_lineage_sha256"].as_str()
            != Some(plan.arms[0].lineage_sha256.as_str())
        || go["answer_first_lineage_sha256"].as_str() != Some(plan.arms[1].lineage_sha256.as_str())
        || go["full_targets"].as_u64() != Some(plan.full_targets)
        || go["geometry_targets"].as_u64() != Some(plan.geometry_targets)
        || go["geometry_batches"].as_u64() != Some(plan.geometry_batches)
        || plan.arms.iter().zip(plan::ARMS).any(|(arm, expected)| {
            arm.name != expected
                || arm.full.expected_eligible_targets != plan.full_targets
                || arm.full.tokens != arm.geometry.train.tokens
        })
    {
        return Err("paired training plan differs from the reviewed GO or arm inventory".into());
    }
    plan.donor_manifest.read()?;
    let drivers = [
        PreparedDriver::load(plan.arms[0].geometry.clone())?,
        PreparedDriver::load(plan.arms[1].geometry.clone())?,
    ];
    let admissions = [
        visible_answer_admission(
            &drivers[0],
            &go_bytes,
            go_sha,
            &plan.arms[0].geometry_order.read()?,
        )?,
        visible_answer_admission(
            &drivers[1],
            &go_bytes,
            go_sha,
            &plan.arms[1].geometry_order.read()?,
        )?,
    ];
    fs::create_dir(out)?;
    let models = [drivers[0].new_model(0)?, drivers[1].new_model(0)?];
    let states = [models[0].snapshot()?, models[1].snapshot()?];
    let hashes = [
        sha256(&serde_json::to_vec(&states[0])?),
        sha256(&serde_json::to_vec(&states[1])?),
    ];
    if hashes[0] != hashes[1] || states.iter().any(|state| state.epoch != 0) {
        return Err("arm initialization differs or is already trained".into());
    }
    plan::write_json(
        &out.join("initialization.json"),
        &json!({
            "snapshot_sha256": hashes[0],
            "epoch": 0,
            "seed": plan::SEED,
            "codebook_initialization": "seeded_device_v1",
            "go_sha256": go_sha,
            "paired_plan_sha256": plan_sha,
            "donor_manifest_sha256": plan.donor_manifest.sha256,
        }),
    )?;

    let mut outcomes = Vec::with_capacity(2);
    let mut digests = Vec::with_capacity(2);
    for (arm, (driver, (admission, model))) in plan
        .arms
        .iter()
        .zip(drivers.iter().zip(admissions.iter().zip(models)))
    {
        let dir = out.join(&arm.name);
        fs::create_dir(&dir)?;
        let (outcome, sections) = fit_arm(
            arm, driver, admission, model, &states[0], &plan, go_sha, &plan_sha, &dir,
        )?;
        outcomes.push(outcome);
        digests.push(sections);
    }
    for field in [
        "batches",
        "geometry_targets",
        "coordinate_trials",
        "row_coordinate_trials",
        "full_targets",
    ] {
        if outcomes[0][field] != outcomes[1][field] {
            return Err(format!("unmatched arm dose at {field}").into());
        }
    }
    let changed_sections = digests[0]
        .iter()
        .filter_map(|(name, hash)| (digests[1].get(name) != Some(hash)).then_some(name))
        .collect::<Vec<_>>();
    plan::write_json(
        &out.join("training-pair.json"),
        &json!({
            "schema": "uor-r4.codex-4312.training-pair/1",
            "arms": outcomes,
            "go_sha256": go_sha,
            "plan_sha256": plan_sha,
            "initialization_snapshot_sha256": hashes[0],
            "matched_budget": true,
            "different_serialized_native_sections": changed_sections,
            "kernels": gpu::kernel_digests(),
        }),
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_arm(
    arm: &plan::Arm,
    driver: &PreparedDriver,
    admission: &VisibleAnswerAdmission,
    mut model: Device,
    initial: &Snapshot,
    plan: &plan::Plan,
    go_sha: &str,
    plan_sha: &str,
    out: &Path,
) -> Result<(Value, BTreeMap<String, String>)> {
    let start = Instant::now();
    let mut cursor = driver.initial_cursor();
    let mut block_receipts = Vec::new();
    let mut coordinate_trials = 0u64;
    let mut row_coordinate_trials = 0u64;
    while cursor.completed_batches < plan.geometry_batches {
        let this_step = (plan.geometry_batches - cursor.completed_batches).min(16);
        let (new_cursor, receipts) = driver.run_visible_answer_transitions(
            &mut model,
            cursor,
            admission,
            NonZeroU64::new(this_step).ok_or("empty block dose")?,
        )?;
        for receipt in &receipts {
            let attempted = u64::from(receipt.k);
            let first = receipt.cursor_before.next_target;
            let rows = (plan.geometry_targets - first).min(u64::from(plan::BATCH));
            coordinate_trials += attempted;
            row_coordinate_trials += attempted * rows;
        }
        cursor = new_cursor;
        block_receipts.extend(receipts);
        eprintln!(
            "{} native geometry {}/{} batches, {} accepted GPU updates",
            arm.name, cursor.completed_batches, plan.geometry_batches, cursor.accepted_updates
        );
        if cursor.completed_batches % 128 == 0 {
            plan::write_json(
                &out.join(format!("intermediate-{:05}.json", cursor.completed_batches)),
                &driver.checkpoint(&model, cursor)?,
            )?;
        }
    }
    if cursor.corpus_pass != 1 || cursor.next_target != 0 {
        return Err(format!(
            "{} did not complete its full native geometry pass",
            arm.name
        )
        .into());
    }
    let geometry_ms = start.elapsed().as_millis();
    let checkpoint = driver.checkpoint(&model, cursor)?;
    let snapshot = model.snapshot()?;
    plan::write_json(&out.join("geometry-checkpoint.json"), &checkpoint)?;
    plan::write_json(&out.join("geometry-batches.json"), &block_receipts)?;
    drop(block_receipts);
    let geometry_replay = verify_geometry(&model, &snapshot, driver.train())?;
    drop(model);
    let corpus = PreparedCorpus::load(&arm.full, plan::VOCAB, &plan::bos())?;
    let (steps, norms) = recurrence_frames(&snapshot)?;
    let (tables, count_receipt) = gpu::fit(
        &snapshot,
        &corpus,
        &steps,
        &norms,
        Some(&arm.targets_map.read()?),
    )?;
    let start_pack = Instant::now();
    let cells = validate_and_extract_cells(&snapshot)?
        .into_iter()
        .map(|cell| cell as u16)
        .collect::<Vec<_>>();
    let (old_manifest, old_bytes) =
        export_and_verify_trunk_artifact(&snapshot, &cells, &steps, &norms, &tables)?;
    drop(tables);
    let bytes = bind_artifact(&old_bytes, arm, plan, go_sha, plan_sha)?;
    drop(old_bytes);
    let sections = section_hashes(&bytes)?;
    let path = out.join("artifact.d11");
    plan::write_new(&path, &bytes)?;
    let artifact_sha = sha256(&bytes);
    drop(bytes);
    let reopened = IntegerDiscreteModel::load(&path)?;
    if reopened.artifact_sha256() != artifact_sha
        || reopened.tokenizer_sha256() != Some(arm.geometry.tokenizer.sha256.as_str())
        || !reopened.is_trunk_d11()
        || reopened.shape().epoch != cursor.accepted_updates
    {
        return Err("ordinary integer client refused trained state or an identity binding".into());
    }
    let receipts = json!({
        "schema": "uor-r4.codex-4312.trained-checkpoint/1",
        "arm": arm.name,
        "artifact_d11_sha256": artifact_sha,
        "deployed_bytes": fs::metadata(&path)?.len(),
        "arm_lineage_sha256": arm.lineage_sha256,
        "checkpoint_model_sha256": checkpoint.model_sha256,
        "tokenizer_sha256": arm.geometry.tokenizer.sha256,
        "full_targets": plan.full_targets,
        "geometry_targets": plan.geometry_targets,
        "batches": cursor.completed_batches,
        "coordinate_trials": coordinate_trials,
        "row_coordinate_trials": row_coordinate_trials,
        "accepted_updates": cursor.accepted_updates,
        "committed_moves": cursor.block.ok_or("missing block cursor")?.committed_moves,
        "geometry_ms": geometry_ms,
        "geometry_replay": geometry_replay,
        "serialize_and_cold_open_ms": start_pack.elapsed().as_millis(),
        "count_fit": count_receipt,
        "all_19_sections_native_round_trip": old_manifest.round_trip_verified,
        "final_snapshot_kernel_sha256": snapshot.kernel_sha256,
        "final_snapshot_geometry_sha256": snapshot.geometry_sha256,
        "learned_geometry_changed_words": {
            "token_vectors": differing_words(&initial.table, &snapshot.table)?,
            "routes": differing_words(&initial.routes, &snapshot.routes)?,
            "gates": differing_words(&initial.gates, &snapshot.gates)?,
        },
        "native_section_sha256": sections,
    });
    plan::write_json(&out.join("training-receipt.json"), &receipts)?;
    Ok((receipts, sections))
}

fn differing_words<T: Eq>(initial: &[T], fitted: &[T]) -> Result<usize> {
    if initial.len() != fitted.len() {
        return Err("learned geometry shape differs from initialization".into());
    }
    Ok(initial
        .iter()
        .zip(fitted)
        .filter(|(old, new)| old != new)
        .count())
}

fn section_hashes(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let header_len = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into()?))?;
    let header: Value = serde_json::from_slice(&bytes[16..16 + header_len])?;
    let body = bytes
        .get((16 + header_len).next_multiple_of(64)..)
        .ok_or("missing D11 body")?;
    let mut sections = BTreeMap::new();
    for table in header["tables"].as_array().ok_or("missing D11 sections")? {
        let name = table["name"].as_str().ok_or("missing D11 section name")?;
        let start = usize::try_from(
            table["span"]["offset"]
                .as_u64()
                .ok_or("missing D11 offset")?,
        )?;
        let len = usize::try_from(
            table["span"]["bytes"]
                .as_u64()
                .ok_or("missing D11 length")?,
        )?;
        let end = start.checked_add(len).ok_or("D11 section range overflow")?;
        let digest = sha256(body.get(start..end).ok_or("D11 section outside body")?);
        if sections.insert(name.to_owned(), digest).is_some() {
            return Err("duplicate native output section".into());
        }
    }
    if sections.len() != 19 {
        return Err("expected all 19 native output sections".into());
    }
    Ok(sections)
}

/// Compare the training device's typed forward and route/carry trace with an
/// independently loaded base D11 artifact, before fitting its output tables.
fn verify_geometry(model: &Device, snapshot: &Snapshot, corpus: &PreparedCorpus) -> Result<Value> {
    let firmware = IntegerDiscreteModel::from_bytes(&export_d11(snapshot)?)?;
    let total = corpus.eligible_len();
    let probes = total.min(64);
    let mut windows = Vec::with_capacity(probes);
    let mut targets = Vec::with_capacity(probes);
    for sample in 0..probes {
        let mut batch = corpus.batch((sample * total / probes) as u64, plan::CONTEXT, 1)?;
        windows.push(batch.windows.pop().ok_or("geometry probe window absent")?);
        targets.push(batch.targets.pop().ok_or("geometry probe target absent")?);
    }
    let carried = model.carry_windows(&windows)?;
    let forward = model.forward_traced(&carried, &targets)?;
    let actual_route = forward
        .read_trace()?
        .ok_or("geometry device omitted route trace")?;
    let actual_vectors = forward.vectors()?;
    let mut expected_route = Vec::new();
    let mut expected_vectors = Vec::new();
    let mut expected_predictions = Vec::new();
    let mut expected_cells = Vec::with_capacity(probes * 16);
    for ids in &windows {
        let prefix = ids.iter().map(|id| id.token).collect::<Vec<_>>();
        let clean = firmware.evaluate_prefix_traced(&prefix)?;
        expected_route.extend(clean.trace_words);
        expected_vectors.extend(clean.evaluation.read_vector);
        expected_predictions.push(clean.evaluation.prediction);
        expected_cells.extend(clean.evaluation.admitted_cells.iter().copied());
        expected_cells.extend(std::iter::repeat_n(
            u32::MAX,
            16 - clean.evaluation.admitted_cells.len(),
        ));
    }
    if actual_route != expected_route
        || actual_vectors != expected_vectors
        || forward.predictions != expected_predictions
        || forward.admitted_cells != expected_cells
    {
        let first_route_diff = actual_route
            .iter()
            .zip(&expected_route)
            .position(|(a, b)| a != b);
        return Err(format!(
            "training-device native trace differs from clean-room D11 (route words {} vs {}, vectors {} vs {}, first route difference {first_route_diff:?})",
            actual_route.len(), expected_route.len(), actual_vectors.len(), expected_vectors.len()
        ).into());
    }
    let digest_words = |words: Vec<u32>| {
        sha256(
            &words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    Ok(json!({
        "probes": probes,
        "route_and_carry_words": actual_route.len(),
        "read_vector_words": actual_vectors.len(),
        "route_and_carry_sha256": digest_words(actual_route),
        "read_vector_sha256": digest_words(actual_vectors.into_iter().map(|word| word as u32).collect()),
        "prediction_sha256": digest_words(forward.predictions),
        "admitted_cell_sha256": digest_words(forward.admitted_cells),
    }))
}

fn recurrence_frames(
    snapshot: &Snapshot,
) -> Result<(
    uor_r4_discrete::position_mechanism::PositionStepTable,
    uor_r4_discrete::normalization_mechanism::NormFrameTable,
)> {
    let offset = snapshot.seed.checked_sub(1).ok_or("nonpositive seed")? as usize;
    Ok((
        build_trunk_step_table(snapshot.spec.blocks, offset)?,
        build_trunk_norm_frames(snapshot.spec.blocks, offset)?,
    ))
}

fn bind_artifact(
    bytes: &[u8],
    arm: &plan::Arm,
    plan: &plan::Plan,
    go_sha: &str,
    plan_sha: &str,
) -> Result<Vec<u8>> {
    let header_len = usize::try_from(u64::from_le_bytes(bytes[8..16].try_into()?))?;
    let mut header: Value = serde_json::from_slice(&bytes[16..16 + header_len])?;
    header["tokenizer_sha256"] = json!(arm.geometry.tokenizer.sha256);
    header["training_binding"] = json!({
        "schema": "uor-r4.codex-4312.artifact-binding/1",
        "arm": arm.name,
        "donor_manifest_sha256": plan.donor_manifest.sha256,
        "corpus_tokens_sha256": arm.full.tokens.sha256,
        "full_target_mask_sha256": arm.full.mask.sha256,
        "full_target_event_order_sha256": arm.targets_map.sha256,
        "geometry_target_mask_sha256": arm.geometry.train.mask.sha256,
        "geometry_event_order_sha256": arm.geometry_order.sha256,
        "training_plan_sha256": plan_sha,
        "go_sha256": go_sha,
        "driver_lineage_sha256": arm.lineage_sha256,
        "protocol": { "bos_id": 0, "eos_id": 1, "user_marker": "\nUser:", "assistant_marker": "\nAssistant:" },
    });
    let header = serde_json::to_vec(&header)?;
    let body_offset = (16 + header_len).next_multiple_of(64);
    let mut output =
        Vec::with_capacity((16 + header.len()).next_multiple_of(64) + bytes.len() - body_offset);
    output.extend_from_slice(&bytes[..8]);
    output.extend_from_slice(&(header.len() as u64).to_le_bytes());
    output.extend_from_slice(&header);
    output.resize(output.len().next_multiple_of(64), 0);
    output.extend_from_slice(&bytes[body_offset..]);
    Ok(output)
}

pub fn finite(out: &Path) -> Result<()> {
    fs::create_dir(out)?;
    let mut tokens = vec![0u16, 13, 18, 13, 42, 1, 0, 0];
    tokens.extend((0..215).map(|i| 20 + (i % 7)));
    tokens.push(1);
    let mut mask = tokens
        .iter()
        .enumerate()
        .map(|(at, &token)| u8::from(token != 0 && (at < 8 || at % 17 == 0)))
        .collect::<Vec<_>>();
    if let Some(last) = mask.last_mut() {
        *last = 1;
    }
    let encoded = plan::uort(&tokens);
    let corpus = PreparedCorpus::from_bytes(
        CorpusSource {
            split: Split::Train,
            tokens: FilePin {
                path: out.join("fixture.tokens"),
                sha256: sha256(&encoded),
            },
            mask: FilePin {
                path: out.join("fixture.mask"),
                sha256: sha256(&mask),
            },
            expected_payload_tokens: tokens.len() as u64,
            expected_eligible_targets: mask.iter().map(|&word| u64::from(word)).sum(),
        },
        &encoded,
        &mask,
        plan::VOCAB,
        plan::bos(),
    )?;
    let (model, optimizer) = visible_answer_finite_optimizer()?;
    eprintln!("finite GPU optimizer positive and identity controls passed");
    let snapshot = model.snapshot()?;
    let geometry_replay = verify_geometry(&model, &snapshot, &corpus)?;
    eprintln!("finite post-update CUDA and D11 geometry replay matched");
    let (steps, norms) = recurrence_frames(&snapshot)?;
    let (tables, receipt) = gpu::fit(&snapshot, &corpus, &steps, &norms, None)?;
    let cells = validate_and_extract_cells(&snapshot)?
        .into_iter()
        .map(|cell| cell as u16)
        .collect::<Vec<_>>();
    let (manifest, artifact) =
        export_and_verify_trunk_artifact(&snapshot, &cells, &steps, &norms, &tables)?;
    plan::write_json(
        &out.join("finite.json"),
        &json!({
            "status": "PASS",
        "optimizer": optimizer,
            "count_fit": receipt,
        "geometry_replay": geometry_replay,
            "artifact_manifest": manifest,
            "clean_room_schema": IntegerDiscreteModel::from_bytes(&artifact)?.artifact_schema(),
        }),
    )
}
