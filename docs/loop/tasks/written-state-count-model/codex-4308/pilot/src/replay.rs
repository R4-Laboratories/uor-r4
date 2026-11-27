#![forbid(unsafe_code)]

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    io::{self, Write},
    path::Path,
    process::Command,
    time::Instant,
};
use uor_r4_count_runtime::{
    p4_serving::{P4Session, P4State, P4_MAX_CONTEXT, P4_VOCAB},
    FrozenPickMixture,
};
use uor_r4_discrete::p5_brier_core::{extract_s1_from_uorlut01, rewrite_uorlut01_s1, P5S1State};
use uor_r4_integer::discrete_d11::{
    count_deployment::{
        load_client_artifact, write_p4_deployment, ClientArtifactFile, DiscreteCountModel, P4,
        SCHEMA_P4,
    },
    IntegerDiscreteModel, MAGIC,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const BASELINE: &str = "e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776";
const TOKENIZER: &str = "a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377";
const A1S_BINDING_END: usize = 44;
const WITNESS_LIMIT: usize = 8;
const DEPLOYMENT_CAP: u64 = 4_000_000_000;
const TOKENIZER_BYTES: u64 = 265_393;

fn fail(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, message.into()).into()
}

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(fail(message));
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn checkpoint_for_replay(
    baseline: &P5S1State,
    trained: &P5S1State,
    minimum_epoch: u64,
) -> Result<()> {
    require(
        trained.parent_trunk_sha256 == baseline.parent_trunk_sha256,
        "P5 checkpoint parent does not match the deployed trunk",
    )?;
    require(
        trained.vocab == baseline.vocab && trained.blocks == baseline.blocks,
        "P5 checkpoint shape does not match the deployed trunk",
    )?;
    require(
        trained.epoch >= baseline.epoch && trained.epoch >= minimum_epoch,
        "P5 checkpoint epoch is stale",
    )?;
    require(
        trained.epoch > baseline.epoch || trained.roots == baseline.roots,
        "P5 checkpoint changed roots without advancing its epoch",
    )
}

fn refused<T>(result: Result<T>, message: &str) -> Result<String> {
    match result {
        Err(error) if error.to_string().contains(message) => Ok(error.to_string()),
        Err(error) => Err(fail(format!("expected refusal {message:?}, got {error}"))),
        Ok(_) => Err(fail(format!("accepted negative control {message:?}"))),
    }
}

// Released seed1 resolves its tokenizer by the original artifact digest. Copy
// that existing binding into the rewritten header so the new digest serves text.
fn pin_tokenizer(mut trunk: Vec<u8>, tokenizer_sha: &str) -> Result<Vec<u8>> {
    let preamble = trunk.get(..16).ok_or_else(|| fail("truncated D11"))?;
    require(&preamble[..8] == MAGIC, "bad D11 magic")?;
    let len = usize::try_from(u64::from_le_bytes(preamble[8..16].try_into()?))?;
    let header_end = 16usize
        .checked_add(len)
        .ok_or_else(|| fail("D11 header overflow"))?;
    let original = trunk
        .get(16..header_end)
        .ok_or_else(|| fail("truncated D11 header"))?;
    let mut header: Value = serde_json::from_slice(original)?;
    let header = header
        .as_object_mut()
        .ok_or_else(|| fail("expected D11 header object"))?;
    if let Some(existing) = header.get("tokenizer_sha256") {
        require(
            existing.as_str() == Some(tokenizer_sha),
            "D11 tokenizer binding mismatch",
        )?;
        return Ok(trunk);
    }
    header.insert("tokenizer_sha256".into(), json!(tokenizer_sha));
    let json = serde_json::to_vec(header)?;
    let old_data_start = header_end.next_multiple_of(64);
    require(
        old_data_start <= trunk.len(),
        "truncated D11 header padding",
    )?;
    let new_data_start = (16 + json.len()).next_multiple_of(64);
    let mut prefix = Vec::with_capacity(new_data_start);
    prefix.extend_from_slice(MAGIC);
    prefix.extend_from_slice(&(json.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&json);
    prefix.resize(new_data_start, 0);
    trunk.splice(..old_data_start, prefix);
    Ok(trunk)
}

// Read component boundaries after the ordinary loader has validated every
// section's length, framing and digest. Borrow the original multi-GB counts.
fn sections(state: &[u8]) -> Result<[&[u8]; 7]> {
    const HEADER: usize = 8 + 7 * 8 + 7 * 32;
    let (header, mut body) = state
        .split_at_checked(HEADER)
        .ok_or_else(|| fail("short P4 state"))?;
    require(&header[..8] == b"UORP4A01", "wrong P4 state magic")?;
    let mut parts = [&[][..]; 7];
    for (dest, word) in parts.iter_mut().zip(header[8..64].chunks_exact(8)) {
        let len = usize::try_from(u64::from_le_bytes(word.try_into()?))?;
        (*dest, body) = body
            .split_at_checked(len)
            .ok_or_else(|| fail("short P4 component"))?;
    }
    require(body.is_empty(), "trailing P4 state bytes")?;
    Ok(parts)
}

fn rebind_a1s(
    manifest: &[u8],
    weights: &[u8],
    parent: &str,
    candidate: &str,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut fit: Value = serde_json::from_slice(manifest)?;
    let binding = fit
        .pointer_mut("/prepared/trunk_sha256")
        .ok_or_else(|| fail("missing P4 fit trunk binding"))?;
    require(
        binding.as_str() == Some(parent),
        "wrong original P4 fit trunk binding",
    )?;
    *binding = json!(candidate);
    let rebound_fit = serde_json::to_vec(&fit)?;
    let original = FrozenPickMixture::from_bytes(weights, Sha256::digest(manifest).into())?;
    let mut rebound_weights = weights.to_vec();
    rebound_weights[12..A1S_BINDING_END].copy_from_slice(&Sha256::digest(&rebound_fit));
    let rebound =
        FrozenPickMixture::from_bytes(&rebound_weights, Sha256::digest(&rebound_fit).into())?;
    require(
        rebound.rule() == original.rule()
            && rebound.weights() == original.weights()
            && rebound_weights[A1S_BINDING_END..] == weights[A1S_BINDING_END..],
        "A1s raw weight words changed during manifest rebinding",
    )?;
    Ok((rebound_fit, rebound_weights))
}

fn with_deployment<T>(
    path: &Path,
    expected_artifact: Option<&str>,
    action: impl FnOnce(DiscreteCountModel<'_>, &[u8], Value) -> Result<T>,
) -> Result<T> {
    let begin = Instant::now();
    let ClientArtifactFile::Embedded {
        model,
        state_bytes,
        expected_state_sha256,
        identity,
    } = load_client_artifact(path, false)?
    else {
        return Err(fail(
            "ordinary loader saw a legacy artifact instead of deployed P4",
        ));
    };
    if let Some(expected) = expected_artifact {
        require(
            identity.artifact_sha256 == expected,
            "wrong frozen P4 baseline artifact",
        )?;
    }
    let artifact_sha = identity.artifact_sha256.clone();
    let loaded =
        DiscreteCountModel::bind_embedded(model, &state_bytes, &expected_state_sha256, identity)?;
    require(
        loaded.p4.is_some(),
        "ordinary loader did not bind the P4 predictor",
    )?;
    let receipt = json!({
        "artifact_sha256": artifact_sha,
        "trunk_sha256": loaded.model.artifact_sha256(),
        "state_sha256": expected_state_sha256,
        "tokenizer_sha256": loaded.model.tokenizer_sha256(),
        "artifact_bytes": fs::metadata(path)?.len(),
        "load_and_bind_ns": begin.elapsed().as_nanos(),
        "components": loaded.p4.as_ref().unwrap().component_digests,
    });
    action(loaded, &state_bytes, receipt)
}

#[derive(PartialEq, Eq)]
struct Observation {
    trace: Value,
    masses: Vec<u64>,
}

fn observe(state: &P4State<'_>, prefix: &[u16]) -> Result<Observation> {
    let mut session = P4Session::new();
    session.sync_prompt(prefix)?;
    let (winner, trace) = session.step(state, true, true)?;
    let t = trace.ok_or_else(|| fail("ordinary P4 trace is missing"))?;
    let masses = &session.last_scores_q48()[1..];
    let max = masses
        .iter()
        .copied()
        .max()
        .ok_or_else(|| fail("missing P4 masses"))?;
    let first_max = 1 + masses.iter().position(|&mass| mass == max).unwrap();
    let ties = masses.iter().filter(|&&mass| mass == max).count();
    let mass_digest = |values: &[u64]| {
        let mut hash = Sha256::new();
        for &value in values {
            hash.update(value.to_le_bytes());
        }
        hex::encode(hash.finalize())
    };
    let queried_prefix: Vec<u8> = prefix.iter().flat_map(|id| id.to_le_bytes()).collect();
    require(
        usize::from(winner) == first_max
            && t.token_id == winner
            && t.mass_q48 == max
            && t.top_ties as usize == ties
            && t.mass_sha256 == mass_digest(masses)
            && t.prefix_sha256 == sha256(&queried_prefix)
            && t.context_tokens == prefix.len(),
        "ordinary P4 argmax, ties, raw scores or causal prefix disagree with its trace",
    )?;
    if prefix.len() < 3 {
        require(
            t.pick_pos.is_none()
                && t.pick_distance.is_none()
                && t.pick_successor.is_none()
                && t.pick_ties == 0
                && t.pick_class == state.a1s_weights.rule().classify(None),
            "one- or two-token witness reported a Pick source",
        )?;
    } else {
        let p = t
            .pick_pos
            .ok_or_else(|| fail("open witness has no Pick source"))? as usize;
        require(
            (1..prefix.len() - 1).contains(&p) && t.pick_successor == Some(prefix[p + 1]),
            "Pick trace does not point to the original causal successor",
        )?;
    }
    Ok(Observation {
        trace: json!({
            "token": t.token_id,
            "bucket": t.bucket,
            "highest_order": t.highest_order,
            "mass_q48": t.mass_q48,
            "top_ties": t.top_ties,
            "mass_sha256": t.mass_sha256,
            "prefix_sha256": t.prefix_sha256,
            "context_tokens": t.context_tokens,
            "pick_pos": t.pick_pos,
            "pick_distance": t.pick_distance,
            "pick_ties": t.pick_ties,
            "pick_successor": t.pick_successor,
            "pick_class": t.pick_class,
            "s3_pkey": t.s3_pkey,
            "s3_backoff_key": t.s3_backoff_key,
            "s3_mode": t.s3_mode,
            "s2_skip_keys": t.s2_skip_keys,
            "s2_skip_hit_mask": t.s2_skip_hit_mask,
            "cache_hit_state": t.cache_hit_state,
            "s4_role_scores": t.s4_role_scores,
            "s4_role_choice": t.s4_role_choice,
        }),
        masses: masses.to_vec(),
    })
}

fn validate_cli(
    record: &Value,
    artifact: &Value,
    prefix: &[u16],
    expected: &Observation,
) -> Result<()> {
    require(
        record["schema"] == "uor-r4.uor-chat-discrete-parity/1"
            && record["model_schema"] == SCHEMA_P4
            && record["read_enabled"] == true
            && record["artifact_sha256"] == artifact["artifact_sha256"]
            && record["p4_trunk_sha256"] == artifact["trunk_sha256"]
            && record["p4_state_sha256"] == artifact["state_sha256"]
            && record["p4_components"] == artifact["components"]
            && record["prompt_ids"] == json!(prefix)
            && record["generated_ids"] == json!([expected.trace["token"]])
            && record["p4_steps"] == json!([expected.trace]),
        "ordinary uor-chat disagrees with exact P4 serving reference",
    )
}

fn wrong_pick_trace(record: &Value) -> (Value, &'static str) {
    let mut planted = record.clone();
    let position = &mut planted["p4_steps"][0]["pick_pos"];
    if let Some(original) = position.as_u64() {
        *position = json!(original + 1);
        (planted, "off_by_one_original_position")
    } else {
        *position = json!(0);
        (planted, "impossible_position_on_no_pick_prefix")
    }
}

fn cli(
    chat: &Path,
    path: &Path,
    artifact: &Value,
    prefix: &[u16],
    expected: &Observation,
) -> Result<Value> {
    let csv = prefix
        .iter()
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let begin = Instant::now();
    let output = Command::new(chat)
        .env("UOR_P1_TIMING", "1")
        .arg("--discrete")
        .arg(path)
        .args(["--prompt-ids", &csv, "--max-tokens", "1", "--p1-trace"])
        .output()?;
    if !output.status.success() {
        return Err(fail(format!(
            "uor-chat {} failed: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let record: Value = serde_json::from_slice(&output.stdout)?;
    validate_cli(&record, artifact, prefix, expected)?;
    let stderr = String::from_utf8(output.stderr)?;
    let timing = stderr
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|entry| entry["schema"] == "uor-r4.p1-client-timing/1")
        .ok_or_else(|| fail("missing ordinary CLI load/decode timing"))?;
    Ok(json!({"wall_ns": begin.elapsed().as_nanos(), "timing": timing, "record": record}))
}

fn pack(trunk: &[u8], parts: [&[u8]; 7], path: &Path) -> Result<()> {
    let [fit, a1s, counts, s3, s2, s4, s6] = parts;
    write_p4_deployment(trunk, fit, a1s, counts, s3, s2, s4, s6, path)?;
    Ok(())
}

fn valid_prefix(prefix: &[u16]) -> Result<()> {
    require(
        (1..=P4_MAX_CONTEXT).contains(&prefix.len())
            && (prefix[0] == 0 || prefix.len() == P4_MAX_CONTEXT)
            && (prefix[0] == 0 || (2..P4_VOCAB as u16).contains(&prefix[0]))
            && prefix[1..]
                .iter()
                .all(|&id| (2..P4_VOCAB as u16).contains(&id)),
        "DEV serving witness must be an open BOS prefix or its exact slid 2048-token history",
    )
}

fn verify_state(model: &IntegerDiscreteModel, p4: &P4<'_>, state: &P5S1State) -> Result<()> {
    let roots: Vec<u32> = state
        .roots
        .chunks_exact(4)
        .map(|r| {
            u32::from(r[0]) | u32::from(r[1]) << 4 | u32::from(r[2]) << 8 | u32::from(r[3]) << 12
        })
        .collect();
    require(
        model.shape().epoch == state.epoch
            && model.packed_codebook_roots() == roots
            && model.cells() == state.derive_cells()?
            && p4.inner.codes == state.derive_pick_codes_4()?,
        "cold ordinary client loaded different S1 epoch, roots, cells or Identity Pick words",
    )
}

/// Deployment spot-check on registered DEV prefix IDs. The optional prefix was
/// chosen after GPU comparison and is reported separately from fixed witnesses.
pub fn run(
    p4_path: &Path,
    trunk_path: &Path,
    checkpoint_path: &Path,
    chat_path: &Path,
    witness_prefixes: &[Vec<u16>],
    known_changed_prefix: Option<&[u16]>,
    output_dir: &Path,
) -> Result<Value> {
    require(
        !witness_prefixes.is_empty() && witness_prefixes.len() <= WITNESS_LIMIT,
        "wrong fixed DEV serving witness count",
    )?;
    let prefixes: Vec<&[u16]> = witness_prefixes
        .iter()
        .map(Vec::as_slice)
        .chain(known_changed_prefix)
        .collect();
    for prefix in &prefixes {
        valid_prefix(prefix)?;
    }
    if let Some(selected) = known_changed_prefix {
        require(
            !witness_prefixes.iter().any(|prefix| prefix == selected),
            "GPU-selected serving prefix duplicates a fixed DEV prefix",
        )?;
    }
    fs::create_dir(output_dir)?;
    let candidate_output = output_dir.join("trained-p4.d11");
    let begin = Instant::now();
    let trunk = fs::read(trunk_path)?;
    let checkpoint = fs::read(checkpoint_path)?;
    let original = extract_s1_from_uorlut01(&trunk)?;
    let trained = P5S1State::from_bytes(&checkpoint)?;
    require(
        trained.encode_bytes()? == checkpoint,
        "S1 checkpoint readback differs from canonical bytes",
    )?;
    checkpoint_for_replay(&original, &trained, original.epoch)?;
    let accepted_updates = trained.epoch - original.epoch;
    let candidate_path = if accepted_updates == 0 {
        p4_path
    } else {
        candidate_output.as_path()
    };
    let baseline_trunk_sha = hex::encode(original.parent_trunk_sha256);
    let original_checkpoint = P5S1State::from_bytes(&original.encode_bytes()?)?;
    let mut wrong_parent = trained.clone();
    wrong_parent.parent_trunk_sha256[0] ^= 1;
    let wrong_parent = P5S1State::from_bytes(&wrong_parent.encode_bytes()?)?;
    let wrong = refused(
        checkpoint_for_replay(&original, &wrong_parent, original.epoch),
        "parent does not match",
    )?;
    let checkpoint_read_ns = begin.elapsed().as_nanos();
    let (rewritten, candidate_trunk_sha, rewrite_ns) = if accepted_updates == 0 {
        require(
            original_checkpoint == trained,
            "zero-update checkpoint differs from the original native state",
        )?;
        (None, baseline_trunk_sha.clone(), None)
    } else {
        let rewrite_begin = Instant::now();
        let rewritten = pin_tokenizer(rewrite_uorlut01_s1(&trunk, &trained)?, TOKENIZER)?;
        let trained_model = IntegerDiscreteModel::from_bytes(&rewritten)?;
        require(
            trained_model.tokenizer_sha256() == Some(TOKENIZER),
            "trained trunk lost original tokenizer binding",
        )?;
        (
            Some(rewritten),
            trained_model.artifact_sha256().to_owned(),
            Some(rewrite_begin.elapsed().as_nanos()),
        )
    };
    drop(trunk);

    let (before, baseline, pack_ns, restore_ns, negatives, weight_words_sha) = with_deployment(
        p4_path,
        Some(BASELINE),
        |mut loaded, state_bytes, baseline| {
            require(
                loaded.model.artifact_sha256() == baseline_trunk_sha
                    && loaded.model.tokenizer_sha256() == Some(TOKENIZER),
                "external seed trunk or tokenizer does not match baseline deployment",
            )?;
            let baseline_p4 = loaded
                .p4
                .as_mut()
                .ok_or_else(|| fail("baseline P4 state missing"))?;
            verify_state(&loaded.model, baseline_p4, &original)?;
            let before = prefixes
                .iter()
                .map(|prefix| observe(&baseline_p4.inner, prefix))
                .collect::<Result<Vec<_>>>()?;
            let parts = sections(state_bytes)?;
            let weight_words_sha = sha256(&parts[1][A1S_BINDING_END..]);
            let Some(rewritten) = rewritten.as_deref() else {
                let negatives = json!({
                    "wrong_checkpoint_parent": wrong,
                    "stale_checkpoint": null,
                    "trained_trunk_stale_manifest": null,
                    "trained_trunk_and_fit_stale_a1s_binding": null,
                    "skip_reason": "zero updates left the deployed trunk and all bindings identical",
                });
                return Ok((before, baseline, None, None, negatives, weight_words_sha));
            };
            let stale = refused(
                checkpoint_for_replay(&original, &original_checkpoint, trained.epoch),
                "epoch is stale",
            )?;
            let restore_start = Instant::now();
            let restored = rewrite_uorlut01_s1(rewritten, &original_checkpoint)?;
            let restored_state = extract_s1_from_uorlut01(&restored)?;
            require(
                restored_state.roots == original.roots && restored_state.epoch == original.epoch,
                "inverse restore failed to recover exact original roots and epoch",
            )?;
            let restored_model = IntegerDiscreteModel::from_bytes(&restored)?;
            require(
                restored_model.cells() == loaded.model.cells(),
                "restored cells differ from baseline",
            )?;
            baseline_p4.inner.codes = restored_model.identity_pick_codes()?;
            require(
                baseline_p4.inner.codes == original.derive_pick_codes_4()?,
                "restored Pick code words differ from baseline",
            )?;
            for (prefix, reference) in prefixes.iter().zip(&before) {
                require(
                    observe(&baseline_p4.inner, prefix)? == *reference,
                    "restoration failed bit-identical P4 replay",
                )?;
            }
            let restore_ns = restore_start.elapsed().as_nanos();
            drop(restored_model);
            drop(restored);

            let pack_begin = Instant::now();
            let (fit, a1s) = rebind_a1s(
                parts[0],
                parts[1],
                &baseline_trunk_sha,
                &candidate_trunk_sha,
            )?;
            let bad_path = output_dir.join("unexpected-accepted-negative.d11");
            let bad_manifest = refused(
                pack(rewritten, parts, &bad_path),
                "P4 fit trunk digest mismatch",
            )?;
            let mut rebound = parts;
            rebound[0] = &fit;
            let bad_weights = refused(
                pack(rewritten, rebound, &bad_path),
                "S5 mixture binding mismatch",
            )?;
            rebound[1] = &a1s;
            pack(rewritten, rebound, candidate_path)?;
            let negatives = json!({
                "stale_checkpoint": stale,
                "wrong_checkpoint_parent": wrong,
                "trained_trunk_stale_manifest": bad_manifest,
                "trained_trunk_and_fit_stale_a1s_binding": bad_weights,
            });
            Ok((
                before,
                baseline,
                Some(pack_begin.elapsed().as_nanos()),
                Some(restore_ns),
                negatives,
                weight_words_sha,
            ))
        },
    )?;
    drop(rewritten);
    let same_artifact_sha = (accepted_updates == 0).then_some(BASELINE);
    let (after, candidate) = with_deployment(
        candidate_path,
        same_artifact_sha,
        |loaded, bytes, receipt| {
            require(
                loaded.model.artifact_sha256() == candidate_trunk_sha
                    && loaded.model.tokenizer_sha256() == Some(TOKENIZER),
                "cold candidate trunk or tokenizer identity mismatch",
            )?;
            require(
                fs::metadata(candidate_path)?.len() + TOKENIZER_BYTES <= DEPLOYMENT_CAP,
                "deployed candidate plus tokenizer exceeds 4 GB",
            )?;
            let p4 = loaded
                .p4
                .as_ref()
                .ok_or_else(|| fail("candidate P4 state missing"))?;
            verify_state(&loaded.model, p4, &trained)?;
            for name in [
                "count_sha256",
                "s3_sha256",
                "s2_skip_sha256",
                "s4_sha256",
                "s6_sha256",
            ] {
                require(
                    receipt["components"][name] == baseline["components"][name],
                    "frozen P4 component changed during repack",
                )?;
            }
            require(
                sha256(&sections(bytes)?[1][A1S_BINDING_END..]) == weight_words_sha,
                "cold candidate A1s weight bytes changed",
            )?;
            let observations = prefixes
                .iter()
                .map(|prefix| observe(&p4.inner, prefix))
                .collect::<Result<Vec<_>>>()?;
            Ok((observations, receipt))
        },
    )?;
    if accepted_updates == 0 {
        require(
            before == after,
            "zero-update reload changed a raw score or P4 trace",
        )?;
    }

    let mut rows = Vec::with_capacity(prefixes.len());
    let mut selected = 0;
    let mut best = 0;
    for (i, (&prefix, (a, b))) in prefixes.iter().zip(before.iter().zip(&after)).enumerate() {
        let words_changed = a
            .masses
            .iter()
            .zip(&b.masses)
            .filter(|(x, y)| x != y)
            .count();
        let source_changed = a.trace["pick_pos"] != b.trace["pick_pos"];
        let argmax_changed = a.trace["token"] != b.trace["token"];
        let rank = 8 * usize::from(words_changed != 0)
            + 4 * usize::from(source_changed)
            + 2 * usize::from(argmax_changed)
            + usize::from(b.trace["pick_pos"].is_u64());
        if rank > best {
            selected = i;
            best = rank;
        }
        rows.push(json!({
            "source": if i < witness_prefixes.len() { "fixed_dev" } else { "gpu_selected_dev" },
            "prefix_ids": prefix,
            "legal_q48_words": a.masses.len(),
            "changed_q48_words": words_changed,
            "pick_position_changed": source_changed,
            "pick_successor_changed": a.trace["pick_successor"] != b.trace["pick_successor"],
            "pick_class_changed": a.trace["pick_class"] != b.trace["pick_class"],
            "argmax_changed": argmax_changed,
            "baseline": a.trace,
            "trained": b.trace,
        }));
    }
    let cli_before = cli(
        chat_path,
        p4_path,
        &baseline,
        prefixes[selected],
        &before[selected],
    )?;
    let cli_after = cli(
        chat_path,
        candidate_path,
        &candidate,
        prefixes[selected],
        &after[selected],
    )?;
    let (planted, wrong_address_control) = wrong_pick_trace(&cli_after["record"]);
    let wrong_address_refusal = refused(
        validate_cli(&planted, &candidate, prefixes[selected], &after[selected]),
        "disagrees with exact P4 serving reference",
    )?;
    let stale_refusal = if accepted_updates == 0 {
        require(
            cli_before["record"] == cli_after["record"],
            "zero-update ordinary CLI replay changed its output or trace",
        )?;
        None
    } else {
        Some(refused(
            validate_cli(
                &cli_before["record"],
                &candidate,
                prefixes[selected],
                &after[selected],
            ),
            "disagrees with exact P4 serving reference",
        )?)
    };

    let report = json!({
        "schema": "uor-r4.p5-wick-pilot-serving/1",
        "checkpoint_sha256": sha256(&checkpoint),
        "baseline_epoch": original.epoch,
        "trained_epoch": trained.epoch,
        "accepted_updates": accepted_updates,
        "deployment_mode": if accepted_updates == 0 { "unchanged_baseline_cold_reload" } else { "trained_repack_cold_reload" },
        "changed_roots": original.roots.iter().zip(&trained.roots).filter(|(a,b)| a != b).count(),
        "baseline": baseline,
        "trained": candidate,
        "trained_path": candidate_path,
        "tokenizer_bytes": TOKENIZER_BYTES,
        "uor_chat_sha256": uor_r4_integer::sha256_file(chat_path)?,
        "a1s_q32_payload_sha256": weight_words_sha,
        "fixed_prefixes": witness_prefixes.len(),
        "gpu_selected_prefixes": usize::from(known_changed_prefix.is_some()),
        "restored_bit_identical_rows": if accepted_updates == 0 { 0 } else { prefixes.len() },
        "unchanged_cold_reload_rows": if accepted_updates == 0 { prefixes.len() } else { 0 },
        "negative_controls": negatives,
        "wrong_pick_address_control": wrong_address_control,
        "refused_wrong_pick_address_trace": wrong_address_refusal,
        "refused_stale_artifact_trace": stale_refusal,
        "selected_cli_row": selected,
        "ordinary_baseline": cli_before,
        "ordinary_trained": cli_after,
        "first_score_or_source_change": rows.iter().position(|row| row["changed_q48_words"].as_u64().unwrap() > 0 || row["pick_position_changed"] == true),
        "rows": rows,
        "checkpoint_read_ns": checkpoint_read_ns,
        "rewrite_ns": rewrite_ns,
        "pack_ns_including_negative_probes": pack_ns,
        "restore_ns": restore_ns,
    });
    let mut output = File::create_new(output_dir.join("serving.json"))?;
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uor_r4_count_runtime::PickClassRule;

    #[test]
    fn admits_exact_zero_updates_and_rejects_stale_or_unadvanced_changes() -> Result<()> {
        let original = P5S1State::from_roots(1, 4, 420, [0x22; 32], vec![1; 4])?;
        let unchanged = P5S1State::from_bytes(&original.encode_bytes()?)?;
        checkpoint_for_replay(&original, &unchanged, original.epoch)?;
        assert!(checkpoint_for_replay(&original, &unchanged, original.epoch + 1).is_err());
        let mut edited = unchanged;
        edited.roots[0] ^= 1;
        assert!(checkpoint_for_replay(&original, &edited, original.epoch).is_err());
        edited.epoch += 1;
        checkpoint_for_replay(&original, &edited, edited.epoch)?;
        edited.parent_trunk_sha256[0] ^= 1;
        assert!(checkpoint_for_replay(&original, &edited, original.epoch).is_err());
        Ok(())
    }

    #[test]
    fn binds_frozen_weights_to_only_the_new_trunk() -> Result<()> {
        let manifest =
            br#"{"prepared":{"trunk_sha256":"original","r_sha256":"frozen"},"arm":"a1s"}"#;
        let mut words = vec![0u64; 320];
        for bucket in words.chunks_exact_mut(4) {
            bucket[0] = 1 << 32;
        }
        let old = FrozenPickMixture::with_rule(
            0,
            Sha256::digest(manifest).into(),
            words.clone(),
            PickClassRule::Five,
        )?
        .to_bytes();
        let (new_manifest, new) = rebind_a1s(manifest, &old, "original", "trained")?;
        assert_eq!(
            serde_json::from_slice::<Value>(&new_manifest)?,
            json!({"prepared":{"trunk_sha256":"trained","r_sha256":"frozen"},"arm":"a1s"})
        );
        let new_sha: [u8; 32] = Sha256::digest(&new_manifest).into();
        assert!(FrozenPickMixture::from_bytes(&old, new_sha).is_err());
        assert_eq!(
            FrozenPickMixture::from_bytes(&new, new_sha)?.weights(),
            words
        );
        assert!(rebind_a1s(manifest, &old, "wrong parent", "trained").is_err());
        assert!(rebind_a1s(manifest, &old[..43], "original", "trained").is_err());
        Ok(())
    }

    #[test]
    fn rejects_shifted_original_source_and_stale_cli_output() -> Result<()> {
        let prefix = [0, 17, 41, 63];
        let identity = json!({
            "artifact_sha256": "new", "trunk_sha256": "train", "state_sha256": "bound", "components": {}
        });
        let exact = Observation {
            trace: json!({"pick_pos": 1, "pick_successor": 41, "token": 41, "mass_sha256": "mass"}),
            masses: vec![1],
        };
        let report = json!({
            "schema": "uor-r4.uor-chat-discrete-parity/1",
            "model_schema": SCHEMA_P4,
            "read_enabled": true,
            "artifact_sha256": "new",
            "p4_trunk_sha256": "train",
            "p4_state_sha256": "bound",
            "p4_components": {},
            "prompt_ids": prefix,
            "generated_ids": [41],
            "p4_steps": [exact.trace],
        });
        validate_cli(&report, &identity, &prefix, &exact)?;
        let (shifted, kind) = wrong_pick_trace(&report);
        assert_eq!(kind, "off_by_one_original_position");
        assert_eq!(shifted["p4_steps"][0]["pick_pos"], 2);
        assert!(validate_cli(&shifted, &identity, &prefix, &exact).is_err());
        let mut stale = report.clone();
        stale["artifact_sha256"] = json!("old");
        assert!(validate_cli(&stale, &identity, &prefix, &exact).is_err());

        let mut short = report;
        let no_pick = Observation {
            trace: json!({"pick_pos": null, "pick_successor": null, "token": 41, "mass_sha256": "mass"}),
            masses: vec![1],
        };
        for prefix in [&[0][..], &[0, 17][..]] {
            valid_prefix(prefix)?;
            short["prompt_ids"] = json!(prefix);
            short["p4_steps"] = json!([no_pick.trace]);
            validate_cli(&short, &identity, prefix, &no_pick)?;
            let (planted, kind) = wrong_pick_trace(&short);
            assert_eq!(kind, "impossible_position_on_no_pick_prefix");
            assert_eq!(planted["p4_steps"][0]["pick_pos"], 0);
            assert!(validate_cli(&planted, &identity, prefix, &no_pick).is_err());
        }
        assert!(valid_prefix(&[]).is_err());
        assert!(valid_prefix(&[0, 17, 0, 63]).is_err());
        assert!(valid_prefix(&[0, 17, 1, 63]).is_err());
        Ok(())
    }
}
