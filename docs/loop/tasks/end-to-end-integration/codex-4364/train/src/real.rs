use crate::cuda_backend::{
    gpu::{CudaV3ButterflyEngine, INT_V3_CUBIN_BYTES},
    FlatV3AdamWBuffers,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use uor_r4_native_vector::butterfly_rotor_v3::ButterflyV3Model;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) const PARENT: &str = "20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589";
pub(crate) const HISTORICAL_CUBIN_SHA: &str =
    "d7594dab97fd490bee2cad35d595a12054f64b2cdebe1eb628ac96ac8f56328a";
const ROSTER: &str = "700c0bd3e78bd2dd8c0644d0abaea3e265538205eec20f69b9da0ccddca1ba92";
pub(crate) const MARGIN: i32 = 256;
pub(crate) const GAIN: i32 = 32768;
pub(crate) const LR_Q8: i32 = 512;
pub(crate) const WD_SHIFT: u32 = 19;
pub(crate) const EPS: u64 = 16;
pub(crate) const START_STEP: u64 = 3333;
const LABELS: u64 = 167;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    id: u64,
    prefix: Vec<u16>,
    expected: Vec<u16>,
    stratum: String,
}

pub(crate) fn require(ok: bool, error: &str) -> Result<()> {
    if !ok {
        return Err(error.into());
    }
    Ok(())
}

pub(crate) fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn sha_words<const N: usize>(words: impl Iterator<Item = [u8; N]>) -> String {
    let mut hasher = Sha256::new();
    for word in words {
        hasher.update(word);
    }
    hex::encode(hasher.finalize())
}

pub(crate) fn cubin_sha() -> String {
    sha(INT_V3_CUBIN_BYTES)
}

pub(crate) fn gpu_uuid(engine: &CudaV3ButterflyEngine) -> Result<String> {
    Ok(hex::encode(engine.ctx.uuid()?.bytes.map(|byte| byte as u8)))
}

fn read_rows(path: &Path) -> Result<Vec<Row>> {
    let bytes = fs::read(path)?;
    require(sha(&bytes) == ROSTER, "roster SHA-256 mismatch")?;
    let rows: Vec<Row> = std::str::from_utf8(&bytes)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    require(rows.len() == 4, "expected exactly four rows")?;
    for (index, row) in rows.iter().enumerate() {
        let start = row.prefix.len();
        let end = start + row.expected.len();
        require(
            row.id == (index + 1) as u64
                && row.prefix.first() == Some(&0)
                && row.prefix.iter().skip(1).all(|&id| id > 0 && id < 4096)
                && row.prefix.iter().filter(|&&id| id == 1).count() == 2
                && matches!(row.stratum.as_str(), "near" | "far" | "intermediate")
                && row.expected.len() >= 2
                && row.expected.last() == Some(&1)
                && row.expected[..row.expected.len() - 1]
                    .iter()
                    .all(|&id| (2..4096).contains(&id))
                && end <= 1088
                && start / 257 == (end - 1) / 257
                && start % 257 != 0,
            "roster is outside the registered third-assistant-turn shape",
        )?;
    }
    require(
        rows.iter().map(|r| r.expected.len() as u64).sum::<u64>() == LABELS,
        "roster count differs from the registered 167 labels",
    )?;
    Ok(rows)
}

pub(crate) fn eligible(model: &ButterflyV3Model) -> [u64; 65] {
    let mut masks = [0_u64; 65];
    for (cell, original) in model.vocab_inv.chunks_exact(64).enumerate() {
        for (slot, &id) in original.iter().enumerate() {
            if usize::from(id) < model.cfg.vocab && (model.cfg.vocab == 1 || id != 0) {
                masks[cell + 1] |= 1 << slot;
            }
        }
        if masks[cell + 1] != 0 {
            masks[0] |= 1 << cell;
        }
    }
    masks
}

fn strongest_other(scores: &[i32; 64], mask: u64, target: usize) -> Result<(usize, i64)> {
    let other = (0..64)
        .filter(|&slot| slot != target && mask & (1 << slot) != 0)
        .max_by_key(|&slot| (scores[slot], std::cmp::Reverse(slot)))
        .ok_or("ordinary opening has no rival")?;
    Ok((other, i64::from(scores[target]) - i64::from(scores[other])))
}

fn fine_scores(model: &ButterflyV3Model, hidden: &[i32], cell: usize) -> [i32; 64] {
    let dim = model.cfg.dim;
    let shift = ButterflyV3Model::head_dot_shift(dim);
    std::array::from_fn(|slot| {
        let start = (cell * 64 + slot) * dim;
        let row = &model.embeddings[start..start + dim];
        let sum = hidden
            .iter()
            .zip(row)
            .map(|(&activation, &weight)| i64::from(activation) * i64::from(weight))
            .sum::<i64>();
        ((sum + (1_i64 << (shift - 1))) >> shift).clamp(-(1 << 22), 1 << 22) as i32
    })
}

fn ordinary(
    engine: &CudaV3ButterflyEngine,
    model: &ButterflyV3Model,
    rows: &[Row],
    masks: &[u64; 65],
    all_transitions: bool,
) -> Result<Vec<Value>> {
    let mut report = Vec::new();
    for row in rows {
        let first = row.expected[0];
        let mut tokens = row.prefix.clone();
        if all_transitions {
            tokens.extend_from_slice(&row.expected);
        } else {
            tokens.push(first);
        }
        let mut mask = vec![u8::from(all_transitions); tokens.len()];
        mask[0] = 0;
        mask[row.prefix.len()] = 1;
        let (mass, greedy) =
            engine.eval_records(&tokens, &mask, &[0], &[tokens.len() as u32], false)?;
        let mut session = model.new_session();
        let mut opening = None;
        for (t, window) in tokens.windows(2).enumerate() {
            let out = model.step(window[0], &mut session)?;
            if mask[t + 1] != 0 {
                require(
                    out.greedy_token == greedy[t + 1]
                        && model.target_q48_mass_from_hidden(&out.hidden_q15, window[1])?
                            == mass[t + 1],
                    "unchanged GPU writer disagrees with ordinary Rust writer",
                )?;
            }
            if t + 1 == row.prefix.len() {
                let slot = model.vocab_perm[first as usize] as usize;
                let cell = slot / 64;
                let (coarse_rival, coarse_gap) =
                    strongest_other(&out.coarse_scores, masks[0], cell)?;
                let gold_scores = fine_scores(model, &out.hidden_q15, cell);
                let (fine_rival, fine_gap) =
                    strongest_other(&gold_scores, masks[1 + cell], slot % 64)?;
                let elected_cell = out.greedy_coarse as usize;
                let elected_scores = if cell == elected_cell {
                    gold_scores
                } else {
                    fine_scores(model, &out.hidden_q15, elected_cell)
                };
                let (winner_rival, winner_gap) = strongest_other(
                    &elected_scores,
                    masks[1 + elected_cell],
                    out.greedy_fine as usize,
                )?;
                opening = Some(json!({
                    "id": row.id, "gold_original_id": first, "elected_original_id": out.greedy_token,
                    "gold_cell": cell, "elected_cell": elected_cell,
                    "gold_coarse_rival_cell": coarse_rival, "gold_coarse_gap_q15": coarse_gap,
                    "gold_fine_rival_slot": fine_rival, "gold_fine_gap_q15": fine_gap,
                    "elected_cell_fine_rival_slot": winner_rival, "elected_cell_fine_gap_q15": winner_gap,
                }));
            }
        }
        report.push(opening.ok_or("ordinary opening was not reached")?);
    }
    Ok(report)
}

fn check_workspace(
    engine: &CudaV3ButterflyEngine,
    model: &ButterflyV3Model,
    rows: &[Row],
    docs: &[Vec<u16>],
) -> Result<usize> {
    let saved = engine.stream.clone_dtoh(&engine.d_act_ws)?;
    let dim = model.cfg.dim;
    let max = engine.max_seq_len;
    let layer_words = model.cfg.layers * max * dim;
    let record_words = (3 * model.cfg.layers + 2) * max * dim;
    let mut compared = 0;
    for (index, (row, tokens)) in rows.iter().zip(docs).enumerate() {
        let mut session = model.new_session();
        let base = index * record_words;
        for (step, &token) in tokens[..tokens.len() - 1].iter().enumerate() {
            for layer in 0..model.cfg.layers {
                let start = base + 2 * layer_words + (layer * max + step) * dim;
                require(
                    saved[start..start + dim] == session.carry[layer],
                    "raw CUDA/Rust pre-input carry mismatch",
                )?;
                for tap in 1..=3 {
                    let reference = &session.shift_hist[layer][(tap - 1) * dim..tap * dim];
                    if let Some(prior) = step.checked_sub(tap) {
                        let start = base + layer_words + (layer * max + prior) * dim;
                        require(
                            saved[start..start + dim] == *reference,
                            "raw CUDA/Rust shift mismatch",
                        )?;
                    } else {
                        require(
                            reference.iter().all(|&word| word == 0),
                            "nonzero absent initial shift",
                        )?;
                    }
                }
            }
            model.step(token, &mut session)?;
            require(
                session.tokens_seen == (step + 1) as u64,
                "consumed token count mismatch",
            )?;
            compared += 1;
        }
        require(
            session.tokens_seen as usize >= row.prefix.len(),
            "missed opening causality boundary",
        )?;
    }
    Ok(compared)
}

pub(crate) fn run(
    parent: &Path,
    tape: &Path,
    receipt: &Path,
    historical: &Path,
    output: &Path,
) -> Result<()> {
    require(
        std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() == Ok("1"),
        "requires CUDA_DISABLE_PTX_JIT=1",
    )?;
    require(!output.exists(), "output directory already exists")?;
    let rows = read_rows(tape)?;
    let parent_bytes = fs::read(parent)?;
    require(
        sha(&parent_bytes) == PARENT,
        "registered frozen parent SHA mismatch",
    )?;
    let model = ButterflyV3Model::from_bytes(&parent_bytes)?;
    require(
        model.step == START_STEP && model.to_bytes() == parent_bytes && model.cfg.vocab == 4096,
        "frozen native parent step/shape/serialization mismatch",
    )?;
    for pair in rows.chunks_exact(2) {
        require(
            model.vocab_perm[pair[0].expected[0] as usize] / 64
                != model.vocab_perm[pair[1].expected[0] as usize] / 64,
            "expected both roster pairs to oppose in coarse cells",
        )?;
    }
    let masks = eligible(&model);
    let docs = rows
        .iter()
        .map(|row| {
            let mut tokens = row.prefix.clone();
            tokens.extend_from_slice(&row.expected);
            tokens
        })
        .collect::<Vec<_>>();
    let stride = docs.iter().map(Vec::len).max().ok_or("empty tape")?;
    let mut tokens = Vec::with_capacity(stride * rows.len());
    let mut mask = Vec::with_capacity(stride * rows.len());
    let mut openings = Vec::new();
    let mut lens = Vec::new();
    for (row, doc) in rows.iter().zip(&docs) {
        let base = tokens.len();
        tokens.extend_from_slice(doc);
        tokens.resize(base + stride, 0);
        mask.resize(base + row.prefix.len(), 0);
        mask.resize(base + doc.len(), 1);
        mask.resize(base + stride, 0);
        openings.push(row.prefix.len() as u32);
        lens.push(doc.len() as u32);
    }
    let moments = model.new_adamw_state();
    let receipt_bytes = fs::read(receipt)?;
    let finite: Value = serde_json::from_slice(&receipt_bytes)?;
    require(
        finite["verdict"] == "PASS"
            && finite["protocol"] == "uor-opening-finite-v1"
            && finite["historical_cubin_sha256"] == HISTORICAL_CUBIN_SHA
            && finite["dim"] == model.cfg.dim
            && finite["registered_constants"]["margin"] == MARGIN
            && finite["registered_constants"]["gain"] == GAIN
            && finite["cubin_sha256"] == cubin_sha(),
        "missing or mismatched planted finite receipt",
    )?;
    let historical_bytes = fs::read(historical)?;
    require(
        sha(&historical_bytes) == HISTORICAL_CUBIN_SHA,
        "historical cubin SHA mismatch",
    )?;
    let mut sham = CudaV3ButterflyEngine::from_historical_cubin(
        &model,
        &moments,
        4,
        stride - 1,
        &historical_bytes,
    )?;
    let uuid = gpu_uuid(&sham)?;
    require(
        finite["gpu_uuid"] == uuid && sham.ctx.compute_capability()? == (8, 9),
        "finite receipt/device mismatch",
    )?;
    let mut active = CudaV3ButterflyEngine::new(&model, &moments, 4, stride - 1)?;
    let before = ordinary(&sham, &model, &rows, &masks, true)?;
    let (sham_loss, sham_labels) = sham.accumulate_batch(&tokens, &mask, &lens, stride, true)?;
    let (active_loss, active_labels, active_trace) = active.accumulate_opening_batch(
        &tokens, &mask, &lens, stride, true, &openings, &masks, MARGIN, [GAIN; 2],
    )?;
    require(
        sham_loss == active_loss && sham_labels == LABELS && active_labels == LABELS,
        "device base losses or exact supervised counts differ",
    )?;
    require(active_trace.len() == 40, "opening trace shape mismatch")?;
    let parity_transitions = check_workspace(&sham, &model, &rows, &docs)?;
    let active_transitions = check_workspace(&active, &model, &rows, &docs)?;
    require(
        active_transitions == parity_transitions,
        "training forward parity counts differ",
    )?;
    for (treated, ordinary) in active_trace.chunks_exact(10).zip(&before) {
        require(
            treated[0] == ordinary["gold_coarse_rival_cell"]
                && treated[1] == ordinary["gold_coarse_gap_q15"]
                && treated[5] == ordinary["gold_fine_rival_slot"]
                && treated[6] == ordinary["gold_fine_gap_q15"],
            "active opening diverged from historical writer raw scores",
        )?;
    }

    let sham_grads = sham.download_grads()?;
    let active_grads = active.download_grads()?;
    let movement = active_grads
        .iter()
        .zip(&sham_grads)
        .filter(|(left, right)| left != right)
        .count();
    // Exactly one physical GPU update per arm. No CPU BPTT or optimizer on authentic rows.
    sham.apply_adamw_step(START_STEP + 1, LR_Q8, WD_SHIFT, EPS, true, true)?;
    active.apply_adamw_step(START_STEP + 1, LR_Q8, WD_SHIFT, EPS, true, true)?;
    let download = |engine: &CudaV3ButterflyEngine| -> Result<_> {
        let mut saved = model.clone();
        let mut state = model.new_adamw_state();
        engine.download_into(&mut saved, &mut state)?;
        saved.step = START_STEP + 1;
        let bytes = saved.to_bytes();
        require(
            ButterflyV3Model::from_bytes(&bytes)? == saved
                && saved.cfg == model.cfg
                && saved.vocab_perm == model.vocab_perm,
            "full native checkpoint read-back mismatch",
        )?;
        Ok((saved, state, bytes))
    };
    let (sham_model, sham_state, sham_bytes) = download(&sham)?;
    let (active_model, active_state, active_bytes) = download(&active)?;
    let sham_flat = FlatV3AdamWBuffers::pack(&sham_model, &sham_state);
    let active_flat = FlatV3AdamWBuffers::pack(&active_model, &active_state);
    let digests = |grads: &[i64], flat: &FlatV3AdamWBuffers| -> Value {
        json!({
            "gradient_le_sha256": sha_words(grads.iter().map(|word| word.to_le_bytes())),
            "weights_le_sha256": sha_words(flat.weights.iter().map(|word| word.to_le_bytes())),
            "adam_m_le_sha256": sha_words(flat.adam_m.iter().map(|word| word.to_le_bytes())),
            "adam_v_le_sha256": sha_words(flat.adam_v.iter().map(|word| word.to_le_bytes())),
        })
    };
    active.use_historical_writer(&historical_bytes)?;
    let sham_after = ordinary(&sham, &sham_model, &rows, &masks, false)?;
    let active_after = ordinary(&active, &active_model, &rows, &masks, false)?;
    let report = json!({
        "protocol": "uor-opening-real-single-update-v1",
        "result": if movement == 0 || sham_bytes == active_bytes { "NO_EFFECTIVE_WEIGHT_CONTRAST" } else { "MEASURED_UPDATE" },
        "parent_sha256": PARENT, "roster_sha256": ROSTER, "finite_receipt_sha256": sha(&receipt_bytes),
        "gpu_uuid": uuid, "cubin_sha256": cubin_sha(), "base_loss_q15": sham_loss.to_string(),
        "labels_per_arm": LABELS, "batch_lengths": lens, "opening_labels": openings,
        "step_before": START_STEP, "step_after": START_STEP + 1, "updates_per_arm": 1,
        "margin_words": MARGIN, "active_gain_q15": GAIN, "sham_gain_q15": 0,
        "baseline_forward_parity_transitions": parity_transitions, "different_gradient_words": movement,
        "historical_control_cubin_sha256": HISTORICAL_CUBIN_SHA, "control": "physical original ABI #4361",
        "sham_checkpoint_sha256": sha(&sham_bytes), "active_checkpoint_sha256": sha(&active_bytes),
        "ordinary_before": before, "ordinary_after_sham": sham_after, "ordinary_after_active": active_after,
        "device_active_opening_trace": active_trace, "sham_words": digests(&sham_grads, &sham_flat), "active_words": digests(&active_grads, &active_flat),
    });
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new().mode(0o700).create(output)?;
    }
    #[cfg(not(unix))]
    fs::create_dir(output)?;
    fs::write(output.join("sham.uorbfy03"), sham_bytes)?;
    fs::write(output.join("active.uorbfy03"), active_bytes)?;
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
