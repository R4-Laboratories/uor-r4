use std::{fs, path::Path};

use serde_json::{json, Value};
use uor_r4_native_vector::{
    butterfly_rotor::loss_and_grad_64,
    butterfly_rotor_v3::{rmsnorm_forward_fast_q15, ButterflyV3Model},
};

use crate::{
    cuda_backend::gpu::CudaV3ButterflyEngine,
    rank::{add_to_legacy, strongest_rival_margin},
    real::{
        self, eligible, require, sha, sha_words, EPS, GAIN, LR_Q8, MARGIN, START_STEP, WD_SHIFT,
    },
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const LENGTHS: [u32; 4] = [980, 1018, 933, 1011];
const OPENINGS: [u32; 4] = [953, 957, 887, 978];
const STRIDE: usize = 1018;
const TRANSITIONS: usize = STRIDE - 1;
const VOCAB: usize = 4096;
const DIM: usize = 512;
const LAYERS: usize = 6;
const LABELS: u64 = 167;
const CAPACITY_CUBIN: &str = "9b8185fd00e3d7896569476b1160102dc0305ddc24fe29d63cb0bd231113d533";

fn fixture() -> (Vec<u16>, Vec<u8>) {
    let mut tokens = vec![0; LENGTHS.len() * STRIDE];
    let mut mask = vec![0; tokens.len()];
    for (row, (&length, &opening)) in LENGTHS.iter().zip(&OPENINGS).enumerate() {
        let base = row * STRIDE;
        let end = base + length as usize;
        let mut state = 0x4364_0001_u32 + row as u32;
        for token in &mut tokens[base + 1..end - 1] {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *token = (2 + (state >> 16) as usize % (VOCAB - 2)) as u16;
        }
        tokens[end - 1] = 1;
        mask[base + opening as usize..end].fill(1);
    }
    (tokens, mask)
}

fn compare_forward(old: &[i32], other: &[i32]) -> Result<usize> {
    let region = TRANSITIONS * DIM;
    let record_words = (3 * LAYERS + 2) * region;
    require(
        old.len() == LENGTHS.len() * record_words && old.len() == other.len(),
        "unexpected device activation workspace shape",
    )?;
    // The final region contains head derivatives, which may change in the active arm.
    // The preceding regions hold layer inputs, normalized shift inputs, carry and prehead.
    let forward_words = record_words - region;
    for (a, b) in old
        .chunks_exact(record_words)
        .zip(other.chunks_exact(record_words))
    {
        require(
            a[..forward_words] == b[..forward_words],
            "forward workspace word mismatch",
        )?;
    }
    Ok(LENGTHS.len() * forward_words)
}

fn head(model: &ButterflyV3Model, prehead: &[i32]) -> Result<Vec<i32>> {
    let (norm, _) = rmsnorm_forward_fast_q15(prehead, &model.final_norm_q12);
    Ok(model.head_proj.forward(&norm)?.0)
}

fn scores(hidden: &[i32], rows: &[i16]) -> [i32; 64] {
    let shift = ButterflyV3Model::head_dot_shift(DIM);
    std::array::from_fn(|slot| {
        let dot = hidden
            .iter()
            .zip(&rows[slot * DIM..(slot + 1) * DIM])
            .map(|(&h, &w)| i64::from(h) * i64::from(w))
            .sum::<i64>();
        ((dot + (1_i64 << (shift - 1))) >> shift).clamp(-(1 << 22), 1 << 22) as i32
    })
}

fn winner(scores: &[i32; 64], mask: u64) -> Result<usize> {
    (0..64)
        .filter(|&slot| mask & (1 << slot) != 0)
        .max_by_key(|&slot| (scores[slot], std::cmp::Reverse(slot)))
        .ok_or_else(|| "no admissible writer slot".into())
}

fn opening_witnesses(
    model: &ButterflyV3Model,
    workspace: &[i32],
    tokens: &[u16],
    writer: &(Vec<u64>, Vec<u16>),
) -> Result<(Vec<Value>, Vec<i64>)> {
    let masks = eligible(model);
    let layer_region = LAYERS * TRANSITIONS * DIM;
    let record_words = (3 * LAYERS + 2) * TRANSITIONS * DIM;
    let mut reports = Vec::with_capacity(LENGTHS.len());
    let mut expected_trace = Vec::with_capacity(LENGTHS.len() * 10);
    for (row, &opening) in OPENINGS.iter().enumerate() {
        let pos = row * STRIDE + opening as usize;
        let hidden_start = row * record_words + 3 * layer_region + (opening as usize - 1) * DIM;
        let hidden = head(model, &workspace[hidden_start..hidden_start + DIM])?;
        let gold = model.vocab_perm[tokens[pos] as usize] as usize;
        let group = gold / 64;
        let coarse = scores(&hidden, &model.coarse_centroids);
        let fine_rows = |cell| &model.embeddings[cell * 64 * DIM..(cell + 1) * 64 * DIM];
        let fine = scores(&hidden, fine_rows(group));
        let elected = winner(&coarse, masks[0])?;
        let elected_fine = scores(&hidden, fine_rows(elected));
        let slot = winner(&elected_fine, masks[elected + 1])?;
        require(
            writer.1[pos] == model.vocab_inv[elected * 64 + slot]
                && writer.0[pos] == model.target_q48_mass_from_hidden(&hidden, tokens[pos])?,
            "saved prehead reconstruction disagrees with the physical writer at opening",
        )?;
        for (score, mask, target) in [
            (&coarse, masks[0], group),
            (&fine, masks[group + 1], gold % 64),
        ] {
            let rank = strongest_rival_margin(score, mask, target, MARGIN, GAIN)
                .map_err(|error| format!("{error:?}"))?
                .ok_or("full-vocabulary opening has no rival")?;
            let (_, mut legacy) = loss_and_grad_64(score, target, model.cfg.loss_kind);
            let clips =
                add_to_legacy(&mut legacy, target, &rank).map_err(|error| format!("{error:?}"))?;
            expected_trace.extend([
                rank.rival as i64,
                rank.gap_q15,
                rank.loss_q15,
                i64::from(rank.derivative_q15),
                i64::from(clips),
            ]);
        }
        reports.push(json!({
            "row": row + 1,
            "opening": opening,
            "synthetic_gold_id": tokens[pos],
            "gold_cell": group,
            "gold_slot_within_cell": gold % 64,
            "writer_id": writer.1[pos],
            "hidden_q15_le_sha256": sha_words(hidden.iter().map(|w| w.to_le_bytes())),
            "coarse_scores_q15_le_sha256": sha_words(coarse.iter().map(|w| w.to_le_bytes())),
            "gold_cell_fine_scores_q15_le_sha256": sha_words(fine.iter().map(|w| w.to_le_bytes())),
            "elected_cell_fine_scores_q15_le_sha256": sha_words(elected_fine.iter().map(|w| w.to_le_bytes())),
        }));
    }
    Ok((reports, expected_trace))
}

fn compare_raw_words<T: cudarc::driver::DeviceRepr + PartialEq>(
    old: &CudaV3ButterflyEngine,
    zero: &CudaV3ButterflyEngine,
    old_words: &cudarc::driver::CudaSlice<T>,
    zero_words: &cudarc::driver::CudaSlice<T>,
    message: &str,
) -> Result<usize> {
    let old_words = old.stream.clone_dtoh(old_words)?;
    let zero_words = zero.stream.clone_dtoh(zero_words)?;
    require(old_words == zero_words, message)?;
    Ok(old_words.len())
}

pub(crate) fn run(parent: &Path, original_cubin: &Path) -> Result<()> {
    require(
        std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() == Ok("1"),
        "requires CUDA_DISABLE_PTX_JIT=1",
    )?;
    let parent_bytes = fs::read(parent)?;
    require(
        sha(&parent_bytes) == real::PARENT,
        "frozen parent SHA mismatch",
    )?;
    let original = fs::read(original_cubin)?;
    require(
        sha(&original) == real::HISTORICAL_CUBIN_SHA,
        "original CUDA cubin SHA mismatch",
    )?;
    require(
        real::cubin_sha() == CAPACITY_CUBIN,
        "embedded CUDA cubin SHA mismatch",
    )?;
    let model = ButterflyV3Model::from_bytes(&parent_bytes)?;
    let cfg = model.cfg;
    require(
        model.step == START_STEP
            && model.to_bytes() == parent_bytes
            && cfg.vocab == VOCAB
            && cfg.dim == DIM
            && cfg.layers == LAYERS
            && cfg.ffn_branches == 4
            && cfg.pkm_sub_keys == 64
            && cfg.pkm_key_dim == 32
            && model
                .vocab_perm
                .iter()
                .enumerate()
                .any(|(id, &slot)| id != slot as usize),
        "frozen parent operating shape, permutation, step or serialization mismatch",
    )?;
    drop(parent_bytes);
    let (tokens, mask) = fixture();
    let masks = eligible(&model);
    let moments = model.new_adamw_state();
    let mut old = CudaV3ButterflyEngine::from_historical_cubin(
        &model,
        &moments,
        LENGTHS.len(),
        TRANSITIONS,
        &original,
    )?;
    let mut zero = CudaV3ButterflyEngine::new(&model, &moments, LENGTHS.len(), TRANSITIONS)?;
    let mut active = CudaV3ButterflyEngine::new(&model, &moments, LENGTHS.len(), TRANSITIONS)?;
    let uuid = real::gpu_uuid(&old)?;
    require(
        old.ctx.compute_capability()? == (8, 9)
            && uuid == real::gpu_uuid(&zero)?
            && uuid == real::gpu_uuid(&active)?,
        "three engines must use the same physical SM89 device",
    )?;

    let baseline = old.accumulate_batch(&tokens, &mask, &LENGTHS, STRIDE, true)?;
    let (zero_loss, zero_labels, zero_trace) = zero.accumulate_opening_batch(
        &tokens, &mask, &LENGTHS, STRIDE, true, &OPENINGS, &masks, MARGIN, [0; 2],
    )?;
    let (active_loss, active_labels, active_trace) = active.accumulate_opening_batch(
        &tokens, &mask, &LENGTHS, STRIDE, true, &OPENINGS, &masks, MARGIN, [GAIN; 2],
    )?;
    require(
        baseline.1 == LABELS
            && baseline == (zero_loss, zero_labels)
            && baseline == (active_loss, active_labels),
        "base losses or supervised label counts differ",
    )?;
    let old_grad = old.download_grads()?;
    require(
        old_grad == zero.download_grads()?,
        "original-vs-zero gradient word mismatch",
    )?;
    let active_grad = active.download_grads()?;
    let embeddings = VOCAB * DIM;
    let changed = |range: std::ops::Range<usize>| {
        old_grad[range.clone()]
            .iter()
            .zip(&active_grad[range])
            .filter(|(a, b)| a != b)
            .count()
    };
    let changed_fine = changed(0..embeddings);
    let changed_coarse = changed(embeddings..embeddings + 64 * DIM);
    require(
        changed_fine > 0 && changed_coarse > 0,
        "treatment made no gradient change at one or both sites",
    )?;
    drop(active_grad);

    let workspace = old.stream.clone_dtoh(&old.d_act_ws)?;
    let forward_words = compare_forward(&workspace, &zero.stream.clone_dtoh(&zero.d_act_ws)?)?;
    compare_forward(&workspace, &active.stream.clone_dtoh(&active.d_act_ws)?)?;
    let offsets = std::array::from_fn::<_, 4, _>(|row| (row * STRIDE) as u64);
    let writer = old.eval_records(&tokens, &mask, &offsets, &LENGTHS, false)?;
    require(
        writer == zero.eval_records(&tokens, &mask, &offsets, &LENGTHS, false)?
            && writer == active.eval_records(&tokens, &mask, &offsets, &LENGTHS, false)?,
        "ordinary CUDA writer emitted ID or target mass mismatch",
    )?;
    let (opening_readback, expected) = opening_witnesses(&model, &workspace, &tokens, &writer)?;
    drop(workspace);
    require(
        active_trace == expected,
        "active device visit, gap, loss, derivative or clamp count mismatch",
    )?;
    require(
        zero_trace.len() == expected.len()
            && zero_trace
                .chunks_exact(5)
                .zip(expected.chunks_exact(5))
                .all(|(zero, active)| zero[..2] == active[..2] && zero[2..] == [0; 3]),
        "scanned zero-gain visitation or stage arguments mismatch",
    )?;
    let visits_with_derivative = |site| {
        active_trace
            .chunks_exact(10)
            .filter(|row| row[site] == i64::from(GAIN))
            .count()
    };
    let live_coarse = visits_with_derivative(3);
    let live_fine = visits_with_derivative(8);
    require(
        live_coarse > 0 && live_fine > 0,
        "registered gain produced no live derivative at one or both sites",
    )?;

    let step_after = START_STEP + 1;
    for engine in [&mut old, &mut zero, &mut active] {
        engine.apply_adamw_step(step_after, LR_Q8, WD_SHIFT, EPS, true, true)?;
    }
    let weights = compare_raw_words(
        &old,
        &zero,
        &old.d_weights,
        &zero.d_weights,
        "weight word mismatch",
    )?;
    let adam_m = compare_raw_words(
        &old,
        &zero,
        &old.d_adam_m,
        &zero.d_adam_m,
        "first moment word mismatch",
    )?;
    let adam_v = compare_raw_words(
        &old,
        &zero,
        &old.d_adam_v,
        &zero.d_adam_v,
        "second moment word mismatch",
    )?;
    let changed_weights = old
        .stream
        .clone_dtoh(&old.d_weights)?
        .iter()
        .zip(active.stream.clone_dtoh(&active.d_weights)?)
        .filter(|(before, after)| **before != *after)
        .count();
    require(
        changed_weights > 0,
        "treatment made no deployed weight change",
    )?;
    let checkpoint = |engine: &CudaV3ButterflyEngine| -> Result<_> {
        let mut updated = model.clone();
        let mut state = model.new_adamw_state();
        engine.download_into(&mut updated, &mut state)?;
        // The ABI receives the post-increment step seed and has no resident step counter.
        updated.step = step_after;
        let bytes = updated.to_bytes();
        let reloaded = ButterflyV3Model::from_bytes(&bytes)?;
        require(
            reloaded == updated && reloaded.step == step_after && reloaded.to_bytes() == bytes,
            "exact checkpoint serialization mismatch",
        )?;
        Ok(bytes)
    };
    let baseline_bytes = checkpoint(&old)?;
    require(
        baseline_bytes == checkpoint(&zero)?,
        "original-vs-zero checkpoint byte mismatch",
    )?;
    let active_bytes = checkpoint(&active)?;
    require(
        baseline_bytes != active_bytes,
        "treatment checkpoint is unchanged",
    )?;
    println!(
        "{}",
        json!({
            "verdict": "PASS",
            "protocol": "uor-opening-capacity-v1",
            "gpu_uuid": uuid,
            "parent_sha256": real::PARENT,
            "historical_cubin_sha256": real::HISTORICAL_CUBIN_SHA,
            "cubin_sha256": CAPACITY_CUBIN,
            "dim": DIM, "layers": LAYERS, "ffn_branches": 4, "pkm_sub_keys": 64, "pkm_key_dim": 32,
            "threads_per_block": DIM.min(256), "workspace_transitions_per_row": TRANSITIONS,
            "batch_lengths": LENGTHS, "opening_labels": OPENINGS,
            "transitions": LENGTHS.iter().map(|n| n - 1).sum::<u32>(),
            "vocab_permutation_le_sha256": sha_words(model.vocab_perm.iter().map(|w| w.to_le_bytes())),
            "synthetic_tokens_le_sha256": sha_words(tokens.iter().map(|w| w.to_le_bytes())),
            "synthetic_mask_sha256": sha(&mask),
            "base_loss_q15": baseline.0.to_string(), "labels_per_arm": baseline.1,
            "equal_gradient_words": old_grad.len(), "equal_weight_words": weights,
            "equal_adam_m_words": adam_m, "equal_adam_v_words": adam_v,
            "equal_forward_workspace_words_per_comparison": forward_words,
            "equal_writer_labels": LABELS, "opening_readback": opening_readback,
            "pkm_slot_export": "unavailable in pinned training ABI; compare layer input and prehead consequences",
            "margin_words": MARGIN, "active_gain_q15": GAIN, "sham_gain_q15": 0,
            "optimizer": {"lr_q8": LR_Q8, "wd_shift": WD_SHIFT, "eps": EPS, "sparse_skip": true, "learn_pkm_keys": true},
            "step_before": START_STEP, "step_after": step_after, "updates_per_arm": 1,
            "sham_trace": zero_trace, "active_trace": active_trace,
            "visits_per_site": OPENINGS.len(), "live_coarse_visits": live_coarse, "live_fine_visits": live_fine,
            "different_tied_embedding_gradient_words": changed_fine,
            "different_coarse_gradient_words": changed_coarse,
            "different_treatment_weight_words": changed_weights,
            "original_and_zero_checkpoint_sha256": sha(&baseline_bytes),
            "active_checkpoint_sha256": sha(&active_bytes),
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_rows_match_operating_shape() {
        let (tokens, mask) = fixture();
        assert_eq!(tokens.len(), 4 * STRIDE);
        assert_eq!(mask.iter().map(|&m| u64::from(m)).sum::<u64>(), LABELS);
        for (row, (&end, &opening)) in LENGTHS.iter().zip(&OPENINGS).enumerate() {
            let base = row * STRIDE;
            let end = end as usize;
            let opening = opening as usize;
            assert_eq!(tokens[base], 0);
            assert_eq!(tokens[base + end - 1], 1);
            assert!(tokens[base + 1..base + end - 1]
                .iter()
                .all(|&id| (2..4096).contains(&id)));
            assert!(mask[base..base + opening].iter().all(|&m| m == 0));
            assert!(mask[base + opening..base + end].iter().all(|&m| m == 1));
            assert!(mask[base + end..base + STRIDE].iter().all(|&m| m == 0));
        }
    }

    #[test]
    fn short_workspace_is_refused() {
        assert!(compare_forward(&[], &[]).is_err());
    }
}
