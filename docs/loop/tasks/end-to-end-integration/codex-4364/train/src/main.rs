#![deny(unsafe_code)]
#![recursion_limit = "256"]

#[cfg(feature = "cuda")]
include!(concat!(env!("OUT_DIR"), "/opening-backend.rs"));
#[cfg(feature = "cuda")]
mod capacity;
#[cfg(feature = "cuda")]
#[path = "../opening_margin.rs"]
mod rank;
#[cfg(feature = "cuda")]
mod real;
#[cfg(feature = "cuda")]
mod physical {
    use super::cuda_backend::{gpu::CudaV3ButterflyEngine, FlatV3AdamWBuffers};
    use super::rank::strongest_rival_margin;
    use crate::real;
    use uor_r4_native_vector::{
        butterfly_rotor::LossKind,
        butterfly_rotor_v3::{ButterflyV3Config, ButterflyV3Model},
    };

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const DIM: usize = 512;
    const VOCAB: usize = 4096;
    const LR: i32 = 512;
    const WD_SHIFT: u32 = 19;
    const EPS: u64 = 16;
    const MARGIN: i32 = 1 << 22;
    const GAIN: i32 = 1 << 18;
    const TOKENS: [u16; 8] = [3, 4, 2, 1, 3, 5, 64, 1];
    const MASK: [u8; 8] = [0, 0, 1, 1, 0, 0, 1, 1];
    const OPENING: [u32; 2] = [2, 2];

    fn ensure(test: bool, message: &str) -> Result<()> {
        if !test {
            return Err(message.into());
        }
        Ok(())
    }

    fn dot(hidden: &[i32], row: &[i16]) -> i32 {
        let shift = ButterflyV3Model::head_dot_shift(DIM);
        let sum = hidden
            .iter()
            .zip(row)
            .map(|(&a, &b)| i64::from(a) * i64::from(b))
            .sum::<i64>();
        ((sum + (1_i64 << (shift - 1))) >> shift).clamp(-(1 << 22), 1 << 22) as i32
    }

    fn plant(model: &mut ButterflyV3Model) -> Result<()> {
        let mut session = model.new_session();
        model.step(3, &mut session)?;
        let hidden = model.step(4, &mut session)?.hidden_q15;
        // None of these output slots is consumed in either planted prompt.
        model.coarse_centroids.fill(0);
        for (slot, row) in model.embeddings.chunks_exact_mut(DIM).enumerate() {
            if ![1, 2, 3, 4, 5, 64].contains(&slot) {
                row.fill(0);
            }
        }
        for (coordinate, &h) in hidden.iter().enumerate() {
            model.embeddings[coordinate] = (h.signum() * 32_767) as i16;
            model.embeddings[16 * DIM + coordinate] = (h.signum() * 16_000) as i16;
            model.coarse_centroids[3 * DIM + coordinate] = (h.signum() * 32_767) as i16;
        }
        Ok(())
    }

    fn expected(model: &ButterflyV3Model, masks: &[u64; 65]) -> Result<Vec<i64>> {
        let mut trace = Vec::new();
        for row in TOKENS.chunks_exact(4) {
            let mut session = model.new_session();
            model.step(row[0], &mut session)?;
            let output = model.step(row[1], &mut session)?;
            let gold = row[2] as usize;
            let group = gold / 64;
            let offset = group * 64 * DIM;
            let fine_scores = std::array::from_fn(|slot| {
                let start = offset + slot * DIM;
                dot(&output.hidden_q15, &model.embeddings[start..start + DIM])
            });
            if group == 0 {
                ensure(
                    output.greedy_coarse != 0,
                    "fixture missed wrong-elected-group case",
                )?;
                ensure(
                    fine_scores[0] > *fine_scores[1..].iter().max().ok_or("fine scores missing")?,
                    "fixture missed strictly strongest masked BOS slot",
                )?;
            }
            for (scores, mask, target) in [
                (&output.coarse_scores, masks[0], group),
                (&fine_scores, masks[1 + group], gold % 64),
            ] {
                let rank = strongest_rival_margin(scores, mask, target, MARGIN, GAIN)
                    .map_err(|err| format!("{err:?}"))?
                    .ok_or("fixture has no rival")?;
                ensure(rank.derivative_q15 > 0, "fixture rank inactive")?;
                trace.extend([
                    rank.rival as i64,
                    rank.gap_q15,
                    rank.loss_q15,
                    i64::from(rank.derivative_q15),
                ]);
            }
        }
        Ok(trace)
    }

    fn opening_gap(model: &ButterflyV3Model) -> Result<(i64, i64)> {
        let mut session = model.new_session();
        model.step(TOKENS[0], &mut session)?;
        let out = model.step(TOKENS[1], &mut session)?;
        let gold = TOKENS[2] as usize;
        let fine = std::array::from_fn(|slot| {
            dot(
                &out.hidden_q15,
                &model.embeddings[slot * DIM..(slot + 1) * DIM],
            )
        });
        let coarse = strongest_rival_margin(&out.coarse_scores, u64::MAX, 0, MARGIN, GAIN)
            .map_err(|err| format!("{err:?}"))?
            .ok_or("missing coarse rival")?;
        let token = strongest_rival_margin(&fine, u64::MAX & !1, gold, MARGIN, GAIN)
            .map_err(|err| format!("{err:?}"))?
            .ok_or("missing fine rival")?;
        ensure(
            [
                out.coarse_scores[0],
                out.coarse_scores[coarse.rival],
                fine[gold],
                fine[token.rival],
            ]
            .iter()
            .all(|value| value.abs() < (1 << 22)),
            "planted gap witness touched score clamp",
        )?;
        Ok((coarse.gap_q15, token.gap_q15))
    }

    pub fn run(historical: &std::path::Path) -> Result<()> {
        ensure(
            std::env::var("CUDA_DISABLE_PTX_JIT").as_deref() == Ok("1"),
            "requires CUDA_DISABLE_PTX_JIT=1",
        )?;
        let old_cubin = std::fs::read(historical)?;
        ensure(
            real::sha(&old_cubin) == real::HISTORICAL_CUBIN_SHA,
            "historical #4361 cubin SHA mismatch",
        )?;
        let mut model = ButterflyV3Model::new(ButterflyV3Config {
            vocab: VOCAB,
            dim: DIM,
            layers: 6,
            ffn_branches: 4,
            pkm_sub_keys: 64,
            pkm_key_dim: 32,
            loss_kind: LossKind::HuiBelkinRescaledSquare { pos_weight_k: 64 },
            seed: 20261128,
        })?;
        model.step = 3333;
        model.set_vocab_perm((0..VOCAB as u16).collect())?;
        plant(&mut model)?;
        let mut masks = [u64::MAX; 65];
        masks[1] &= !1;
        let expected_trace = expected(&model, &masks)?;
        let initial = model.new_adamw_state();
        let mut baseline =
            CudaV3ButterflyEngine::from_historical_cubin(&model, &initial, 2, 3, &old_cubin)?;
        ensure(
            baseline.ctx.compute_capability()? == (8, 9),
            "requires sm_89",
        )?;
        let mut sham = CudaV3ButterflyEngine::new(&model, &initial, 2, 3)?;
        let mut active = CudaV3ButterflyEngine::new(&model, &initial, 2, 3)?;
        let (loss, labels) = baseline.accumulate_batch(&TOKENS, &MASK, &[4; 2], 4, true)?;
        let (sham_loss, sham_labels, sham_trace) = sham.accumulate_opening_batch(
            &TOKENS, &MASK, &[4; 2], 4, true, &OPENING, &masks, MARGIN, [0; 2],
        )?;
        let (active_loss, active_labels, active_trace) = active.accumulate_opening_batch(
            &TOKENS, &MASK, &[4; 2], 4, true, &OPENING, &masks, MARGIN, [GAIN; 2],
        )?;
        ensure(
            (loss, labels) == (sham_loss, sham_labels)
                && (loss, labels) == (active_loss, active_labels)
                && labels == 4,
            "base losses and supervised labels must match",
        )?;
        let baseline_grad = baseline.download_grads()?;
        ensure(
            baseline_grad == sham.download_grads()?,
            "zero-gain gradient parity",
        )?;
        let mut reference_grad = model.zero_grad();
        let mut reference_labels = 0;
        let mut reference_loss = 0;
        for (tokens, mask) in TOKENS.chunks_exact(4).zip(MASK.chunks_exact(4)) {
            let (loss, labels) =
                model.accumulate_sequence_grad(tokens, mask, &mut reference_grad, true)?;
            reference_labels += labels;
            reference_loss += u128::from(loss);
        }
        ensure(
            (loss, labels) == (reference_loss, reference_labels),
            "native Rust base loss/labels",
        )?;
        ensure(
            baseline_grad == FlatV3AdamWBuffers::pack_grad(&model, &reference_grad),
            "native Rust base gradients",
        )?;
        ensure(
            active_trace.len() == 20 && sham_trace.len() == 20,
            "opening trace shape",
        )?;
        for ((active_stage, sham_stage), expected_stage) in active_trace
            .chunks_exact(5)
            .zip(sham_trace.chunks_exact(5))
            .zip(expected_trace.chunks_exact(4))
        {
            ensure(
                &active_stage[..4] == expected_stage,
                "device rank did not use expected stage, eligibility or rival",
            )?;
            ensure(
                sham_stage[..2] == expected_stage[..2] && sham_stage[2..] == [0; 3],
                "sham rank or applied loss mismatch",
            )?;
        }
        let active_grad = active.download_grads()?;
        ensure(
            active_grad[..VOCAB * DIM] != baseline_grad[..VOCAB * DIM],
            "fine tied rows received no rank gradient",
        )?;
        ensure(
            active_grad[VOCAB * DIM..(VOCAB + 64) * DIM]
                != baseline_grad[VOCAB * DIM..(VOCAB + 64) * DIM],
            "coarse rows received no rank gradient",
        )?;
        ensure(
            active_trace[5] == 16,
            "fixture missed the planted fine rival",
        )?;
        let mut switched_model = model.clone();
        for coordinate in 0..DIM {
            switched_model
                .embeddings
                .swap(16 * DIM + coordinate, 17 * DIM + coordinate);
        }
        let mut switched = CudaV3ButterflyEngine::new(&switched_model, &initial, 2, 3)?;
        let (_, _, switched_trace) = switched.accumulate_opening_batch(
            &TOKENS, &MASK, &[4; 2], 4, true, &OPENING, &masks, MARGIN, [GAIN; 2],
        )?;
        ensure(
            switched_trace[5] == 17 && switched_trace[6..9] == active_trace[6..9],
            "planted rival swap failed to preserve rank gap while changing rival",
        )?;
        let switched_gradient = switched.download_grads()?;
        ensure(
            switched_gradient[16 * DIM..18 * DIM] != active_grad[16 * DIM..18 * DIM],
            "planted rival swap failed to change the tied row derivatives",
        )?;
        let mut reference = model.clone();
        let mut ref_state = reference.new_adamw_state();
        reference.apply_adamw_step(
            &reference_grad,
            &mut ref_state,
            LR,
            WD_SHIFT,
            EPS,
            true,
            true,
        );
        let expected = FlatV3AdamWBuffers::pack(&reference, &ref_state);
        for engine in [&mut baseline, &mut sham] {
            engine.apply_adamw_step(reference.step, LR, WD_SHIFT, EPS, true, true)?;
            let mut observed = model.clone();
            let mut state = observed.new_adamw_state();
            engine.download_into(&mut observed, &mut state)?;
            let observed = FlatV3AdamWBuffers::pack(&observed, &state);
            ensure(
                observed.weights == expected.weights
                    && observed.adam_m == expected.adam_m
                    && observed.adam_v == expected.adam_v,
                "zero-gain full native AdamW parity",
            )?;
        }
        active.apply_adamw_step(reference.step, LR, WD_SHIFT, EPS, true, true)?;
        let mut updated = model.clone();
        let mut updated_state = updated.new_adamw_state();
        active.download_into(&mut updated, &mut updated_state)?;
        updated.step = reference.step;
        let actual = FlatV3AdamWBuffers::pack(&updated, &updated_state);
        ensure(
            actual.weights != expected.weights && actual.adam_m != expected.adam_m,
            "nonzero gain must change checkpoint words and optimizer moments",
        )?;
        let reloaded = ButterflyV3Model::from_bytes(&updated.to_bytes())?;
        ensure(reloaded == updated, "exact artifact roundtrip")?;
        let before_gaps = opening_gap(&model)?;
        let sham_gaps = opening_gap(&reference)?;
        let active_gaps = opening_gap(&reloaded)?;
        ensure(
            (active_gaps.0 > before_gaps.0 && active_gaps.0 > sham_gaps.0)
                || (active_gaps.1 > before_gaps.1 && active_gaps.1 > sham_gaps.1),
            "no planted unclamped opening gap moved in the predicted direction vs parent and sham",
        )?;
        let (masses, ids) = active.eval_records(&TOKENS[..4], &MASK[..4], &[0], &[4], false)?;
        let mut session = reloaded.new_session();
        for (index, pair) in TOKENS[..4].windows(2).enumerate() {
            let out = reloaded.step(pair[0], &mut session)?;
            if MASK[index + 1] == 1 {
                ensure(
                    ids[index + 1] == out.greedy_token
                        && masses[index + 1]
                            == reloaded.target_q48_mass_from_hidden(&out.hidden_q15, pair[1])?,
                    "post-reload ordinary CUDA/native replay diverged",
                )?;
            }
        }
        // Repeat on EOS-only positive masks with the auxiliary gain still live.
        let mut excluded = CudaV3ButterflyEngine::new(&model, &initial, 2, 3)?;
        let mask = [0, 0, 0, 1, 0, 0, 0, 1];
        let baseline_eos = excluded.accumulate_batch(&TOKENS, &mask, &[4; 2], 4, true)?;
        let baseline_eos_grad = excluded.download_grads()?;
        excluded.zero_grads()?;
        let (eos_loss, eos_labels, eos_trace) = excluded.accumulate_opening_batch(
            &TOKENS, &mask, &[4; 2], 4, true, &[0; 2], &masks, MARGIN, [GAIN; 2],
        )?;
        ensure(
            (eos_loss, eos_labels) == baseline_eos
                && eos_labels == 2
                && eos_trace
                    .chunks_exact(5)
                    .all(|stage| stage == [-3, 0, 0, 0, 0])
                && excluded.download_grads()? == baseline_eos_grad,
            "EOS-only fixture changed",
        )?;
        ensure(
            excluded
                .accumulate_opening_batch(
                    &TOKENS, &mask, &[4; 2], 4, true, &OPENING, &masks, MARGIN, [GAIN; 2],
                )
                .is_err(),
            "opening on an unsupervised position was accepted",
        )?;
        ensure(
            excluded
                .accumulate_opening_batch(
                    &TOKENS, &mask, &[4; 2], 4, true, &[3; 2], &masks, MARGIN, [GAIN; 2],
                )
                .is_err(),
            "EOS accepted as opening",
        )?;
        ensure(
            excluded.download_grads()? == baseline_eos_grad,
            "prelaunch invalid-row refusal altered gradients",
        )?;
        let mut invalid_masks = masks;
        invalid_masks[1] &= !(1 << TOKENS[2]);
        ensure(
            excluded
                .accumulate_opening_batch(
                    &TOKENS,
                    &MASK,
                    &[4; 2],
                    4,
                    true,
                    &OPENING,
                    &invalid_masks,
                    MARGIN,
                    [GAIN; 2],
                )
                .is_err(),
            "inadmissible gold accepted",
        )?;
        ensure(
            excluded.download_grads()?.iter().all(|&g| g == 0),
            "device-side invalid gold left reusable gradients",
        )?;
        let mut registered = CudaV3ButterflyEngine::new(&model, &initial, 1, 3)?;
        let (legacy_loss, legacy_labels, _) = registered.accumulate_opening_batch(
            &TOKENS[..4],
            &MASK[..4],
            &[4],
            4,
            true,
            &OPENING[..1],
            &masks,
            real::MARGIN,
            [0; 2],
        )?;
        let registered_sham = registered.download_grads()?;
        registered.zero_grads()?;
        let (registered_loss, registered_labels, registered_trace) = registered
            .accumulate_opening_batch(
                &TOKENS[..4],
                &MASK[..4],
                &[4],
                4,
                true,
                &OPENING[..1],
                &masks,
                real::MARGIN,
                [real::GAIN; 2],
            )?;
        let registered_grad = registered.download_grads()?;
        let embedding_words = VOCAB * DIM;
        let different_fine_words = registered_grad[..embedding_words]
            .iter()
            .zip(&registered_sham[..embedding_words])
            .filter(|(a, b)| a != b)
            .count();
        let different_coarse_words = registered_grad[embedding_words..embedding_words + 64 * DIM]
            .iter()
            .zip(&registered_sham[embedding_words..embedding_words + 64 * DIM])
            .filter(|(a, b)| a != b)
            .count();
        ensure(
            (registered_loss, registered_labels) == (legacy_loss, legacy_labels)
                && registered_trace[3] == i64::from(real::GAIN)
                && registered_trace[8] == i64::from(real::GAIN)
                && different_fine_words > 0
                && different_coarse_words > 0,
            "registered margin and gain failed to alter quantized gradients at both stages",
        )?;
        println!(
            "{}",
            serde_json::json!({
                "verdict": "PASS",
                "protocol": "uor-opening-finite-v1",
                "gpu_uuid": real::gpu_uuid(&baseline)?,
                "cubin_sha256": real::cubin_sha(),
                "historical_cubin_sha256": real::sha(&old_cubin),
                "dim": DIM,
                "gaps": [before_gaps, sham_gaps, active_gaps],
                "registered_constants": {
                    "margin": real::MARGIN, "gain": real::GAIN, "trace": registered_trace,
                    "different_fine_gradient_words": different_fine_words,
                    "different_coarse_gradient_words": different_coarse_words
                },
                "cases": ["sham and Rust full-step parity", "two live native stages",
                    "different winner changes tied-row derivatives", "masked BOS",
                    "wrong elected cell", "zero auxiliary on EOS-only labels"]
            })
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() -> std::process::ExitCode {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let result = match args.as_slice() {
        [mode, historical] if mode == "finite" => physical::run(historical.as_ref()),
        [mode, parent, historical] if mode == "capacity" => capacity::run(parent.as_ref(), historical.as_ref()),
        [mode, parent, roster, finite_receipt, historical, output] if mode == "one-update" => real::run(
            parent.as_ref(),
            roster.as_ref(),
            finite_receipt.as_ref(),
            historical.as_ref(),
            output.as_ref(),
        ),
        _ => Err(
            "usage: checker finite HISTORICAL_CUBIN | checker capacity PARENT HISTORICAL_CUBIN | checker one-update PARENT ROSTER FINITE_RECEIPT HISTORICAL_CUBIN NEW_OUTPUT"
                .into(),
        ),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("REFUSED: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "cuda"))]
compile_error!("build this GPU checker with --features cuda");
