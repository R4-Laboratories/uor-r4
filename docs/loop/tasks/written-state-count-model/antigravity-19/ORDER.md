# ORDER: written-state-count-model, antigravity-19 (Phase P5 PR 2 of 2 — Wick RTX 4090 mixed-mass Brier S1 training & sealed DEV-test gate)

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P5`, [`DECISION.md` §4 `P5`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L335-L346)). Base: `9631cc2fd` (PR `#4306`).

## Goal & scope

Execute the pre-registered, non-author `FIT`-gated **Phase `P5` PR 2 of 2** qualification on Wick's RTX 4090 (`$0.00` pod spend, `0` pods; allowed scope `docs/loop/tasks/written-state-count-model/antigravity-19/**`, `docs/research/notebook/m4-serving.md`, `docs/roadmap/items/written-state-count-model.md`; `0` `crates/` or root `Cargo.toml` edits):

1. **Gate 1 (AOT `sm_89` audit):** Build offline `p5-train/` (`nvcc -cubin -arch=sm_89 -O3`, `CUDA_DISABLE_PTX_JIT=1`, `0` floats, `0` MMA/matmul, `0` hardware variable division) and verify `0` PTX/SASS float/MMA opcodes and all required kernel symbols.
2. **Gate 2 (Finite GPU-vs-CPU parity & `4,096`-row `Driver::run_block_transitions` check):** Assert `0` raw-word diffs between `p5_kernels.cu` and `uor_r4_discrete::p5_brier_core` + `P4Session::step` (`K = 1, 8, 16, 64`, `|S_t| = 0, 1, >= 2` joint-accept/reject, `resulting_port` tie-break, `L <= 2,048`, target shift), and run live `4,096`-row `Driver::run_block_transitions` / `Device::score_rows_k` on Wick RTX 4090 via `uor-r4-discrete --features cuda`.
3. **Gate 3 (`R` `rows-K` + `BVG-K` training vs equal-trial controls):** Holding `p4.d11`'s `S0`/`S2`/`S3`/`S4`/`S6` frozen (`P4_MAX_CTX = 2048`), train `16,384` `S1` coordinates (`<= 1e5`) on `R` (`15,947,584` targets, `3` epochs = `11,679` `4,096`-row batches = `45.6` sweeps = `11,211,840` trials/arm at `K = 64`) for `s1_p5_from_seed1`, `ctrl_reseed_canonical`, and `ctrl_reseed_random` (`DEV-tune` checkpoints only; `0` `DEV-test` peeks).
4. **Gate 4 (Single-shot sealed `DEV-test` gate, `DECISION.md:342`):** Score all `5` frozen arms (`baseline_seed1`, `s1_p5_from_seed1`, `ctrl_reseed_canonical`, `ctrl_reseed_random`, `ctrl_reseed_static`) on all `5,321` records / `21,349,576` targets with full-vocab Q32 Brier, Q12 BPB, 1,000-draw paired record bootstrap 95% CIs, CPU `P4Session` spot checks (`0` diffs), and determinism repeat (`0` diffs). Pass iff `DEV-test` Brier clears `>= 1% / GPU-day` (non-saturated, beating both equal-trial controls, 95% CI `< 0`) without raising BPB (then pack `p5.d11` `<= 4 GB`, verify parity, re-score Bar A); otherwise freeze `S1` at `seed1.d11` and close the `P5` `S1`-Pick configuration.

## Five native-space items (`D85`, `native-space-cycle`)

1. **State schema:** `P5S1State` (`16,384` `Cl(0,6)` orbit roots `0..16`, `epoch`, `UORP5S10`) + frozen `p4.d11` (`UORP4A01`, `UORKNC03`, `UORS3C01`, `UORS2C01`, `UORS4P01`, `UORS6B01`, `CausalSessionCache`).
2. **Shared operators:** `p5_kernels.cu` runs the exact integer `P4Session::step`, `code_pick` (`P4_MAX_CTX = 2048`), `P5RowBrier`, `rows-K`, and `BVG-K` (`resulting_port` tie-break, `L(S_t) < L({best})`) operators shared with `p5_brier_core` and `p4_serving`.
3. **Replay oracle:** Gate 2 asserts `0` diffs vs `p5_brier_core` & `P4Session::step`; Gate 4 asserts `0` diffs on `>= 512` CPU `P4Session` spot checks and determinism repeat.
4. **Update rule:** Checked `u64` full-vocab Q32 Brier sums over `4,096` rows, strict improvement, joint verify, in-place `MOVE_ROOT` composition, exact `15 * K` equal-trial controls.
5. **Lossless serialization:** Self-describing `UORP5S10` checkpoints and (on pass) `rewrite_uorlut01_s1` -> `uor-r4.lut-discrete-p4/1` (`p5.d11`), `receipts.sha256` sealed.
