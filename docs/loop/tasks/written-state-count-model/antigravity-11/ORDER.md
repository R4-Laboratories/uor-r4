# ORDER: written-state-count-model, antigravity-11

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Base: `b2c458e7b`.

## Goal

Pre-register and execute **Stage B3c (`P3` GPU repair, [`DECISION.md §4`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320))** targeting the three empirical mechanisms identified in `antigravity-10` while keeping sealed `DEV-test` strictly untouched:

1. **`A1s`-warm-started `S6` + multi-order `S2` (`<= 100 MB`) + `S3` Brier/EM ablation (`R` -> `DEV-tune`, `UORS6B01`):** Warm-start `S6` from Stage B1a `P1(A1s)` (`125,750,699,315` Q12 / `1,400,045` micro-BPB) instead of uniform weights, replace 22 MB `S2` order-2 with bounded `<= 100 MB` (`<= 25 MB/order`) injective 64-bit `S2` orders `1..=4` `i8` Brier error-written tables (`14*ctxs + 7*rows`), and compare conservative Brier `S6` updates against pure-integer EM `S6` refits over `S3`/`S2` Brier error cells (`p3c_a1s_brier_s6`, `p3c_a1s_em_s3_brier`, `p3c_a1s_em_s2_multi_brier`, `p3c_a1s_brier_s2_multi`) on `DEV-tune` (`1,000`-resample complete-record block bootstrap `seed = 4390` vs `P1(A1s)` and `0.95/0.90/0.80 × R*`).
2. **Balanced `S4` + `D87 R` `15.9M`-token co-occurrence propagation + active `s4_hinge_step` (`UORS4P01`):** Remove uneven `TRAIN_MWORLD` query duplication and `4:1` pre-stuffing, extract pure-integer distributional co-occurrence associations (`±8` window over all `15,947,584` `D87 R` tokens) from `TRAIN_MWORLD`-derived anchors (`0` eval-panel words), run active `s4_hinge_step` (`margin = 16 * P3_Q16_ONE`, `step = 8`), and score `R1`/`R2` (`m2-fresh256` `256` rows / `128` pairs) and `R4` (`240` rows: `120` in-stem + `120` OOV-synonym) via GPU `p3_s4_vote_batch` against `NoMemory` and `ShuffledRole`.

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` tokens (`V = 4096`), `UORKNC03` (`full-c.v3`), `A1s` `UORS5W02`, `golden_shell` `UORS3T02`, injective 64-bit `S2` orders `1..=4` `i8` error cells (`<= 100,000,000` B), `UORS6B01`, `UORS4P01`, and `LawfulStopConfig`. Zero floats, zero matmul, zero softmax.
2. **Shared operators:** Standalone `p3c-eval` depends on `uor_r4_count_runtime::p3_state` (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`), with mirrored pure-integer `p3_brier_kernels.cu` verified via PTX/SASS 0-float audit and finite/runtime parity.
3. **Transition parity:** GPU vs CPU finite parity (`0` diffs across all 5 kernels including `p3_em_step_batch` and 64-bit `p3_s2_lookup_and_accum`), `65,536 == 65,521` batch parity, and `>= 12,500` spot-check probes (`0` divergences).
4. **Update rule:** Pure-integer EM & conservative Brier simplex steps + `i8` error-writes on `R`; `s4_hinge_step` on `TRAIN_MWORLD` (`updates_applied > 0`).
5. **Lossless serialization:** Whole-payload SHA-256 + `train_sha256` binding on `UORS4P01` and `UORS6B01`, sealed `GO.json`, and `evidence/receipts.sha256`.

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-11/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `pro` `FIT.md` (`TEST FITNESS: FIT`), `GO.json`, `rustfmt --check`, `cargo test`, `clippy -D warnings`, PTX/SASS 0-float audit, and `receipts.sha256` pass on archived Wick evidence with `DEV-test` untouched and `0` pods owned by `antigravity`.
2. `REPORT.md` (negatives first), `m4-serving.md`, `written-state-count-model.md`, fresh `pro` exact-head review (`VERDICT: APPROVE`), and CI merge to `main` pass.
