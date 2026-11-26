# ORDER: written-state-count-model, antigravity-10

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Base: `8a8b0cf3b`.

## Goal

Pre-register and execute **Stage B3b (`P3` GPU evaluation, [`DECISION.md §4`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320))** over the pure-integer runtime landed in Stage B3a (`#4275`), keeping sealed `DEV-test` strictly locked and untouched:

1. **Integer Brier error-write (`R` -> `DEV-tune`, `UORS6B01`):** Run 3 epochs of GPU pure-integer Brier LMS updates (`p3_brier_kernels.cu` + `p3_state::{project_simplex_q32, apply_s3_s4_err_write}`) over Stage B1a count/Pick experts (`A1s`, `125,750,699,315` Q12 / `1,400,045` micro-BPB baseline), `S3` `golden_shell` (`e4c4d1a8...`), and a top-order `S2` `i8` error-cell arm (`S6`, `S6+S3`, `S6+S2_err`), checking monotonic Brier decrease on `R`, `>= 1%` `DEV-tune` bit gain over count-only (`1,000`-resample complete-record block bootstrap, `seed = 4390`), and standing against `0.95 / 0.90 / 0.80 × R*` (`DEV-tune` `R* = 1,432,318` micro-BPB).
2. **Pointer-valued `S4` (`UORS4P01`) + hinge + `R1` / `R2` / `R4` reply panels:** Count pointer-valued `S4` from `D87 TRAIN` + domain-tagged `ft/M-World` TRAIN (0 overlap with `ft/dev`, `m2-fresh256`, and `R4`), run `s4_hinge_step`, apply `apply_lawful_stop_and_rep_guard` with `UORKNC03` `full-c.v3` + `S5` Pick, and score single-shot free generation on sealed `m2-fresh256` (`R1 >= 52/128` pairs, `R2 >= 200/256` rows via unchanged `FrozenAnswers` + `pair_id`) and fresh `R4` (`>= 200` held-out-phrasing rows) against `NoMemory` and `ShuffledRole` controls via `paired_band_compare`.

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` tokens (`V = 4096`), `UORKNC03` count/Pick state (`A1s`), `golden_shell` `UORS3T02`, `S4` pointer rows (`UORS4P01`, `i8 err_cell`), `S6` Brier weights (`UORS6B01`), `S2` `i8` error cells, and `LawfulStopConfig` traces. Zero floats, zero matmul, zero softmax.
2. **Shared operators:** Standalone `p3-eval` depends directly on `uor_r4_count_runtime::p3_state` (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`), with mirrored pure-integer `p3_brier_kernels.cu` verified via PTX/SASS 0-float audit and finite/spot-check parity.
3. **Transition parity:** GPU vs CPU `C128` finite Brier parity (`0` diffs), shifted batch determinism (`65,536 == 65,521`), and `>= 12,500` spot-check replay probes (`0` divergent words).
4. **Update rule:** Exact Q32 simplex Brier LMS error-write (`brier_error_write_epoch`) + `i8` cell updates on `R`; example-ordered `s4_hinge_step` on `TRAIN + ft/M-World TRAIN`.
5. **Lossless serialization:** Whole-payload SHA-256 + `train_sha256` binding on `UORS4P01` and `UORS6B01`, sealed `GO.json`, and `evidence/receipts.sha256`.

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-10/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `pro` `FIT.md` (`TEST FITNESS: FIT`), `GO.json`, `rustfmt --check`, `cargo test`, `clippy -D warnings`, PTX/SASS 0-float audit, and `receipts.sha256` all pass on archived Wick/pod evidence, with zero `DEV-test` access and all rented pods torn down (`0` pods owned by `antigravity`).
2. `REPORT.md` (negatives first), `m4-serving.md`, `written-state-count-model.md`, fresh `pro` exact-head review (`VERDICT: APPROVE`), and CI merge to `main` pass.
