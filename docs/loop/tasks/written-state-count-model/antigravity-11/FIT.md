# Non-Author Test Fitness Review: `antigravity-11` (`written-state-count-model`, Stage B3c `P3` GPU Repair)

Reviewer: `Stage B3c Non-Author Test Fitness Reviewer R4` (`pro`, confirming R3 after 3 `--features cuda` clippy alias cleanups in `cuda.rs`)  
Verdict: **`TEST FITNESS: FIT`**

---

Independent non-author pre-run TEST FITNESS review for Stage B3c (`written-state-count-model`, `antigravity-11`) at `/Users/casey.allard/uor-r4-b3c-p3-repair/docs/loop/tasks/written-state-count-model/antigravity-11/`.

All 12 package files (`ORDER.md`, `SHEET.md`, `dispatch.sh`, `p3c-eval/Cargo.toml`, `p3c-eval/Cargo.lock`, `p3c-eval/build.rs`, `p3c-eval/src/{main,cuda,s3_decode,brier_eval,reply_eval}.rs`, `p3c-eval/src/p3_brier_kernels.cu`) were read and audited against `CHARTER.md`, `PLAN.md`, `docs/research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320`, and `crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs`.

## Audit Findings

1. **Zero floats, MMA/Tensor Core ops, softmax/squashing, matmul, or runtime CUDA division:**
   - `p3_brier_kernels.cu` uses strictly integer types (`U8`, `I8`, `U16`, `U32`, `U64`, `I64`), `__umul64hi`, unsigned 64-bit `atomicAdd`, and restoring binary division `mul_div_u64`. The only `/` and `%` tokens in `p3_brier_kernels.cu` are compile-time macro `(P3_Q48_ONE / P3_VOCAB)` (line 16) and constant divisors `6561u` / `9u` in `Starling-7` coordinate unpacking (lines 291–298) — zero runtime `/` or `%` on variable divisors. `mul_div_u64` invariants hold at all 4 call sites (`wm_q48 <= mixed`, `eff_hit <= eff_total`, `e_counts[idx] <= total`), with zero `u64` atomic accumulator overflow within `CUDA_CHUNK = 65_536` (`<= 2^48`) and `R_TARGETS = 15_947_584` (`<= 2.04e9`).
   - `p3c-eval/build.rs` and `src/*.rs` contain zero float types/ops, zero MMA/Tensor Core ops, zero softmax/squashing, and zero matmul; all Rust integer `/` and `%` match `p3_state.rs` under D81 (hardware integer divide on CPU/serving/evaluation).
   - `dispatch.sh` Gate 1 (`lines 125–142`) enforces `PTX_FLOAT_HITS == 0` on `nvcc -ptx` and `FLOAT_HITS == 0` on `cuobjdump --dump-sass` before any kernel runs.

2. **Safety (`forbid(unsafe_code)`), `cuda` isolation, and compile-cleanliness:**
   - `#![forbid(unsafe_code)]` is present at line 1 of `build.rs`, `s3_decode.rs`, `brier_eval.rs`, and `reply_eval.rs`.
   - `main.rs:1` enforces `#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`.
   - In `cuda.rs`, all code and all 5 `unsafe` kernel launch sites (lines 162, 257, 424, 482, 619) reside exclusively inside `#[cfg(feature = "cuda")] mod gpu` with `// SAFETY:` justifications.
   - `nvcc` is not present on this macOS host, so `--features cuda` was verified by full manual audit across all 6 Rust files (`main.rs`, `cuda.rs`, `s3_decode.rs`, `brier_eval.rs`, `reply_eval.rs`, `build.rs`): all `#[cfg(feature = "cuda")]` / `#[cfg(not(feature = "cuda"))]` / `#[cfg(any(test, not(feature = "cuda")))]` gates, `#[allow(clippy::too_many_arguments)]`, `#[allow(unused_mut)]`, imports, and all 5 CUDA kernel argument signatures match `p3_brier_kernels.cu` in count, order, and types with zero expected warnings or errors on the pod.

3. **Hard lock on `DEV-test`:**
   - `dispatch.sh:15-18` scans all string values in `GO.json` for `dev-test|rstar-devtest|023784b7|ec048217` (case-insensitive) and exits `99` on any match, and pins all `DEV-tune` input SHA-256s (`lines 71-76`).
   - `brier_eval.rs:119-179` (`DevTuneData::load_checked`) refuses `SEALED_DEV_TEST_CENSUS_SHA256` (`ec048217...`) and `SEALED_DEV_TEST_UORT_SHA256` (`023784b7...`), requires exact `DEV_TUNE_CENSUS_SHA256` (`90fbc73a...`), `split == "dev-tune"`, `2,670` records, `10,810,087` tokens, and `10,807,417` targets.

4. **Multi-order `S2` (`1..=4`):**
   - `s3_decode.rs:443-461` (`pack_s2_key`): injective 64-bit encoding `(u64::from(order) << 48)` plus 12-bit packed context tokens `prev1..prev_order` (zero hash collisions).
   - `s3_decode.rs:472-573` (`S2OrderErrorTable::build_order_from_stream`): resets `(p1..p4) = 0` at every `tok == 0` record boundary (matching `stream_golden_and_multi_s2_keys:743-776`), aggregates sorted events, ranks contexts by descending `context_total` (deterministic key tie-break), enforces `14 * ctxs + 7 * rows <= 25_000_000` B per order and `<= 100_000_000` B total across orders `1..=4` (`brier_eval.rs:730-733`, `dispatch.sh:168-169`).

5. **`A1s`-warm-start `S6`, EM vs Brier ablation, error-write `lr_shift`, parity, and 4-arm agreement:**
   - `brier_eval.rs:442-496` (`init_a1s_warmstart_weights`) preserves proportional `A1s` bucket weights (`880` buckets, reserving `P3_Q32_ONE >> 7` per extra expert and enforcing exact `2^32` simplex sum with `>= 1` floor).
   - `step_s6_em_weights` (`brier_eval.rs:569-594`, `prior_weight = 64`) pairs bit-exact with `p3_em_step_batch` (`p3_brier_kernels.cu:154-190`); `step_s6_brier_weights` (`brier_eval.rs:596-629`, `lr_shift = 7`, floor `1 << 10`) pairs bit-exact with `p3_brier_step_batch`; `S3` and `S2(1..=4)` error-writes use `lr_shift = 9`.
   - Independent table reset (`reset_dynamic_tables`, `brier_eval.rs:1171-1206`) clears CPU & GPU `S3` and `S2` `err_i8` cells and refreshes `r_19` / `tune_19` between Arms 2, 3, and 4.
   - `65,536 == 65,521` batch parity is verified for both Brier (`brier_eval.rs:871-887`) and EM (`brier_eval.rs:996-1020`).
   - `DEV-tune` spot checks verify `13,153 >= 12,500` probes with `0` divergences across `S6` Brier, EM, `S3`, and `S2(1..=4)`.
   - Exact 4-arm agreement (`p3c_a1s_brier_s6`, `p3c_a1s_em_s3_brier`, `p3c_a1s_em_s2_multi_brier`, `p3c_a1s_brier_s2_multi`) holds across `ORDER.md`, `SHEET.md`, `brier_eval.rs`, and `dispatch.sh`.

6. **Balanced `S4`, `D87 R` `±8` co-occurrence propagation, hinge, and GPU vote parity:**
   - `reply_eval.rs:581-840` derives anchors strictly from `TRAIN_JOB_QUERIES` and `TRAIN_HOME_QUERIES` (excluding top-48 `D87 R` tokens), scans all `15,947,584` `D87 R` tokens in record-bounded `±8` windows (`0` eval-panel words touched), verifies `0` collisions against `HISTORICAL_FT_DEV_POOL` (`ft/dev`), `m2-fresh256`, and `R4` via `verify_disjoint_sets` (`lines 1136-1138`), constructs symmetric `24`-episode (`3x8` literals, `4x6` queries) balanced `M-World` training examples without `4:1` stuffing, runs 3 epochs of `s4_hinge_step` (`margin = 16 * P3_Q16_ONE`, `step_i8 = 8`, asserting `total_hinge_updates > 0`), and checks `s4_gpu_vs_cpu_divergences == 0` across training epochs and both `m2-fresh256` (`256` rows / `128` pairs) and `R4` (`240` rows: `120` in-stem + `120` OOV-synonym) under `Active`, `NoMemory`, and `ShuffledRole`.

## Read-Only Verification Results (`CARGO_TARGET_DIR=/tmp/p3c-eval-audit3`)

- `bash -n dispatch.sh`: **PASS** (exit `0`)
- `rustfmt --edition 2021 --check p3c-eval/build.rs p3c-eval/src/*.rs`: **PASS** (exit `0`)
- `cargo test --manifest-path p3c-eval/Cargo.toml --offline --locked`: **PASS** (`6 passed; 0 failed`)
- `cargo clippy --manifest-path p3c-eval/Cargo.toml --offline --locked --no-deps --all-targets -- -D warnings`: **PASS** (exit `0`, `0` warnings in `p3c-eval`)

TEST FITNESS: FIT

## Verified File SHA-256 Digests

```text
e278486d33da8a93417dc9205114efb7076e978ff1aa67d0e662aa4b4d2f4340  ORDER.md
ffaec1115f3a8218fb0c5015cd21c60821fa45a070f8931783ee3e373747d725  SHEET.md
4cae4af95d588609fa4c5c9bc90b2c6c969fa7b8f0308630ddf15f6d0a00681e  dispatch.sh
894e1430ebf2d715b9a32d9950b2e5499cb256aafa313d0f4c2d80110c4f16ed  p3c-eval/Cargo.toml
e2e0afc49fd2eaa38a2df33c88a37c1fbb5cce9eec8cf0827da8c76c0896885c  p3c-eval/Cargo.lock
bd981dc9f9bd97838d2f82b2793cbb8d223dd3ec7e2556083e2292ebafce7944  p3c-eval/build.rs
f9bcf674fc0906a72562a64472d7dcfc186821910b612726cea804ea10d26250  p3c-eval/src/main.rs
5ece5f06044109f757b675819dbd2987439696f81690f3cbc02fe4c597c71497  p3c-eval/src/cuda.rs
f2d0c603ed7000c3add12d26df3d2849a07e382a6f86b9af57d32b1d89dff0f4  p3c-eval/src/s3_decode.rs
6fe56c66695a6a0b9d26fc01af592225b2e02058a8ece2cefae9ade49d51269b  p3c-eval/src/brier_eval.rs
0f22cff6c85402836061f47f32e2966bf4a1c00dc3bf42ebd59161a8b75335a4  p3c-eval/src/reply_eval.rs
fd0e33ddfc0e9b56d7cbf633ef3c8175d13af5792cf0c16bf340f07080734380  p3c-eval/src/p3_brier_kernels.cu
```
