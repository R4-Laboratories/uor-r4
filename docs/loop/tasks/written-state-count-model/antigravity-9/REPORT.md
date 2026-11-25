# REPORT: written-state-count-model, antigravity-9 (`UORS4P01` / `UORS6B01` Stage B3a Host + WASM `P3` Runtime)

Lab: antigravity. Item: [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Order: [`ORDER.md`](ORDER.md). Base: `e86b0a1bf`.

## What did not work, first

- **Host & WASM runtime/oracle qualification only — no GPU `P3` `DEV-tune`/`DEV-test` or reply-panel run yet:** This Stage B3a unit implements and certifies the shared pure-integer `P3` building blocks on host CPU (`native` ARM64 + `wasm32-unknown-unknown`, `$0.00` pod spend). Pre-registered GPU `P3` training (`S4` on `TRAIN` + domain-tagged `ft/M-World`, Brier error-write over `S3`/`S4`/`S6`, and `R1`/`R2`/`R4` reply panels per [`DECISION.md`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L316)) is **Stage B3b** (`antigravity-10`).
- **Workspace `.cargo/config.toml` `+simd128` flag and ARM64 NEON vectorization caught & resolved during binary audits:**
  - `.cargo/config.toml` enables `-C target-feature=+simd128` workspace-wide on `wasm32-unknown-unknown` (from Stage A5), which caused default WASM builds of `uor-r4-count-runtime` to emit `0xFD` SIMD instructions flagged by `audit_zero_matmul_wasm.py`; building the audit artifact with `RUSTFLAGS="-C target-feature=-simd128"` verified **`333` decoded functions, `float=0`, `unclassified=0`, `errs=0`**.
  - On initial ARM64 `otool -tV` disassembly, `project_simplex_q32` (`1` hit) and `apply_lawful_stop_and_rep_guard` (`8` hits) emitted NEON `fmov` register moves from fixed-length `4096` vector reductions/unrolls; folding the projection sum inside the projection loop and unifying the EOS tie-break/floor scan eliminated every NEON vector move (**`55` `p3_state` symbols, `c3_violations=0`, `fmov_hits=0`**).
  - Pre-existing untouched `s3_state.rs` retains `13` `clippy::type_complexity` warnings under Rust 1.97; `p3_state.rs` (`3` initial lints fixed) and `count_runtime.rs` have **`0` clippy warnings**.
  - In `crates/uor-r4-integer/Cargo.toml`, `p1-count-serving` enables `serving-wasm`, which gates off `uor_r4_integer::lawful_likelihood` (`native_geometry.rs:61-62`, same separation noted in `kn_score.rs:27-30`). Thus `p3_state` and `lawful_likelihood` are verified in adjacent feature lanes without widening `uor-r4-integer/Cargo.toml` outside `ORDER.md`: `p1-count-serving` verifies `apply_lawful_stop_and_rep_guard` preserves `s >= 1` for all `4,096` entries, while `not(feature = "serving-wasm")` (`p3_prll_positive_floor_mass_invariants`) verifies `mle_masses_from_counts` + `bits_q12` on `>= 1` post-guard `4,096`-entry shapes.
  - Initial `pro` adversarial review at `969a4fa6a` flagged four defects (Heron underestimate in `isqrt_u64`, top-8-bit truncation in `unpack_s4_pair_key`, `usize` vs `u64` modulo in `ShuffledRole` on `wasm32`, and dead `#[cfg(not(feature = "serving-wasm"))]` PRLL code) plus `REPORT.md` / `apply_s3_s4_err_write` rounding details; all were fixed (binary restoring `isqrt_u64`, `(key >> 24) != 0` rejection, `state % (k as u64)`, split-lane PRLL verification, and truncating `lr_shift` division) before re-verification.

## What changed

- **Shared pure-integer `P3` core (`#![forbid(unsafe_code)]`) [antigravity]:**
  - [`crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs): Implements order-free `S4` pair keys (`s4_pair_key`, fail-closed 24-bit `unpack_s4_pair_key`, `extract_turn_pair_keys`), `p3_token_primes` + `s4_prime_product_u64` bijection oracle, `S4ConjunctionTable` (`MAGIC_S4 = b"UORS4P01"`, `encode_bytes`, fail-closed `from_bytes`, `vote_candidates`, `pick_pointer`) with all three registered `S4ControlMode` modes (`Active`, `NoMemory`, `ShuffledRole`), `s4_hinge_step` margin trainer, pure-integer Brier engine (`brier_mix_q48`, `integer_brier_target_q32`, exact Euclidean `project_simplex_q32`, `apply_s3_s4_err_write`, `BrierS6Weights` `MAGIC_S6_BRIER = b"UORS6B01"` with `brier_error_write_epoch`), lawful stop/repetition guard (`LawfulStopConfig`, `apply_lawful_stop_and_rep_guard`), and binary restoring `isqrt_u64` / `isqrt_ceil_u64` / `paired_band_compare`.
  - [`crates/uor-r4-discrete/src/p3_state_rows.rs`](../../../../../crates/uor-r4-discrete/src/p3_state_rows.rs), [`crates/uor-r4-discrete/src/lib.rs`](../../../../../crates/uor-r4-discrete/src/lib.rs), and [`crates/uor-r4-discrete/src/count_runtime.rs`](../../../../../crates/uor-r4-discrete/src/count_runtime.rs): Exposes `uor_r4_discrete::p3_state_rows` (with `6` unit tests covering `S4` prime bijection/symmetry, hinge separation vs `NoMemory` & `ShuffledRole`, 3-epoch strictly decreasing Brier error-write & exact `2^32` simplex preservation, 4-gram cycle breaking + EOS termination preserving `s >= 1`, `wasm32`/corruption/`0x1000_0001` key refusals, and exact `isqrt_u64`/`isqrt_ceil_u64`/`paired_band_compare` boundary cases) and `uor_r4_count_runtime::p3_state` over the shared `p3_state.rs` include.
- **Cross-crate integer & wire parity qualification [antigravity]:**
  - [`crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs`](../../../../../crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs#L1878-L2100): `uors4p01_uors6b01_p3_parity_controls_and_refusals` (`#[cfg(feature = "p1-count-serving")]`) checks exact equality between `p3_token_primes()` and `uor_r4_integer::fbs_pair_rows::token_primes(4096)`, `s4_prime_product_u64`, `UORS4P01` & `UORS6B01` wire round-trips, `Active` vs `NoMemory` & `ShuffledRole` paired-band separation (`80` examples), 3-epoch strictly falling Brier error-write, signed `apply_s3_s4_err_write`, positive stop-guard masses + EOS termination, `isqrt_u64(200)` / `isqrt_ceil_u64(200)`, and high-bit `0x1000_0001` key / magic corruption refusals; `p3_prll_positive_floor_mass_invariants` (`#[cfg(not(feature = "serving-wasm"))]`) checks `mle_masses_from_counts` + `bits_q12` on `>= 1` post-guard `4,096`-entry mass vectors.
- [`docs/roadmap/items/written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md): Recorded Stage B3a completion and remaining Stage B3b GPU evaluation dependencies.

## Native-space cycle verification (D85)

1. **Single persistent state schema:** 12-bit `UORT` tokens (`P3_VOCABULARY = 4096`, `P3_BOS = 0`, `P3_EOS = 1`), 24-bit order-free `S4` pair keys, `(role_class, span_offset, count, i8 err_cell)` pointer entries, Q32 simplex weights (`>= floor >= 1`), Q48 masses (`1..=2^48`), and Q32 Brier scores.
2. **Shared typed operators:** `uor_r4_discrete::p3_state_rows` and `uor_r4_count_runtime::p3_state` include the same [`crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs) under `#![forbid(unsafe_code)]`, verified across native ARM64 and `wasm32-unknown-unknown`.
3. **Exact transition replay:** `p3_token_primes() == token_primes(4096)` across all 4,096 vocabulary IDs; `UORS4P01` and `UORS6B01` encode/decode bit-identically with fail-closed digest & structural checks; `apply_lawful_stop_and_rep_guard` preserves `>= 1` positive entries compatible with `mle_masses_from_counts` and `bits_q12`.
4. **Pure-integer updates:** Deterministic example-ordered margin hinge (`s4_hinge_step`) and integer Brier LMS updates (`brier_error_write_epoch`) with exact Euclidean `project_simplex_q32` (`sum == 2^32`, `w >= floor >= 1`).
5. **Lossless serialization:** Self-describing `UORS4P01` and `UORS6B01` formats with checked `wasm32` size math and whole-payload SHA-256 verification.

## Measurements

Host & WASM runtime/parity verification only (`$0.00` rented pod spend; GPU `P3` `DEV-tune` + reply-panel qualification is Stage B3b):

- **Direct WASM audit (`libuor_r4_count_runtime-55e6929a465cdf32.rlib`, `907,914 B`, `sha256: ef609ef1740a24e0f6598142177a63e33e67ee748dd88bff2a2369a62663b67b`, `RUSTFLAGS="-C target-feature=-simd128"`):** `10` WASM code objects, **`333` decoded functions, `float=0`, `unclassified=0`, `errs=0`**.
- **Direct native ARM64 audit (`libuor_r4_count_runtime-08c25bda1ac08409.rlib`, `907,584 B`, `sha256: f7c8b8d99e3c4e512f57d64781d49a01e69a2083b42f7483767b09efd8c2ee76`):** `otool -tV` + `audit_zero_matmul_serving.py::classify_instructions(..., stack=True)` across **`55` `p3_state` symbols: `c3_violations=0`, `fmov_hits=0`** (`scalar_mul=60`, `scalar_div=6`).

## Checks run at the head

- `rustfmt --edition 2021 --check` on `p3_state.rs`, `p3_state_rows.rs`, `count_runtime.rs`, and `discrete_sparse_table_parity.rs` — **PASS**.
- `cargo test --locked -p uor-r4-discrete --lib p3_state_rows` — **6 passed, 0 failed** (`445 filtered out`).
- `cargo test --locked -p uor-r4-count-runtime` — **2 passed, 0 failed**.
- `cargo test --locked -p uor-r4-integer --lib --features serving-wasm,p1-count-serving` — **736 passed, 0 failed, 1 ignored**.
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --test discrete_sparse_table_parity` — **8 passed, 0 failed, 3 ignored benchmarks**.
- `cargo test --locked -p uor-r4-integer --features p1-count-serving --test discrete_sparse_table_parity uors4p01_uors6b01_p3_parity_controls_and_refusals` — **1 passed, 0 failed** (`10 filtered out`).
- `cargo test --locked -p uor-r4-integer --test discrete_sparse_table_parity p3_prll_positive_floor_mass_invariants` — **1 passed, 0 failed** (`8 filtered out`).
- `cargo clippy --locked -p uor-r4-count-runtime --all-targets` — **0 warnings in `p3_state.rs` or `count_runtime.rs`**.
- Direct WASM (`333` functions, `float=0`, `unclassified=0`) and native ARM64 (`55` `p3_state` symbols, `c3_violations=0`, `fmov_hits=0`) audits — **PASS**.
- `uor-harness` checks (`check --staged`, `check --only harness.native_space_contract`, `ratchet d23|d25|d32`, `claim-wording`, `loop report --check`, `loop pr-check --local`) — **PASS**.

## Skills used

- `native-space-cycle`

## Still open

- **Stage B3b (`P3` Wick/cloud GPU evaluation):** Pre-register and run GPU `S4` + `ft/M-World` mixed training, 2–3 epoch integer Brier error-write over `S3`/`S4`/`S6` (with `S2` error-cell arm), and `R1`/`R2`/`R4` reply panels on `DEV-tune` and `DEV-test`, followed by `Stage B4` browser Studio WASM integration.

## Plain summary

Implemented and certified the pure-integer `P3` shared runtime (`p3_state.rs`, `#![forbid(unsafe_code)]`) across `uor-r4-discrete` and `uor-r4-count-runtime`: `S4` order-free conjunction pointer tables (`UORS4P01`) with `Active` / `NoMemory` / `ShuffledRole` controls and margin hinge updates, exact `2^32`-simplex integer Brier error-write (`UORS6B01`, strictly decreasing over 3 epochs), lawful 4-gram cycle-breaking & EOS termination preserving PRLL `>= 1` masses, and binary restoring `isqrt_u64` paired-band comparison — verified with `6 + 2 + 8` tests, `0` WASM floats (`333` functions), and `0` ARM64 `fmov`s (`55` symbols) at `$0.00` GPU spend ahead of Stage B3b.
