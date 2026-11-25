# REPORT: written-state-count-model, antigravity-6 (`UORS3T01` Stage B2a Host + WASM S3 Runtime)

Lab: antigravity. Item: [`written-state-count-model`](../../../../roadmap/items/written-state-count-model.md). Order: [`ORDER.md`](ORDER.md). Base: `de0a94ebd`.

## What did not work, first

- **Host & WASM codec/oracle qualification only — no Wick GPU `DEV-tune`/`DEV-test` P2 evaluation yet:** This Stage B2a unit implements and qualifies the pure-integer `UORS3T01` `S3` state-row runtime, Starling 7-neighbour miss read, linear E8 output Pick expert, and equal-byte fair control builders on host CPU (`native` + `wasm32-unknown-unknown`). Sealed Wick `DEV-tune`/`DEV-test` P2 GPU evaluation is **Stage B2b** and has not been dispatched yet (`$0.00` pod spend).
- **Round 1 adversarial review (`6d60e552`) returned `REQUEST_CHANGES` (`1` CRITICAL, `4` MAJOR) before PR creation; all resolved:**
  1. *(CRITICAL)* `encoded_s3_bytes` and `from_bytes` used unchecked 32-bit `usize` arithmetic (`wasm32`) allowing multiplication wrap before `Vec::with_capacity`, and `encode_bytes` used unchecked `as u32` casts: fixed via checked `encoded_s3_bytes`, checked LE offset reads, `u32::try_from` / `checked_add`, and pre-allocation length checks.
  2. *(MAJOR #1)* Primary keys `(product_key << 16) | last_token` lacked `last_token < 4096` validation: added strict `< S3_VOCABULARY` checks across `encode_bytes`, `from_bytes`, `context_keys`, `token_e8_root`, `build_equal_byte_s3_table`, and `random_transport_blocks`.
  3. *(MAJOR #2 & #3)* `RandomTransport` permuted `e8_snap_240` (preserving the E8 Voronoi partition on shared trunk blocks), `SplitMixPrefix` leaked absolute `prefix.len()`, `TokenContext` collided mod `57,600` or duplicated backoff at `k=1`, and non-E8 controls paid a `4,096 B` `token_e8_roots` tax: replaced `root_permutation` with pure-integer path-dependent `random_transport_blocks` ($B_8$ signed permutation + `(3 * v) >> 2` decay), removed `prefix.len()` leakage in `SplitMixPrefix`, replaced header `reserved` with `flags` (`S3_FLAG_HAS_TOKEN_ROOTS = 1`) + `Option<[u8; S3_VOCABULARY]>`, and changed `output_pick_score_q16` to pure affine-linear weights (`{16384, 28672, 40960, 53248, 65536}`).
  4. *(MAJOR #4)* `uor-r4-integer` does not yet link `s3_state` outside integration tests (its `wasm`/`rlib`/`uor-chat` audits certify only P1/D11 non-regression), and ARM64 release codegen of `.sum()` (`encode_bytes`) and `[i64; 8]::saturating_add` (`lookup_q48`, `predicted_root_and_pick`) emitted NEON `fmov` lane moves (`5` then `12` hits): replaced with scalar `checked_add` loops and `accumulate_weighted_root` (`0` `fmov` / Class III violations across all `69` `s3_state` symbols in `libuor_r4_count_runtime-*.rlib`).
- **Round 2 adversarial review (`d56bcdfb`) returned `REQUEST_CHANGES` (`0` CRITICAL, `1` MAJOR, `2` MINOR); all resolved at this head:**
  1. *(MAJOR)* Commit `95612c5ac` had allowed `RandomTransport` to carry `token_e8_roots = Some(...)` and enter `StarlingMiss`, contradicting [`ORDER.md`](ORDER.md) (`for E8Product only`) and `DECISION.md` P2 control (ii) (`removes the geometry`): restricted `allows_token_roots_flag` so ONLY `E8Product` has `Some(token_roots)` (`flags == 1`) while all three controls (`TokenContext`, `RandomTransport`, `SplitMixPrefix`) enforce `None` (`flags == 0`, neutral `output_pick_q16 = S3_Q16_ONE`), restricted `StarlingMiss` strictly to `S3AddressArm::E8Product`, and added `lookup_q48` tests for `RandomTransport` (confirming `ExactPrimary`, `ExactBackoff`, `UniformFallback`, and no `StarlingMiss`).
  2. *(MINOR #1 & #2)* Changed `TokenContext { k: 2 }` short-prefix sentinel from `0` to `S3_VOCABULARY as u16` (`4096`, avoiding `&[t0]` vs `&[0, t0]` collision for all non-empty prefixes; empty `&[]` still maps `last_tok = 0`), added `encode_uors3t01` wrapper delegating to `S3StateTable::encode_bytes`, added `lookup_q48` coverage for `TokenContext { k: 2, 6 }`, `SplitMixPrefix`, and high-count `snap_i64_accum` (`u32::MAX`), and corrected the `discrete_sparse_table_parity.rs` line anchor, `--serving --call-graph` flag, and `uor-r4-count-runtime` unit test attribution below.
- **Unrelated formatting diff cleanup:** Six unrelated `uor-r4-discrete` files modified by an earlier workspace `rustfmt` pass were restored via `git checkout --` so this branch touches only `s3_state.rs`, `s3_state_rows.rs`, `lib.rs`, `count_runtime.rs`, `discrete_sparse_table_parity.rs`, and `antigravity-6/`.

## What changed

- **Shared pure-integer `UORS3T01` `S3` core (`#![forbid(unsafe_code)]`) [antigravity]:**
  - [`crates/uor-r4-discrete/src/kn_ceiling/runtime/s3_state.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/s3_state.rs): Implements compile-time precomputed `E8_ROOT_COORDS_2X` and `E8_NEIGHBOURS_56`, `e8_snap_240`, `e8_root_coords_2x`, `e8_root_dot_2x`, `e8_root_neighbours_56`, `starling_k_neighbours` (`STARLING_NEIGHBOURS = 7`), affine-linear `output_pick_score_q16`, pure-integer path-dependent `random_transport_blocks`, `S3AddressArm::context_keys` (`E8Product`, collision-free `TokenContext { k: 1, 2 }` on non-empty prefixes + `k: 3..=12`, `RandomTransport { seed }`, position-invariant `SplitMixPrefix { seed }`), `S3HitKind` (`ExactPrimary`, `ExactBackoff`, `StarlingMiss` for `E8Product` only, `UniformFallback`), `S3_FLAG_HAS_TOKEN_ROOTS`, checked `encoded_s3_bytes`, scalar `accumulate_weighted_root` + `snap_i64_accum`, `S3StateTable::{encode_bytes,from_bytes,lookup_q48}`, `encode_uors3t01`, and `build_equal_byte_s3_table` (`MAGIC_S3 = b"UORS3T01"`). Zero floats, matmul, softmax, or `unsafe`.
  - [`crates/uor-r4-discrete/src/s3_state_rows.rs`](../../../../../crates/uor-r4-discrete/src/s3_state_rows.rs), [`crates/uor-r4-discrete/src/lib.rs`](../../../../../crates/uor-r4-discrete/src/lib.rs), and [`crates/uor-r4-discrete/src/count_runtime.rs`](../../../../../crates/uor-r4-discrete/src/count_runtime.rs): Exposes `uor_r4_discrete::s3_state_rows` and `uor_r4_count_runtime::s3_state` over the same `kn_ceiling/runtime/s3_state.rs` include (`s3_state_rows` hosts 3 unit tests covering all 240 roots, 56-neighbour adjacency, linear Pick values, Starling miss, `RandomTransport` non-Starling/non-Pick checks, `TokenContext k=1`, `SplitMixPrefix` position invariance, `wasm32` overflow refusal, and `< 4096` bounds; `uor-r4-count-runtime` hosts 2 `frozen_pick` tests, while `uor_r4_count_runtime::s3_state` is exercised cross-crate in `discrete_sparse_table_parity.rs`).
- **Cross-crate integer & parity qualification [antigravity]:**
  - [`crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs`](../../../../../crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs#L1408-L1876): `uors3t01_e8_parity_starling_miss_equal_byte_arms_and_refusals` (`#[cfg(feature = "p1-count-serving")]`) checks bit-identity between `e8_snap_240` and `uor_r4_integer::fbs_address::e8_cell` across 12 `ADDRESS_SHARED_BLOCKS`, `i32::MIN`/`i32::MAX` boundaries, and `10,000` pseudo-random 8-blocks, all 5 linear Pick shells, `random_transport_blocks` divergence and `lookup_q48` without `StarlingMiss`/`token_e8_roots`, `TokenContext { k: 1, 2, 6 }` (including `&[42]` vs `&[0, 42]`) & `SplitMixPrefix` `lookup_q48`, high-count `u32::MAX` `snap_i64_accum` scaling via `encode_uors3t01`, overflow refusals, and header/body corruption refusals.

## Native-space cycle verification (D85)

1. **Single persistent state schema:** 12-bit `UORT` tokens (`S3_VOCABULARY = 4096`), 8-bit E8 240-root codes (`0..240`), `57,600` `product_key` indices (`240 × 240`), `(product_key << 16) | last_token` primary keys, `product_key` backoff keys, Q48 masses (`S3_Q48_ONE = 1 << 48`), and Q16 linear Pick scores (`16384..=65536` for `E8Product`, `S3_Q16_ONE` neutral constant for non-E8 controls).
2. **Shared typed operators:** `uor_r4_discrete::s3_state_rows` and `uor_r4_count_runtime::s3_state` share [`crates/uor-r4-discrete/src/kn_ceiling/runtime/s3_state.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/s3_state.rs) under `#![forbid(unsafe_code)]`, compiled for native ARM64 and `wasm32-unknown-unknown`.
3. **Bit-identical transition replay:** `e8_snap_240` matches `uor_r4_integer::fbs_address::e8_cell` bit-for-bit across all 12 `ADDRESS_SHARED_BLOCKS`, `i32::MIN`/`i32::MAX` boundaries, and `10,000` pseudo-random 8-blocks; all 240 roots decode to unique norm² = `8` vectors with `56` dot-4 neighbours.
4. **Pure-integer updates:** Deterministic ascending pruning inside `build_equal_byte_s3_table` enforces equal encoded byte budgets across `E8Product` and all three control arms (`TokenContext`, `RandomTransport`, `SplitMixPrefix`).
5. **Lossless serialization:** Self-describing `UORS3T01` wire codec with checked size arithmetic, arm-consistent `S3_FLAG_HAS_TOKEN_ROOTS`, and fail-closed structural + SHA-256 verification.

## Measurements

Host & WASM parity/geometry checks only (`$0.00` rented pod spend; Wick GPU `DEV-tune`/`DEV-test` evaluation is Stage B2b):

- **Conway-Sloane E8 240-root snap & geometry:** `e8_snap_240 == uor_r4_integer::fbs_address::e8_cell` on `12` `ADDRESS_SHARED_BLOCKS`, extreme `i32::MIN`/`i32::MAX` vectors, and `10,000` pseudo-random 8-blocks (`0` mismatches); `240` unique roots (`56` neighbours each, `dot_2x == 4`); linear `output_pick_score_q16` exact on `{16384, 28672, 40960, 53248, 65536}`.
- **Direct WASM audit of `s3_state` (`libuor_r4_count_runtime-8cedc23d8de5f66c.rlib`):** `704,202 B` (`sha256: cb110e5268ffd427c057296a347418869dcf72e8a8cda826174d05b36e216669`) decoded across all `7` WASM code object members (`2` data-only objects) using `audit_zero_matmul_wasm.py`: **`267` defined functions decoded, `float=0`, `unclassified=0`, `errs=0`**.
- **Direct native ARM64 audit of `s3_state` (`libuor_r4_count_runtime-08c25bda1ac08409.rlib`):** `695,120 B` (`sha256: 2d2ed9b4a4db9d70728bed4354b545daf88b485dafd737834e4b40695c50c2fc`) disassembled with `otool -tV` and classified via `audit_zero_matmul_serving.py::classify_instructions`: **`69` `s3_state` symbols checked, `c3_violations=0`, `fmov_hits=0`**.
- **Non-regression P1/D11 serving audits (note: `s3_state` is not yet wired into `uor-r4-integer` runtime / `uor-chat`):**
  - `python3 scripts/audit_zero_matmul_wasm.py --serving --call-graph` on `uor_r4_integer.wasm` (`1,348,554 B`, `sha256: 505ff30930af2b07ae82b6dbfc5cd6a9d2b6e836a642e610f5d1054f1b28a1e9`) — `7/7` symbols clean (`float=0`, `unclassified=0`).
  - `libuor_r4_integer.rlib` (`sha256: 4f0696b08ad615f0fbd05ef3c81077e669e9fc666a11471cf9960171c0c92b53`) — `60` matched symbol ranges, `5/5` TAP ok (`0` violations).
  - `uor-chat` (`sha256: 47eacb4c6234b3973b2deb6d47130b8344dc0b0c3b9a5c51acd9307043a94e1b`) — `55` matched symbol ranges, `112` transitive call-graph functions, `5/5` TAP ok (`0` violations).

## Checks run at the head

All checks below ran cleanly after the final R2 code edit:

- `rustfmt --edition 2021 --check` (on `s3_state.rs`, `s3_state_rows.rs`, `count_runtime.rs`, `discrete_sparse_table_parity.rs`) — **PASS**.
- `cargo test --locked -p uor-r4-discrete --lib s3_state_rows` — **3 passed, 0 failed** (`442` filtered out).
- `cargo test --locked -p uor-r4-count-runtime` (`frozen_pick` unit tests) — **2 passed, 0 failed**.
- `cargo test --locked -p uor-r4-integer --lib --features serving-wasm,p1-count-serving` — **736 passed, 0 failed, 1 ignored**.
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --test discrete_sparse_table_parity` — **7 passed, 0 failed, 3 ignored benchmarks**.
- Direct `libuor_r4_count_runtime` WASM (`267` functions, `float=0`, `unclassified=0`, `errs=0`) and native ARM64 (`69` `s3_state` symbols, `c3_violations=0`, `fmov_hits=0`) audits — **PASS**.
- Non-regression `uor_r4_integer.wasm` (`--serving --call-graph`, `7/7`), `libuor_r4_integer.rlib` (`5/5` TAP), and `uor-chat` (`5/5` TAP) serving audits — **PASS**.
- `uor-harness` checks (`check --staged`, `native_space_contract`, `ratchet d23|d25|d32`, `claim-wording`, `loop report --check`, `loop pr-check --local`) — **PASS**.

## Skills used

- `native-space-cycle`

## Still open

- **Stage B2b (`P2` Wick GPU evaluation):** Pre-register and evaluate `UORS3T01` (`E8Product` + Starling 7-neighbour miss + linear Pick) vs the three fair equal-byte controls on `DEV-tune` and sealed `DEV-test`.

## Plain summary

Resolved all Round 1 & Round 2 adversarial findings (`encoded_s3_bytes` 32-bit `wasm32` overflow safety, `last_token < 4096` bounds checks, restricting `StarlingMiss` and `token_e8_roots` strictly to `E8Product` with `RandomTransport` stripped of E8 geometry, `SplitMixPrefix` position invariance, collision-free `TokenContext k=1/k=2` on non-empty prefixes via sentinel `4096`, `encode_uors3t01`, precomputed 240×56 E8 tables, linear `output_pick_score_q16`, and scalar `accumulate_weighted_root` eliminating all ARM64 `fmov`s) and directly audited `libuor_r4_count_runtime` across `267` WASM functions (`float=0`) and `69` native ARM64 `s3_state` symbols (`c3_violations=0`, `fmov_hits=0`) alongside `736 + 7 + 3 + 2` passing tests ahead of Stage B2b.
