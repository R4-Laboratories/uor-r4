# ORDER: written-state-count-model, antigravity-6

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (#4063, M2+/P2). Base: `de0a94ebd`.

## Goal

Implement and qualify on host & WASM the pure-integer D85 S3 geometric state-row runtime (`UORS3T01`), Starling 7-neighbour miss read, E8 geometric output Pick expert, and equal-byte fair control arms (`TokenContext`, `RandomTransport`, `SplitMixPrefix`) ahead of Stage B2b Wick GPU evaluation, while Stage B1b PR awaits GitHub `Pull Requests` outage recovery (`de0a94ebd` / `antigravity/b1b-devtest` @ `07609e82d` already `PASS` and `APPROVE`d locally):

1. Exact Conway-Sloane E8 240-root snap (`e8_snap_240`), root coordinates (`e8_root_coords_2x`), root dot product (`e8_root_dot_2x`), 56-neighbour E8 adjacency (`e8_root_neighbours_56`), and deterministic Starling `k`-neighbour ranking (`starling_k_neighbours`, `k = 7`).
2. Lossless binary `UORS3T01` codec & reader (`S3StateTable::from_bytes`, `encode_uors3t01`) with primary `(product_key, last_token)` rows, back-off `product_key` rows, `token_e8_roots[4096]`, exact Starling 7-neighbour miss recovery (for `E8Product` only), geometric output Pick scoring, and fail-closed structural/SHA-256 checks.
3. Equal-byte table builder (`build_equal_byte_s3_table`) matching `E8Product` against all three registered P2 controls (`TokenContext { k: 1..=12 }`, `RandomTransport { seed }`, `SplitMixPrefix { seed }`) at identical encoded byte caps.

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` tokens (`V=4096`), E8 240-root block codes (`0..240`), `57,600` product keys (`240 × 240`), `(product_key, last_token)` primary keys, `i8` error cells (`0` before P3), `u32` counts, `u64` totals, `token_e8_roots[4096]`, Q48 masses (`1..=2^48`), and Q16 output Pick scores. Zero floats, matmul, softmax, or `unsafe`.
2. **Shared operators:** Single `kn_ceiling/runtime/s3_state.rs` core shared across `uor_r4_discrete::s3_state_rows` and `uor_r4_count_runtime::s3_state` (`#![forbid(unsafe_code)]`, native + `wasm32-unknown-unknown`).
3. **Exact transition replay:** Cross-crate unit parity test verifying `e8_snap_240 == fbs_address::e8_cell` across all 12 `ADDRESS_SHARED_BLOCKS` plus `10,000` pseudo-random 8-blocks, Starling miss recovery, output Pick alignment, equal-byte parity across all 4 arms, and 6 planted refusal cases.
4. **Update rule:** Pure-integer deterministic accumulation & ascending `(count, key, token)` budget pruning (`build_equal_byte_s3_table`); error-write cell updates belong to Stage B3 (`P3`).
5. **Lossless serialization:** Self-describing `UORS3T01` wire format with whole-payload SHA-256 verification.

## Cancel probe (run first)

1. Confirm `/Users/casey.allard/uor-r4-b2-p2-s3` clean from `main` (`de0a94ebd`), `$0.00` pod spend.
2. Confirm `e8_snap_240` matches `uor_r4_integer::fbs_address::e8_cell` with zero discrepancies before wiring equal-byte builders or WASM exports.

## Files in scope

- `crates/uor-r4-discrete/**`
- `crates/uor-r4-integer/tests/discrete*`
- `docs/loop/tasks/written-state-count-model/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. `s3_state.rs`, `s3_state_rows.rs`, `count_runtime.rs`, and `discrete_sparse_table_parity.rs` pass `rustfmt --check`, unit/integration tests, WASM `7/7` audit (`float=0`), and native serving audit (`0` violations).
2. `REPORT.md`, `uor-harness` checks (`check --staged`, `ratchet d23|d25|d32`, `claim-wording`, `loop report --check`, `loop pr-check --local`), fresh `pro` review, and PR delivery (once GitHub PR API recovers) pass.

## Stop and ask if

- Any `e8_snap_240` disagrees with `fbs_address::e8_cell`, any audit fails, or GitHub PR API outage persists after Stage B2a finishes local review.
