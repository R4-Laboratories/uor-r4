# ORDER: written-state-count-model, antigravity-9

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Base: `e86b0a1bf`.

## Goal

Following Stage B2-Fallback (`antigravity-8`, PR #4274) closing all four `100 MB` static `S3` fallback configurations on `DEV-tune` (`DEV-test` untouched) and advancing per [`DECISION.md`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320) to **`P3`**, implement and certify on host and WASM (`$0.00` GPU spend) the shared pure-integer **`P3` runtime**:

1. **Conjunction pointer table (`S4`, `UORS4P01`) & hinge trainer:** Order-free squarefree pair keys `(lo << 12) | hi` bijection-checked against `prime(a) * prime(b)` (`token_primes`), `S4ConjunctionTable` lossless serialization and fail-closed validation, three registered evaluation controls (`Active`, `NoMemory`, `ShuffledRole`), and pure-integer margin hinge pointer-choice error-write (`s4_hinge_step`).
2. **Integer Brier error-write engine (`S6` / `S3` / `S4`, `UORS6B01`):** Exact Q48 mixing (`brier_mix_q48`), Q32 integer Brier scoring (`integer_brier_target_q32`), exact `2^32` integer simplex projection preserving uniform floor `>= 1` (`project_simplex_q32`), multi-epoch integer Brier LMS updates (`brier_error_write_epoch`), signed `i8` error-cell writes (`apply_s3_s4_err_write`), and lossless `UORS6B01` wire codec.
3. **Lawful integer stop & repetition guard + paired band grader:** Pure-integer 4-gram cycle breaker, consecutive repeat suppressor, and span/max-length EOS promoter (`apply_lawful_stop_and_rep_guard`) keeping all masses `>= 1` for PRLL (`mle_masses_from_counts`), plus exact integer `isqrt_u64` / `paired_band_compare`.

Pre-registered GPU `P3` `DEV-tune` training, Brier error-write over S3/S4/S6, and reply-panel (`R1`/`R2`/`R4`) runs belong to Stage B3b (`antigravity-10`).

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` tokens (`V = 4096`, `BOS = 0`, `EOS = 1`), 24-bit order-free `S4` pair keys, `(role_class, span_offset, count, i8 err_cell)` pointer entries, Q32 simplex weights (`>= floor >= 1`), Q48 integer masses (`1..=2^48`), Q32 integer Brier scores, and integer stop/repeat guard traces. Zero floats, zero matmul, zero softmax, zero `unsafe`.
2. **Shared operators:** Single `crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs` core shared across `uor_r4_discrete::p3_state_rows` and `uor_r4_count_runtime::p3_state` (`#![forbid(unsafe_code)]`, native + `wasm32-unknown-unknown`).
3. **Exact transition replay & controls:** Cross-crate integration tests in `discrete_sparse_table_parity.rs` verify `s4_prime_product_u64 == token_primes(a) * token_primes(b)`, `S4ControlMode::{Active, NoMemory, ShuffledRole}` separation under `paired_band_compare`, strict 3-epoch integer Brier decrease with exact `2^32` simplex preservation, cycle-breaking EOS termination with PRLL `mle_masses_from_counts` compliance, and `wasm32` overflow + header/body corruption refusals.
4. **Update rule:** Pure-integer commutative hinge margin and Brier LMS error-write accumulation applied in canonical key/bucket order with deterministic simplex projection.
5. **Lossless serialization:** Self-describing `UORS4P01` and `UORS6B01` wire codecs with checked `wasm32`-safe size arithmetic and whole-payload SHA-256 verification.

## Cancel probe (run first)

1. Confirm worktree `/Users/casey.allard/uor-r4-b3a-p3-runtime` clean from `main` (`e86b0a1bf`), `$0.00` pod spend.
2. Confirm `s4_prime_product_u64` matches `uor_r4_integer::fbs_pair_rows::token_primes` and `uor-r4-count-runtime` WASM + ARM64 builds compile with `float=0` and `fmov=0`.

## Files in scope

- `crates/uor-r4-discrete/**`
- `crates/uor-r4-integer/tests/discrete*`
- `docs/loop/tasks/written-state-count-model/**`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. `p3_state.rs`, `p3_state_rows.rs`, `count_runtime.rs`, `lib.rs`, and `discrete_sparse_table_parity.rs` pass `rustfmt --check`, `cargo clippy -- -D warnings`, unit/integration tests, direct WASM `float=0` audit, and direct native ARM64 `fmov=0` / `c3_violations=0` audit.
2. `REPORT.md`, `uor-harness` checks (`check --staged`, `native_space_contract`, `ratchet d23|d25|d32`, `claim-wording`, `loop report --check`, `loop pr-check --local`), fresh `pro` exact-head review (`VERDICT: APPROVE`), and CI merge to `main` pass.

## Stop and ask if

- Any `uor-r4-integer` / `uor-r4-discrete` build or non-regression serving audit breaks, or ARM64 scalar codegen emits `fmov` / Class-III instructions that cannot be eliminated.
