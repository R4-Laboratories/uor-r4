# ORDER: written-state-count-model, antigravity-16

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P4`). Base: `37ad6b3b4`.

## Goal

Following Stage B5b (`antigravity-15`, PR `#4298`) qualifying frozen `P3` (`p4_full_errwrite`: `1,205,738` micro-BPB, `86.94%` of `P1(A1s)`, `0.8502 × R*`, controlled `errwrite - cache_static` `[−1,104.7M, −987.7M]` Q12) and cold Stage B5a `S4` (`cd0e4385`) on sealed `DEV-test` (`DECISION.md §4`), implement **Phase `P4` PR 1 of 2** (`$0.00` GPU spend):

1. **Shared pure-integer `P4` runtime (`crates/uor-r4-discrete/src/kn_ceiling/runtime/p4_serving.rs`, `count_runtime.rs`, `rank.rs`):**
   - Lossless `UORS3C01` compact codec for Golden-Shell `S3` (`UORS3T02` + `i8` errors, packing `100.0 MB + 12.3 MB` losslessly to `~54.4 MB`) with zero-copy `CompactS3View` and single-pass `fill_masses_q48` oracle-matched against single-target `lookup_modulated_q48` (primary hit, backoff hit, Starling 7-neighbour miss read, uniform floor).
   - Lossless `UORS2C01` compact codec for disjoint-`C` `S2-skip` (`UORS2S01` + `i8` errors, packing `80.1 MB + 11.4 MB` losslessly to `~38.9 MB`) with zero-copy `CompactS2SkipView` and single-pass `fill_masses_q48`.
   - Incremental `CausalSessionCache` (`O(1)` per-token update, `BOS = 0` reset, `0` B disk) bit-identical to `cache_eval.rs`, plus `P4State` / `P4Session` over the exact 22-expert Stage B5b `UORS6B01` mixture (`[KN, ML_0..10, Uniform, Pick, S3, S2-skip_0..2, Cache_0..3]`), `2,640` buckets, `2,048`-token causal context, optional `S4` pointer table (`UORS4P01`) + lawful integer stop/repetition guard, and full `P4StepTrace` raw-word trace.
2. **Deployment container & `uor-chat` integration (`crates/uor-r4-integer/src/discrete_count.rs`, `src/bin/uor-chat.rs`):**
   - Add `SCHEMA_P4 = "uor-r4.lut-discrete-p4/1"` (`UORP4A01` inner container) alongside existing `uor-r4.lut-discrete-count/1`, streaming the embedded D11 trunk via `load_client_artifact` without duplicating `file_bytes`, and add `write_p4_deployment`.
   - Auto-dispatch `P4` inside `uor-chat` at `context_limit = 2,048` with incremental `P4Session`, `--p1-trace` raw-word `P4StepTrace`, and `UOR_P1_TIMING` measurement.
3. **Synthetic round-trip, fault & compliance verification (`crates/uor-r4-integer/tests/discrete_p4_serving.rs`):**
   - Verify `H4_SHELL` equivalence, `UORS3C01` / `UORS2C01` / `CausalSessionCache` / `P4Session` bit-identity vs `brier_mix_q48`, `2,048`-token context handling, corruption refusals, `uor-chat` trace execution, `cargo clippy -- -D warnings`, and native/WASM zero-float/zero-matmul audits.

Full-dose `<= 4.00 GB` packing, M1 `2,048`-context warm/cold `tok/s` & RSS, `>= 64`-prompt CUDA-vs-M1 raw-word parity, and Wick GPU Bar A (`chat-v0-p2`) PRLL scoring follow in Phase `P4` PR 2 (`antigravity-17`).

## Native-space cycle contract (D85)

1. **State schema:** Original 12-bit `UORT` tokens (`V = 4,096`, `BOS = 0`, `EOS = 1`), `UORKNC03`, `UORS3C01`, `UORS2C01`, `UORS4P01`, `UORS6B01` (`22` experts × `2,640` buckets Q32 simplex), `CausalSessionCache` (`0` B disk), `UORP4A01` / `uor-r4.lut-discrete-p4/1` container. Zero floats, matmul, softmax, or runtime `unsafe`.
2. **Shared operators:** `uor_r4_count_runtime::{CompactCountLookup, FrozenPickMixture, code_pick, p3_state, s3_state, p4_serving}` shared between host runtime/packers and `uor-chat`.
3. **Bit-identical replay oracles:** `discrete_p4_serving.rs` verifies `0` ULP/bit differences between sparse-accelerated `P4Session::step` and exhaustive `4,095`-token `brier_mix_q48` / `lookup_modulated_q48`, plus end-to-end `uor-chat` CLI trace consistency.
4. **Update & serialization:** Serving holds frozen integer weights/`i8` error cells (`write_p4_deployment` validates all SHA-256 digests and decodes through `P4::load`).

## Files in scope

- `crates/uor-r4-discrete/src/count_runtime.rs`
- `crates/uor-r4-discrete/src/kn_ceiling/runtime/{p4_serving.rs,rank.rs}`
- `crates/uor-r4-integer/src/discrete_count.rs`
- `crates/uor-r4-integer/src/bin/uor-chat.rs`
- `crates/uor-r4-integer/tests/discrete_p4_serving.rs`
- `docs/loop/tasks/written-state-count-model/antigravity-16/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. All unit/integration tests, `cargo fmt --check`, `cargo clippy -- -D warnings`, and native/WASM zero-float/zero-matmul audits pass on host.
2. `REPORT.md`, `m4-serving.md`, and `written-state-count-model.md` are updated, all `uor-harness` checks pass (`verdict: PASS`), exact-head `pro` review approves, CI passes, and PR merges to `main`.
