# REPORT: native-space-state-contract, antigravity-1

Lab: antigravity. Item: [`native-space-state-contract`](../../../../roadmap/items/native-space-state-contract.md) (Issue `#3973`, D85). Order: [ORDER.md](ORDER.md). Decision: [DECISION.md](DECISION.md).

## What did not work, first

- **Initial test/registry mismatches before polish:** The first run of `crates/uor-r4-discrete/tests/native_space_state_contract.rs` surfaced three concrete gaps: (1) the operator registry entry for `discrete_action_and_compose` cited `"discrete_d11"` in `discrete_d11.rs` rather than the checked-in symbol `"fn action"` (fixed to `"fn action"`); (2) `IntegerDiscreteModel::evaluate_prefix_traced` leaves `trace_words` empty on `DiscreteReadMode::Pick` (as designed in `discrete_d11.rs:4480`), whereas `CpuDiscreteProposalScorer` returns Pick trace words — Pick replay now asserts `read_vector`, `admitted_cells`, `top8`, `prediction`, and direct `CpuDiscretePick::read_traced` block equality explicitly while `compare_trace_words` remains `true` on Route and Flock; and (3) Fault `F6` checked a slightly wrong error-message prefix (`"discrete serving loader"` vs `"serving loader refused artifact"`).
- **Formatting caught on first check:** `rustfmt --edition 2021 --check` flagged multi-line formatting diffs on initial `contract.rs` and `native_space_state_contract.rs`; both files were formatted with `rustfmt`.
- **Host-only execution scope (`$0.00` GPU spend):** This item is `needs_gpu: false` and ran exclusively on the local macOS host, referencing existing CUDA parity legs (`native_vector_cuda.rs`, `cpu_oracle_parity.rs`, `pick_read_cuda.rs`) and verifying every CUDA kernel symbol directly in `discrete.cu`, `native_vector.cu`, `ngram_count.cu`, and `kn_score.cu`.

## What changed

- [`crates/uor-r4-native-vector/src/contract.rs`](../../../../../crates/uor-r4-native-vector/src/contract.rs) & [`crates/uor-r4-native-vector/src/lib.rs`](../../../../../crates/uor-r4-native-vector/src/lib.rs): Added dependency-free versioned contract types and canonical tables:
  - `NATIVE_STATE_SCHEMA_ID` (`"uor-r4.native-state-schema/1"`, `24` carried fields across `DiscreteSpiralCore`, `WrittenStateCount`, and `NativeQ31Vector`);
  - `NATIVE_OPERATOR_REGISTRY_ID` (`"uor-r4.native-operator-registry/1"`, `11` operators across CUDA training, integer serving, CPU oracle, and WASM);
  - `NATIVE_BOUNDARY_INVENTORY_ID` (`"uor-r4.native-boundary-inventory/1"`, `15` classified paths separating native default/runtime files from legacy `uor-r4-float-train` and `fbs-export.rs`);
  - `validate_contract()` plus unit test `canonical_native_space_contract_is_complete_and_valid`.
- [`crates/uor-r4-discrete/tests/native_space_state_contract.rs`](../../../../../crates/uor-r4-discrete/tests/native_space_state_contract.rs): Added `4` integration tests verifying (1) workspace file & symbol presence for every registry & boundary entry, (2) raw-word replay across discrete Route (before & after native `Table` move `epoch 0 -> 1`), Flock (`126` kept/action slots), Pick, and `ExactCopyState`, (3) raw-word replay across Q31 `/12` (`Q31HamiltonModel` + `UR4SESS` save/restore), `/13` (`Q31LaneGates`), and `/14` (`Q31FirstPickRead` `apply` vs `step` vs `ServedQ31Read`), and (4) planted faults `F1`–`F7` failing through the shared replay comparators.
- [`docs/model/native-space-state-contract.md`](../../../../model/native-space-state-contract.md), [DECISION.md](DECISION.md), [`docs/roadmap/items/native-space-state-contract.md`](../../../../roadmap/items/native-space-state-contract.md), and [`docs/research/notebook/float-training.md`](../../../../research/notebook/float-training.md): Documented the contract, closed item `#3973` (`status: done`, `done_class: host`), and recorded one notebook row.

## Measurements (`$0.00` pod spend)

| Fixture / fault | Scope / field | Measured raw-word count or refusal |
|---|---|---|
| Schema & registry validation | `24` fields, `11` operators, `15` paths | `100%` workspace files & symbols verified; `5/5` D85 parity fields covered |
| Discrete Route replay (`39` prefixes) | `RouteChoices`, `CarryState`, `MemoryAddresses`, `TokenIds` | `1,248` trace words, `624` carry words, `78` cells, `78` `top8`, `39` preds (`0` diff pre- & post-step `epoch 0->1`) |
| Discrete Flock replay (`39` prefixes) | `ReadChoices`, `CarryState`, `MemoryAddresses`, `TokenIds` | `4,680` trace words (`126` kept slots, `126` action slots), `624` carry words (`0` diff) |
| Discrete Pick + `ExactCopyState` | `ReadChoices`, `CarryState`, `MemoryAddresses`, `TokenIds` | `624` carry words, `CpuDiscretePick` match, `ExactCopyStep = [1,0,1,2,0,0,1,0]` |
| Native Q31 `/12`, `/13`, `/14` replay | `CarryState`, `ReadChoices`, `MemoryAddresses`, `TokenIds` | `4` gate words, `128` `/12` transitions (`960` words) + `4` `UR4SESS` tokens, `8` `/13` lane gates × `2` layers × `8` steps, `256` `/14` address words + `8` picks + `256` out words (`0` diff) |
| Planted `F1_operator` | `q31_hamilton_rotate` shift `31 -> 30` | `112` `CarryState` mismatches caught |
| Planted `F2_scale` | `Q31FirstPickRead` `gain_shift 4 -> 5` | `7` `CarryState` mismatches caught |
| Planted `F3_route` | `discrete.routes[0]` port `1 -> 2` | `39` `RouteChoices` mismatches caught |
| Planted `F4a_flock` / `F4b_q31` | Flock gate `1 -> -1` / Q31 gate `+2^24` | `64` / `1` `CarryState` mismatches caught |
| Planted `F5a_q31_pick` / `F5b_flock_read` | 1-LSB key nudge / Flock root `0 -> 6` | `apply` vs `step` `[0,1,2] -> [0,1,1]` / `25` `ReadChoices` mismatches |
| Planted `F6_cell_refused` | Corrupted `discrete.cells[0]` | Fail-closed loader refusal (`cells != table`) |
| Planted `F7_token` | Token `0 <-> 1` swapped in table & cells | `15` `TokenIds` mismatches caught |

## Checks run at the head

- `rustfmt --edition 2021 --check crates/uor-r4-native-vector/src/lib.rs crates/uor-r4-native-vector/src/contract.rs crates/uor-r4-discrete/tests/native_space_state_contract.rs` — PASS.
- `cargo test --locked -p uor-r4-native-vector` — **39 passed, 0 failed**.
- `cargo test --locked -p uor-r4-discrete --test native_space_state_contract -- --nocapture` — **4 passed, 0 failed**.
- `cargo test --locked -p uor-r4-cpu-train --lib native_space_parity` — PASS (`2` passed).
- `cargo test --locked -p uor-r4-integer --features serving-wasm q31_serving` and `--features serving-wasm,fbs-synthetic --test q31_training_to_serving` — PASS.
- `cargo check --locked -p uor-r4-native-vector --target wasm32-unknown-unknown` and `cargo check --locked -p uor-r4-integer --lib --features serving-wasm --target wasm32-unknown-unknown` — PASS.
- `cargo clippy --locked --no-deps -p uor-r4-native-vector --lib -- -D warnings` — **PASS** (`0` warnings).
- `cargo clippy --locked --no-deps -p uor-r4-native-vector --all-targets` and `cargo clippy --locked --no-deps -p uor-r4-discrete --test native_space_state_contract` — **PASS (`0` warnings in touched files** `contract.rs`, `uor-r4-native-vector/src/lib.rs`, `native_space_state_contract.rs`; `--all-targets -- -D warnings` hits only pre-existing lints in `crates/uor-r4-native-vector/src/read_dump.rs:200` and `uor-r4-discrete` lib).
- `uor-harness check --only harness.native_space_contract`, `uor-harness check --staged`, `uor-harness ratchet d23|d25|d32`, `uor-harness claim-wording`, `uor-harness loop report --check --order docs/loop/tasks/native-space-state-contract/antigravity-1/ORDER.md`, `uor-harness loop pr-check --local`, `cargo run -q --locked -p xtask -- links`, `git diff --check --cached`, and `scripts/pod/uor-pod status` (`Running: 0 pod(s)`) — PASS.

## Skills used

- `native-space-cycle`
- `solution-integrity`

## Still open

- On-model language capability runs remain gated under the written-state count track (`P1`–`P5`, `object-store-consumer-m3` M3 qualification after M2 selects an accepted stack artifact).

## Plain summary

`antigravity-1` closes `native-space-state-contract` (`#3973`, D85) with zero GPU spend (`$0.00`). A dependency-free schema, source-level operator registry, and legacy/native boundary inventory now live in `uor-r4-native-vector::contract`, verified against workspace files and symbols, and exercised by a unified raw-word replay fixture in `uor-r4-discrete` (`native_space_state_contract.rs`) covering all five D85 parity fields plus nine planted-fault checks (`F1`–`F7`) across operator, scale, route, carry, read-choice, memory-address, and token-id mutations.
