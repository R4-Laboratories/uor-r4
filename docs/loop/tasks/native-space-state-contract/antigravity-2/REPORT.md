# REPORT: native-space-state-contract, antigravity-2

Lab: antigravity. Item: [`native-space-state-contract`](../../../../roadmap/items/native-space-state-contract.md) (Issue `#3973`, D85). Order: [ORDER.md](ORDER.md). Prior pass: [`antigravity-1/REPORT.md`](../antigravity-1/REPORT.md). Decision: [`antigravity-1/DECISION.md`](../antigravity-1/DECISION.md).

## What did not work, first

- **Review `CHANGES` on `d805f2740` (`PR #4250`, subagent `2b318186`) surfaced 4 blocking + 2 non-blocking defects:**
  1. `crates/uor-r4-native-vector/src/contract.rs` pointed `count.s5_pick_block_code`'s `source_anchor` at nonexistent `crates/uor-r4-discrete/count-runtime/src/lib.rs::code_pick` (`count-runtime/Cargo.toml` maps `[lib]` to `../src/count_runtime.rs`, including `src/kn_ceiling/runtime/code_pick.rs`), and `contract_registry_files_and_symbols_exist_in_workspace` did not validate `STATE_FIELDS_V1`'s `source_anchor`s.
  2. `compare_q31_read_replay` computed `q31_read_address` words on `train_read` without decoding and comparing against the served `q31.read` gates, and `compare_q31_hamilton_replay` incremented `emitted_tokens_compared` while discarding `step.token` (`Q31HamiltonModel` has no token head; `UR4SESS` save/restore separately checks 4 continuation tokens).
  3. `F5a_q31_pick` called `Q31FirstPickRead::apply` vs `step` directly instead of routing through `compare_q31_read_replay`.
  4. Relative markdown links in `antigravity-1/ORDER.md` and `antigravity-1/REPORT.md` were off by one `../`.
  5. `OPERATOR_REGISTRY_V1` cited `observe_identity_pick` (`Relation`-only) and `ExactCopyState` instead of `pick_row` and `evaluate_prefix_internal`.
  6. `docs/model/native-space-state-contract.md` omitted the Pick `trace_words` caveat and CUDA host non-execution note.
- **Host-only execution scope (`$0.00` GPU spend):** This item is `needs_gpu: false` and ran on macOS CPU; `.cu` kernels are covered by existing `#[cfg(feature = "cuda")]` parity tests rather than executed on this host.

## What changed

- [`crates/uor-r4-native-vector/src/contract.rs`](../../../../../crates/uor-r4-native-vector/src/contract.rs): Fixed `count.s5_pick_block_code` `source_anchor` to `crates/uor-r4-discrete/src/kn_ceiling/runtime/code_pick.rs::code_pick` and updated `discrete_pick_read` (`pick_row`) and `discrete_writer_admission_and_emission` (`evaluate_prefix_internal`).
- [`crates/uor-r4-discrete/tests/native_space_state_contract.rs`](../../../../../crates/uor-r4-discrete/tests/native_space_state_contract.rs):
  - Verified all `24` `STATE_FIELDS_V1` `source_anchor` files and leaf symbols exist on disk in `contract_registry_files_and_symbols_exist_in_workspace`.
  - Removed the vacuous `emitted_tokens_compared` counter from `compare_q31_hamilton_replay`.
  - Updated `compare_q31_read_replay` to decode served `q31.read` via `read_q31_read_uorlut01` and compare `query` + `key` folded `q31_read_address` words against `train_read` (`ParityField::MemoryAddresses`).
  - Routed `F5a_q31_pick` through `compare_q31_read_replay` against an exported `/14` fault artifact with perturbed `key_gates`, asserting both `ReadChoices` and `MemoryAddresses` mismatches alongside the `1`-LSB tie-sensitivity check (`[0, 1, 2] -> [0, 1, 1]`).
- [`docs/model/native-space-state-contract.md`](../../../../model/native-space-state-contract.md), [`antigravity-1/ORDER.md`](../antigravity-1/ORDER.md), [`antigravity-1/REPORT.md`](../antigravity-1/REPORT.md), and [ORDER.md](ORDER.md): Fixed all relative links and added explicit Pick `trace_words` (`discrete_d11.rs:4480`), `/12` `UR4SESS` 4-token vs transport-only, `/14` `q31.read` address comparison, and CUDA non-execution caveats plus updated `F5a` counts.

## Measurements (`$0.00` pod spend)

| Fixture / fault | Scope / field | Measured raw-word count or refusal |
|---|---|---|
| Schema, registry & boundary | `24` fields, `11` operators, `15` paths | `100%` workspace files & symbols verified (`source_anchor` + operators + boundary) |
| Discrete Route / Flock / Pick + ExactCopy | All `5` D85 parity fields | Route `1,248` trace + `624` carry (`epoch 0->1`); Flock `4,680` trace (`126` kept/action slots); Pick `624` carry (`compare_trace_words=false`) + `[1,0,1,2,0,0,1,0]` |
| Native Q31 `/12`, `/13`, `/14` replay | `CarryState`, `ReadChoices`, `MemoryAddresses`, `TokenIds` (`UR4SESS`) | `/12`: `4` gate + `128` transitions + `4` `UR4SESS` continuation tokens; `/13`: `8` lane gates; `/14`: `256` `q31.read` address words + `8` picks + `256` out words (`0` diff) |
| Planted faults `F1`–`F4b` | Operator, gain shift, route, Flock/Q31 carry | `F1=112`, `F2=7`, `F3=39`, `F4a=64`, `F4b=1` |
| Planted `F5a_q31_pick` | Rotated `/14` `key_gates` via `compare_q31_read_replay` (+ tie check) | `3` `ReadChoices` pick + `8` `MemoryAddresses` key-address mismatches (tie `[0,1,2] -> [0,1,1]`) |
| Planted `F5b_flock_read` / `F6` / `F7` | Flock root, `discrete.cells[0]`, swapped tokens | `F5b=25` `ReadChoices`, `F6=true` (`cells != table`), `F7=15` `TokenIds` |

## Checks run at the head

- Relative markdown link check across all touched/new `.md` files — **PASS** (`0` broken links).
- `rustfmt --edition 2021 --check crates/uor-r4-native-vector/src/lib.rs crates/uor-r4-native-vector/src/contract.rs crates/uor-r4-discrete/tests/native_space_state_contract.rs` — **PASS**.
- `cargo test --locked -p uor-r4-native-vector` — **39 passed, 0 failed**.
- `cargo test --locked -p uor-r4-discrete --test native_space_state_contract -- --nocapture` — **4 passed, 0 failed**.
- `cargo check --locked -p uor-r4-native-vector --target wasm32-unknown-unknown` — **PASS**.
- `cargo clippy --locked --no-deps -p uor-r4-native-vector --lib -- -D warnings` — **PASS** (`0` warnings).
- `uor-harness check --only harness.native_space_contract`, `uor-harness check --staged`, `uor-harness ratchet d23|d25|d32`, `uor-harness claim-wording`, `uor-harness loop report --check --order docs/loop/tasks/native-space-state-contract/antigravity-2/ORDER.md`, `uor-harness loop pr-check --local`, `cargo run -q --locked -p xtask -- links`, `git diff --check --cached`, and `scripts/pod/uor-pod status` (`Running: 0 pod(s)`) — **PASS**.

## Skills used

- `native-space-cycle`
- `solution-integrity`

## Still open

- On-model language capability runs remain gated under the written-state count track (`P1`–`P5`, `object-store-consumer-m3` M3 qualification after M2 selects an accepted stack artifact).

## Plain summary

`antigravity-2` resolves all six findings from the exact-head review of `d805f2740` (`PR #4250`): `count.s5_pick_block_code`'s `source_anchor` now points to `src/kn_ceiling/runtime/code_pick.rs::code_pick` and every `STATE_FIELDS_V1` field is checked against workspace files and symbols; `compare_q31_read_replay` decodes served `q31.read` gates via `read_q31_read_uorlut01` to compare all `256` `q31_read_address` words against training; `F5a_q31_pick` now fails through `compare_q31_read_replay` (`3` pick + `8` address mismatches); operator bindings use `pick_row` and `evaluate_prefix_internal`; and all relative links and caveats are updated and verified.
