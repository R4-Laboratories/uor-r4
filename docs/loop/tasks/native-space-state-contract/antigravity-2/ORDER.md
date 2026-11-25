# ORDER: native-space-state-contract, antigravity-2

Lab: antigravity. Item: [`native-space-state-contract`](../../../../roadmap/items/native-space-state-contract.md) (Issue `#3973`, D85).

## Finished in `antigravity-1`

- Added `docs/loop/tasks/native-space-state-contract/antigravity-1/ORDER.md` alone in its folder (`15a60eee2`).
- Implemented `NativeStateSchemaV1`, `NativeOperatorRegistryV1`, and `NativeBoundaryInventoryV1` in `crates/uor-r4-native-vector/src/{contract.rs,lib.rs}`, added raw-word replay and planted-fault tests in `crates/uor-r4-discrete/tests/native_space_state_contract.rs`, documented the contract and closed item `#3973`, and opened PR `#4250` (`d805f2740`), where fresh-context adversarial review returned `CHANGES`.

## Task (repair pass on review of `d805f2740`)

1. Fix `count.s5_pick_block_code` `source_anchor` (`crates/uor-r4-discrete/src/kn_ceiling/runtime/code_pick.rs::code_pick`) and refine `discrete_pick_read` (`pick_row`) and `discrete_writer_admission_and_emission` (`evaluate_prefix_internal`) in `crates/uor-r4-native-vector/src/contract.rs`.
2. Verify every `STATE_FIELDS_V1` `source_anchor` against workspace files/symbols in `contract_registry_files_and_symbols_exist_in_workspace`, remove the vacuous `emitted_tokens_compared` increment in `compare_q31_hamilton_replay`, compare served `q31.read` `query_gates`/`key_gates` addresses via `read_q31_read_uorlut01` in `compare_q31_read_replay`, and route `F5a_q31_pick` through `compare_q31_read_replay` against an exported `/14` fault artifact.
3. Fix broken relative markdown links (`../../../../` / `../../../../../`) in `antigravity-1/ORDER.md` and `antigravity-1/REPORT.md`, add `REPORT.md` in `antigravity-2/`, update `docs/model/native-space-state-contract.md` (with explicit Pick `trace_words` and CUDA non-execution caveats) and `DECISION.md` / `float-training.md` if counts change.

## Five required native-space items (D85 / `.agents/skills/native-space-cycle/SKILL.md`)

1. **Schema (`uor-r4.native-state-schema/1`):** 24 integer/fixed-point carried fields across `DiscreteSpiralCore`, `WrittenStateCount`, and `NativeQ31Vector`; zero float fields.
2. **Operators (`uor-r4.native-operator-registry/1`):** 11 operators mapped across CUDA, integer serving, CPU oracle, and WASM (`pick_row`, `evaluate_prefix_internal`, etc.).
3. **Replay & faults:** Raw-word replay across all 5 D85 parity fields plus `F1`–`F7` through the shared comparators (`read_q31_read_uorlut01` + `ServedQ31Read`).
4. **In-state updates:** Direct integer `apply`/`step` transitions with zero float master.
5. **Serialization:** Lossless `UORLUT01` (`lut-discrete/1`, `lut-discrete-trunk/1`, `lut-discrete-count/1`, `lut-stack/12..14`) and `UR4SESS` byte codecs.

## Scope & untouched paths

Writes only inside `docs/loop/tasks/native-space-state-contract/**`, `crates/uor-r4-native-vector/src/contract.rs`, `crates/uor-r4-discrete/tests/native_space_state_contract.rs`, `docs/model/native-space-state-contract.md`, and `docs/research/notebook/float-training.md`. Zero edits to `crates/uor-r4-integer/`, `Cargo.lock`, `docs/harness/native-space.json`, pinned docs, or peer PR files.
