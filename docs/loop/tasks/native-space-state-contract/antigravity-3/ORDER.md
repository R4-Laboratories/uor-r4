# ORDER: native-space-state-contract, antigravity-3

Lab: antigravity. Item: [`native-space-state-contract`](../../../../roadmap/items/native-space-state-contract.md) (Issue `#3973`, D85).

## Finished in `antigravity-1` and `antigravity-2`

- `antigravity-1`: Committed `ORDER.md` alone (`15a60eee2`), delivered schema/operator/replay/faults/boundary and opened PR `#4250` (`d805f2740`).
- `antigravity-2`: Committed `ORDER.md` alone (`ffb436df7`), fixed `count.s5_pick_block_code` anchor (`code_pick.rs::code_pick`), `OPERATOR_REGISTRY_V1` bindings (`pick_row`, `evaluate_prefix_internal`), all 24 `source_anchor` disk checks, `read_q31_read_uorlut01` address comparison, `F5a_q31_pick` comparator routing, relative links (`68/68`), and Pick/CUDA caveats (`b03bec112`).

## Task (repair pass on R2 review of `b03bec112`)

1. Fix `/12` `carry_words_compared` count in `docs/model/native-space-state-contract.md`: `960` (`8 * 16 * 4 + 7 * 16 * 4`), not `1,016`.
2. Fix discrete `top8` word count in `docs/model/native-space-state-contract.md` and `antigravity-1/REPORT.md`: `78` (`39 * 2` admitted tokens), not `312` (`39 * 8`), and refine `antigravity-1/REPORT.md` `/13` phrasing to match `8 lane gates × 2 layers × 8 steps`.
3. Assert exact `top8_words_compared == 78` and `summary_12.carry_words_compared == 960` (plus print them) in `crates/uor-r4-discrete/tests/native_space_state_contract.rs`, and keep `fault_key_read` `weights` identical (`1 << 25`) to `base_read` in `F5a_q31_pick` so only `key_gates` varies.
4. Write `antigravity-3/REPORT.md`, verify all checks cleanly at HEAD, push, and complete fresh-context adversarial review R3.

## Five required native-space items (D85 / `.agents/skills/native-space-cycle/SKILL.md`)

1. **Schema (`uor-r4.native-state-schema/1`):** 24 verified integer/fixed-point carried fields; zero float fields.
2. **Operators (`uor-r4.native-operator-registry/1`):** 11 verified bindings across CUDA, integer serving, CPU oracle, and WASM.
3. **Replay & faults:** Raw-word replay across all 5 D85 parity fields plus `F1`–`F7` with exact asserted counts (`960`, `78`).
4. **In-state updates:** Direct integer `apply`/`step` transitions with zero float master.
5. **Serialization:** Lossless `UORLUT01` and `UR4SESS` byte codecs.

## Scope & untouched paths

Writes only inside `docs/loop/tasks/native-space-state-contract/**`, `crates/uor-r4-discrete/tests/native_space_state_contract.rs`, and `docs/model/native-space-state-contract.md`. Zero edits to `crates/uor-r4-integer/`, `Cargo.lock`, `docs/harness/native-space.json`, pinned docs, or peer PR files.
