# REPORT: native-space-state-contract, antigravity-3

Lab: antigravity. Item: [`native-space-state-contract`](../../../../roadmap/items/native-space-state-contract.md) (Issue `#3973`, D85). Order: [ORDER.md](ORDER.md). Prior passes: [`antigravity-1/REPORT.md`](../antigravity-1/REPORT.md), [`antigravity-2/REPORT.md`](../antigravity-2/REPORT.md). Decision: [`antigravity-1/DECISION.md`](../antigravity-1/DECISION.md).

## What did not work, first

- **Review R2 `CHANGES` on `b03bec112` (`PR #4250`, subagent `3e2564db`) caught two factual word-count errors plus one non-blocking isolation note:**
  1. `docs/model/native-space-state-contract.md` listed `1,016` transport + continuity `i64` words on `/12`; `compare_q31_hamilton_replay` actually accumulates `8 * 16 * 4 + 7 * 16 * 4 = 960` words (`prev` is `None` on step `0`, skipping `64` words), and `native_space_state_contract.rs` only asserted `>= 512`.
  2. `docs/model/native-space-state-contract.md` and `antigravity-1/REPORT.md` listed `312` `top8` `u32` IDs across `39` prefixes; `kept_cells = 2` admits `2` tokens per prefix (`39 * 2 = 78` `top8` IDs), not `312`.
  3. Non-blocking: `F5a_q31_pick` `fault_key_read` set `weights = 1 << 24` instead of matching `base_read`'s `1 << 25`, and `antigravity-1/REPORT.md` wrote `8 lane gates × 16 carries` instead of `8 lane gates × 2 layers × 8 steps`.
- **Host-only execution scope (`$0.00` GPU spend):** `needs_gpu: false`; ran on macOS CPU (`.cu` kernels are verified by existing `#[cfg(feature = "cuda")]` parity tests).

## What changed

- [`crates/uor-r4-discrete/tests/native_space_state_contract.rs`](../../../../../crates/uor-r4-discrete/tests/native_space_state_contract.rs): Asserted `top8_words_compared == 78` (`prediction_words_compared == 39`) on Route & Flock, asserted & printed `summary_12.carry_words_compared == 960` on `/12`, and aligned `fault_key_read` `weights` to `vec![1_i64 << 25; 32]` so `F5a_q31_pick` isolates the `key_gates` mutation cleanly.
- [`docs/model/native-space-state-contract.md`](../../../../model/native-space-state-contract.md) & [`antigravity-1/REPORT.md`](../antigravity-1/REPORT.md): Corrected `1,016 -> 960` on `/12`, `312 -> 78` `top8` across Route/Flock/Pick, and `8 lane gates × 2 layers × 8 steps` on `/13`.

## Measurements (`$0.00` pod spend)

| Fixture / fault | Scope / field | Asserted / printed result |
|---|---|---|
| Schema, registry & boundary | `24` fields, `11` operators, `15` paths | `100%` verified against workspace files & symbols |
| Discrete Route / Flock / Pick | All `5` D85 parity fields (`39` prefixes) | Route `1,248` trace, `624` carry, `78` cells, `78` `top8`, `39` preds (`epoch 0->1`); Flock `4,680` trace (`126` kept/action slots); Pick `624` carry (`compare_trace_words=false`) |
| Native Q31 `/12`, `/13`, `/14` | `CarryState`, `ReadChoices`, `MemoryAddresses`, `TokenIds` | `q31_12_carry_words=960` (`8` steps) + `4` `UR4SESS` tokens; `8` `/13` lane gates × `2` layers × `8` steps; `/14` `256` `q31.read` addresses + `8` picks + `256` out words |
| Planted faults `F1`–`F7` | All `7` fault classes (`weights = 1<<25` in `F5a`) | `F1=112`, `F2=7`, `F3=39`, `F4a=64`, `F4b=1`, `F5a_q31_pick_mismatches=3` + `F5a_q31_addr_mismatches=8` (tie `[0,1,2]->[0,1,1]`), `F5b=25`, `F6=true`, `F7=15` |

## Checks run at the head

- Relative markdown link check across touched/new `.md` files — **PASS** (`0` broken links).
- `rustfmt --edition 2021 --check crates/uor-r4-native-vector/src/lib.rs crates/uor-r4-native-vector/src/contract.rs crates/uor-r4-discrete/tests/native_space_state_contract.rs` — **PASS**.
- `cargo test --locked -p uor-r4-native-vector` — **39 passed, 0 failed**.
- `cargo test --locked -p uor-r4-discrete --test native_space_state_contract -- --nocapture` — **4 passed, 0 failed**.
- `cargo check --locked -p uor-r4-native-vector --target wasm32-unknown-unknown` and `cargo clippy --locked --no-deps -p uor-r4-native-vector --lib -- -D warnings` — **PASS**.
- `uor-harness check --only harness.native_space_contract`, `uor-harness check --staged`, `uor-harness ratchet d23|d25|d32`, `uor-harness claim-wording`, `uor-harness loop report --check --order docs/loop/tasks/native-space-state-contract/antigravity-3/ORDER.md`, `uor-harness loop pr-check --local`, `cargo run -q --locked -p xtask -- links`, `git diff --check --cached`, and `scripts/pod/uor-pod status` (`Running: 0 pod(s)`) — **PASS**.

## Skills used

- `native-space-cycle`
- `solution-integrity`

## Still open

- On-model language capability runs remain gated under the written-state count track (`P1`–`P5`, `object-store-consumer-m3` M3 qualification after M2 selects an accepted stack artifact).

## Plain summary

`antigravity-3` fixes the two documentation word-count discrepancies flagged in R2 review of `b03bec112` (`960` `/12` carry words and `78` discrete `top8` IDs across `39` prefixes), asserts both exact numbers directly in `native_space_state_contract.rs`, and isolates `F5a_q31_pick`'s `key_gates` mutation with identical `1 << 25` weights.
