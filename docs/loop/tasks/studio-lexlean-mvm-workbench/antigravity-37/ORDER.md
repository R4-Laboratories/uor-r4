# ORDER: studio-lexlean-mvm-workbench, antigravity-37

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (Stage A2: `afflom/verdant-studio` Lean-Certified Planning Backend Integration)

Execute Stage A2 of the approved 5-repo integration plan at `$0.00` GPU spend:

1. **Import Lean-Certified `verdant_studio` Crate & Build Virtual-FS Driver (`core.wasm`):**
   - Import MIT/Apache-2.0 `certified/packages/rust-std/verdant_studio/src/lib.rs` (SHA-256 `1642b5877447becf3d28ab4f5af119e7100f9690b85dffc6895fda80a76ce55f`) and `provenance.json` from `afflom/verdant-studio` into `web/studio/foundry/studio/core/src/verdant_studio.rs` & `verdant_provenance.json` with zero changes to `Cargo.toml` / `Cargo.lock`.
   - Implement `#![forbid(unsafe_code)]` virtual-FS effect driver [`verdant_driver.rs`](../../../../../web/studio/foundry/studio/core/src/verdant_driver.rs) over `BTreeMap<String, Vec<u8>>` + `BTreeSet<String>` (`dirs`), refusing `..` traversal and OS process spawn, supporting `auto_init` minimal registers and `inline_input`, and wire `"verdant-studio"` in [`abi.rs`](../../../../../web/studio/foundry/studio/core/src/abi.rs) + 8 MB WASM stack in [`build_foundry_studio.sh`](../../../../../web/studio/build_foundry_studio.sh).
   - Add Rust integration tests in [`core/tests/verdant_studio.rs`](../../../../../web/studio/foundry/studio/core/tests/verdant_studio.rs).
2. **Studio Agent Planning Tool (`verdant_plan`) & Curated Standards Index:**
   - Expose `verdant_plan` in [`r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs) & [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js) (plus `window.__uorR4.verdantPlan`, keeping IPC `command-registry` `368/179/310/306` and casual `FOUNDRY_CHAT_SYSTEM_PROMPT` unchanged).
   - Update [`build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs) with `web_search`/`verdant_plan` and 3 curated Verdant Studio sections (`verdant-studio-certified-api`, `verdant-studio-admission-and-strata`, `verdant-studio-game-and-web-standards-catalog`) within `< 110 KB`.
   - Add live Node WASM instantiation (`core.wasm` `holo_run`) and agent routing tests in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs).

## Cancel probe

- Any edit touches `CHARTER.md`, `DECISIONS.md`, `docs/harness/native-space.json`, `Cargo.lock`, `crates/uor-r4-*`, `src-tauri/**`, or `web/studio/foundry/studio/generated/**`.
- `core.wasm` size exceeds `2.5 MB`, WASM stack overflows, or any D23/D25/D32 ratchet (`4451`/`93`/`31`), `audit_zero_matmul_wasm.py` (`float=0`), `studio_phase1_3` provenance digest, or Chrome CDP verification regresses.

## Scope

Writes inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-37/**`, `web/studio/foundry/studio/core/src/**`, `web/studio/foundry/studio/core/tests/**`, `web/studio/foundry/studio/shell/**`, `web/studio/foundry/studio/tools/**`, `web/studio/build_foundry_studio.sh`, and `scripts/studio/build-standards-index.mjs`.

## Done check

`cargo test --locked --manifest-path web/studio/foundry/studio/core/Cargo.toml`, `bash web/studio/build_foundry_studio.sh`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `shell_parse.mjs`, all 10 relay/Foundry Node test suites, `command_ownership_registry_browser.mjs`, `verify_studio_worker.mjs`, `runtime-files.sh --verify web/studio/dist --typed`, `audit_zero_matmul_wasm.py`, Chrome CDP `verify_foundry_studio.mjs`, `uor-harness` checks/ratchets/`loop pr-check`, fresh `pro` exact-head review, merge, and live `gh-pages` Studio deploy PASS at `$0.00` GPU spend.
