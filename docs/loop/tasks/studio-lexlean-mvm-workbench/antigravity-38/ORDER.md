# ORDER: studio-lexlean-mvm-workbench, antigravity-38

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (Stage A3: `afflom/vRAM` Pure-Integer 4 KiB Geometric Page Store Integration)

Execute Stage A3 of the approved 5-repo integration plan at `$0.00` GPU spend under CHARTER §3 quarantine (`vTensor` kept completely out of `crates/uor-r4-*`, zero floats, zero softmax/matmul):

1. **Pure-Integer 4 KiB Lossless Geometric Page Store (`core.wasm`):**
   - Implement `#![forbid(unsafe_code)]` [`vram_store.rs`](../../../../../web/studio/foundry/studio/core/src/vram_store.rs) (attributing `afflom/vRAM` `MIT OR Apache-2.0` `device/src/*`) inside `web/studio/foundry/studio/core/src/` with zero edits to `Cargo.toml` / `Cargo.lock`.
   - Support all 7 lossless 4 KiB representations (`Fill`, `Repeated` folded & shifted, `Sparse`, `Selector`, `Corrected`, `Packed`, `Shared`/`Residual` content-addressed literal pool with `Weak<Literal>` reclamation) verified byte-for-byte before admission, sparse two-level `Directory` (`GROUP=64`, `FANOUT=256`, `LEAF=16,384` pages), cross-page `write_range`/`read_range`, atomic rollback on `max_backing_bytes` overflow, `discard_pages`, pure-integer `VramStats` (`savings_permille`), binary `vram_*` WASM exports, and `holo_run` `op: "vram"` in [`abi.rs`](../../../../../web/studio/foundry/studio/core/src/abi.rs) + [`lib.rs`](../../../../../web/studio/foundry/studio/core/src/lib.rs).
   - Add comprehensive Rust integration tests in [`core/tests/vram_store.rs`](../../../../../web/studio/foundry/studio/core/tests/vram_store.rs).
2. **Browser vRAM Runtime, Worker Memory-Pack Backing & Curated Index:**
   - Add [`vram-store.mjs`](../../../../../web/studio/foundry/studio/shell/vram-store.mjs) (`createVramStore`, blob allocator with page reclamation, registered in [`shell_parse.mjs`](../../../../../web/studio/foundry/studio/tools/shell_parse.mjs)), expose `vramExports` in [`inference.js`](../../../../../web/studio/foundry/studio/shell/inference.js), expose `window.__uorR4.vram` & `foundry_status` vRAM telemetry in [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js) (keeping IPC `command-registry` `368/179/310/306` unchanged), and upgrade `'memory'` fallback pack caching in [`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs) without slowing OPFS/IndexedDB hot paths.
   - Add `vram-geometric-page-store` section to [`build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs) (`< 110 KB`) and Stage A3 live WASM + shell tests in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs).

## Cancel probe

- Any edit touches `CHARTER.md`, `DECISIONS.md`, `docs/harness/native-space.json`, `Cargo.toml`, `Cargo.lock`, `crates/uor-r4-*`, `src-tauri/**`, or `web/studio/foundry/studio/generated/**`.
- `core.wasm` size exceeds `2.5 MB`, WASM scratch buffers leak on errors, or any D23/D25/D32 ratchet (`4451`/`93`/`31`), `audit_zero_matmul_wasm.py` (`float=0`), or Chrome CDP verification regresses.

## Scope

Writes inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-38/**`, `web/studio/foundry/studio/core/src/**`, `web/studio/foundry/studio/core/tests/**`, `web/studio/foundry/studio/shell/**`, `web/studio/foundry/studio/tools/**`, and `scripts/studio/build-standards-index.mjs`.

## Done check

`cargo test --locked --manifest-path web/studio/foundry/studio/core/Cargo.toml --lib --test verdant_studio --test vram_store`, `bash web/studio/build_foundry_studio.sh`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `shell_parse.mjs`, all 10 relay/Foundry Node test suites, `command_ownership_registry_browser.mjs`, `verify_studio_worker.mjs`, `runtime-files.sh --verify web/studio/dist --typed`, `audit_zero_matmul_wasm.py`, Chrome CDP `verify_foundry_studio.mjs`, `uor-harness` checks/ratchets/`loop pr-check`, fresh `pro` exact-head review, merge, and live `gh-pages` Studio deploy PASS at `$0.00` GPU spend.
