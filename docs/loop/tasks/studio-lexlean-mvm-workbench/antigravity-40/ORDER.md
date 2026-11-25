# ORDER: studio-lexlean-mvm-workbench, antigravity-40

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (Stage A5: `afflom/vTensor` CHARTER §3 Quarantine, Integer WASM SIMD128 Roadmap & Capability Probe)

Execute Stage A5 (final stage of the approved 5-repo Phase A) at `$0.00` GPU spend under strict CHARTER §3 quarantine:

1. **Quarantine & Pure-Integer WASM SIMD128 Specification (`docs/integration/vtensor-quarantine-and-wasm-simd.md`):**
   - Document the verified `afflom/vTensor` audit (`codex/vtensor-implementation` at `b0ce3f6b72ce3b12ea3081977b61cbb01b36980f`, dual-licensed `MIT OR Apache-2.0`): GGML device backend (`VT-01..VT-46`) built over `verdant-tensor = "=2.17.0"` (`afflom/verdant-engine` rev `4654da8724f142a248b32739f1d930066e281bd5`), using F64/F32/F16/BF16 floats, Berkeley SoftFloat reductions, float RMS_NORM/L2/unary ops, and 4x4 F64 matrices.
   - Bind the CHARTER §3.1/§3.2/§3.3/§3.7 quarantine boundary: zero `vTensor`, `verdant-tensor`, or `uor-matmul` imports/calls in `Cargo.toml`, `crates/uor-r4-*`, `core.wasm`, `uor-foundry-v13-wasm`, or Studio runtime bundles.
   - Specify the Stage B4 pure-integer browser WASM SIMD128 acceleration plan: whitelist integer-only `0xFD` SIMD128 opcodes (`i8x16`/`i16x8`/`i32x4`/`i64x2`/`v128` memory/bitwise/arithmetic/`i32x4.dot_i16x8_s`) in `audit_zero_matmul_wasm.py` while keeping `f32x4.*`/`f64x2.*` rejected as `float > 0`, vectorize integer kernels under `core::arch::wasm32` (`+simd128`) with bit-identical parity against scalar Rust integer kernels, and dispatch on runtime SIMD support paired with Stage A3 `vram-store.mjs`.
2. **Runtime WASM Integer-SIMD128 Capability Probe (`uor-r4-runtime.mjs` & `host.js`):**
   - Add pure-integer `WASM_INT_SIMD128_PROBE` (`v128.const 0` + `i32x4.dot_i16x8_s` `0xfd 0xba 0x01`, zero floats) and `detectWasmCapabilities(opts)` to [`uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs), exposing `wasmCapabilities` in `getStatus()` and `window.__uorR4.getWasmCapabilities` in [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js), leaving Home Chat and IPC `command-registry` (`368/179/310/306`) unchanged.
3. **Standards Index & Unit Tests:**
   - Add `vtensor-quarantine-and-integer-simd` section (`38` total, `< 110 KB`) in [`build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs) and Stage A5 unit tests (including Cargo quarantine scan) in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs).

## Cancel probe

- Any edit touches `CHARTER.md`, `DECISIONS.md`, `docs/harness/native-space.json`, `Cargo.toml`, `Cargo.lock`, `crates/uor-r4-*`, `web/studio/foundry/studio/core/**`, `src-tauri/**`, `scripts/audit_zero_matmul_wasm.py`, or `web/studio/foundry/studio/generated/**`.
- Any forbidden crate reference enters `Cargo.toml` or `crates/uor-r4-*`, or any `audit_zero_matmul_wasm.py`, `command_ownership_registry_browser.mjs` (`368/179/310/306`), or CDP check regresses.

## Scope

Writes inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-40/**`, `docs/integration/vtensor-quarantine-and-wasm-simd.md`, `web/studio/foundry/studio/shell/**`, `web/studio/foundry/studio/tools/**`, `scripts/studio/build-standards-index.mjs`, and regenerated `web/studio/**` build outputs.

## Done check

`bash web/studio/build_foundry_studio.sh`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `shell_parse.mjs`, all 10 Node test suites, `verify_studio_worker.mjs`, `runtime-files.sh --verify --typed`, `audit_zero_matmul_wasm.py`, `command_ownership_registry_browser.mjs`, dual-origin Chrome CDP `verify_foundry_studio.mjs` + `window.__uorR4.getWasmCapabilities()`, `uor-harness` checks/ratchets/`loop pr-check`, fresh `pro` exact-head review, merge, and live `gh-pages` Studio deploy PASS at `$0.00` GPU spend.
