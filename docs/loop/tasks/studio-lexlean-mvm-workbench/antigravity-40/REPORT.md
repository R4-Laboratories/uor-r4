# REPORT: studio-lexlean-mvm-workbench, antigravity-40

Lab: `antigravity`. Order: [`ORDER.md`](ORDER.md). Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7). Base: `650e86c3b` (`ORDER.md` committed alone first at `1fdb1a1a1`). GPU spend: `$0.00`.

## What did not work

- **Intentional CHARTER §3 rejection / quarantine of `afflom/vTensor` (`b0ce3f6b72ce3b12ea3081977b61cbb01b36980f`):** `afflom/vTensor` (`MIT OR Apache-2.0`) wraps GGML buffer/graph ops (`VT-01..VT-46`) over `verdant-tensor = "=2.17.0"` (`afflom/verdant-engine` rev `4654da8724f142a248b32739f1d930066e281bd5`) using `F64`/`F32`/`F16`/`BF16` floats, Berkeley SoftFloat reductions, float `RMS_NORM`/L2 ops, and `4x4` `F64` matrices. Importing it would violate CHARTER §3.1/§3.2/§3.3/§3.7, so `vTensor` and `verdant-tensor` remain strictly excluded; runtime acceleration remains `'wasm-scalar-integer'` until Stage B4 builds `+simd128` pure-integer WASM modules and updates [`scripts/audit_zero_matmul_wasm.py`](../../../../../scripts/audit_zero_matmul_wasm.py) (`0xFD` prefix instructions are still marked `unclassified` in this commit).
- **Initial Stage A5 Cargo quarantine regex matched a comment (`task-17316`):** Scanning raw [`Cargo.toml`](../../../../../Cargo.toml) matched the historical `#` comment at line 83 about D29's removal of `uor-r4-model-source` and the excluded `uor-matmul` family (`crates/uor-r4-model-source`, non-member under D29). Updating [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs) to strip `#` comments and inspect `[workspace] members`, `Cargo.lock`, and `web/studio/foundry/studio/core/Cargo.toml` resolved the false positive (`71/71` pass).
- **R1 fresh-context `pro` adversarial review (`67928eb1-7a7d-4ee1-b762-f576a31009ed` at `f57bac71b`) returned `APPROVE` (`0` CRITICAL, `0` MAJOR, `3` MINOR):** All `3` polish items were resolved and re-verified across build, tests, and dual-origin Chrome CDP:
  1. `MINOR-1`: Corrected `audit_zero_matmul_wasm.py` line citations (`236-247`, `270-271`, `751-757`), removed duplicate `f64x2.*`, and added `i32x4.trunc_sat_f64x2_*_zero` in [`docs/integration/vtensor-quarantine-and-wasm-simd.md`](../../../../../docs/integration/vtensor-quarantine-and-wasm-simd.md).
  2. `MINOR-2`: Guarded primitive `opts` (`opts && typeof opts === 'object'`), gated `validate()` behind `if (wasmSupported)` (requiring `instantiate` as well as `validate`), passed `WASM_INT_SIMD128_PROBE.slice()` in [`uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs), and added regression cases in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs).
  3. `MINOR-3`: Clarified in [`docs/integration/vtensor-quarantine-and-wasm-simd.md`](../../../../../docs/integration/vtensor-quarantine-and-wasm-simd.md) that the pre-existing locked `uor_foundry_v13_wasm_bg.wasm` contains an unused pure-integer helper `lexlean_r4_wasm::vtensor_semiring_3d` unrelated to `afflom/vTensor`.

## What changed

- **Quarantine & pure-integer WASM SIMD128 roadmap:**
  - [`docs/integration/vtensor-quarantine-and-wasm-simd.md`](../../../../../docs/integration/vtensor-quarantine-and-wasm-simd.md): Documents the verified `afflom/vTensor` audit (`codex/vtensor-implementation` `b0ce3f6b72ce3b12ea3081977b61cbb01b36980f`), the binding CHARTER §3 quarantine boundary, and the Stage B4 pure-integer WASM SIMD128 acceleration plan (integer-only `0xFD` opcode allowlist + `f32x4.*`/`f64x2.*` rejection in `audit_zero_matmul_wasm.py`, `uor-foundry-v13-wasm` & `vram_store.rs` `core::arch::wasm32` integer kernels, bit-identical scalar/SIMD oracle per §3.7/D85, and runtime dispatch over `vram-store.mjs`).
- **Studio runtime WASM SIMD128 capability probe:**
  - [`web/studio/foundry/studio/shell/uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs): Adds `WASM_INT_SIMD128_PROBE` (64-byte pure-integer `v128.const 0` x2 + `i32x4.dot_i16x8_s` `0xfd 0xba 0x01` module, zero floats), exports cached `detectWasmCapabilities(opts)` (testing `WebAssembly.validate` + `instantiate`, `BigInt64Array`, `SharedArrayBuffer`, `Atomics.wait`, `crossOriginIsolated`, reporting `activeIntegerKernel: 'wasm-scalar-integer'`, `plannedSimdKernel: 'wasm-simd128-integer-b4'`, `vtensorQuarantined: true`), and surfaces `wasmCapabilities` in `getStatus()`.
  - [`web/studio/foundry/studio/shell/host.js`](../../../../../web/studio/foundry/studio/shell/host.js): Exposes `window.__uorR4.getWasmCapabilities` (`uorR4.detectWasmCapabilities`) without altering IPC handlers (`368/179/310/306`) or `FOUNDRY_CHAT_SYSTEM_PROMPT`.
- **Standards index, regenerated Studio outputs & tests:**
  - [`scripts/studio/build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs): Appends `vtensor-quarantine-and-integer-simd` (`38` total sections, `59,009` B `< 110 KB`).
  - [`web/studio/foundry/studio/tools/studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs): Relaxes A4 exact `37` count to `>= 37` and adds 2 Stage A5 tests checking `detectWasmCapabilities()` (including partial `validate`-only mock and primitive `opts`), `getStatus().wasmCapabilities`, `buildStandardsIndex` (`38` sections `< 110 KB`), `host.js` exposure, `docs/integration/vtensor-quarantine-and-wasm-simd.md`, and `[workspace] members` + `Cargo.lock` + `core/Cargo.toml` quarantine.

## Measurements

- **`standards-index.json`:** `59,009` B (`38` sections, SHA-256 `af55310a7b3a97c9de1a2e8ce5e32c357a155c1f0bfeddedce9251083ec8c5cb`).
- **`host.bundle.js`:** `1,106,137` B (SHA-256 `8e35f52e6e978837287cb3eb971b14134564ab0ff5ad8fafdb86ee05f80fad4f`).
- **`core.wasm`:** `1,033,712` B (SHA-256 `c3bbdbd959d3ea798cfafdd5b21cb523f63c2ebb93ae11c45bd696370978a761`).
- **`manifest.json`:** `62,692` B (SHA-256 `514c5e8fa3f9460bda3adfc0802136f18bb09499c90095653ecf09a7c2ec208c`, `485` closure files).
- **`uor-r4-runtime.mjs`:** `30,503` B (SHA-256 `1390df97f1edba15cb59232c6cb1b18794b6452b90bbf85a0d8f79889b1d14fa`).
- **WASM zero-matmul & zero-float audit (`uor_r4_integer_bg.wasm`, `1,289,272` B, SHA-256 `1f018be5955c034d04ecbcccc28388bca7f310b6add9a84293b448f40f3236ac`):** `ok 1..7`, `float=0`, `unclassified=0`.
- **Headless Chrome `window.__uorR4.getWasmCapabilities()` & `getStatus().wasmCapabilities`:** Both returned `{"wasmSupported":true,"simd128IntegerSupported":true,"bigInt64ArraySupported":true,"sharedArrayBufferSupported":false,"atomicsSupported":true,"crossOriginIsolated":false,"activeIntegerKernel":"wasm-scalar-integer","plannedSimdKernel":"wasm-simd128-integer-b4","vtensorQuarantined":true}`.
- **Foundry Studio CDP E2E (`verify_foundry_studio.mjs web/studio/dist`):**
  - Root `/`: Home Chat Turn 1 (`34` chars) & Turn 2 (`126` chars), Build App `app=2` tool round-trip, Linux `6.18.7`, GitHub/Stitch (`githubPushSha: 7dd5deb7`), Custom AI API Endpoint, `Qwen3-1.7B` mount `35,930 ms` (`backend=opfs`), `TTFT=2,078 ms`, `7.79 tok/s`, `total=3,086 ms`.
  - Subpath `/uor-r4/studio/`: Home Chat Turn 1 (`34` chars) & Turn 2 (`126` chars), Build App `app=2` tool round-trip, Linux `6.18.7`, GitHub/Stitch, Custom AI API Endpoint, `Qwen3-1.7B` mount `35,552 ms` (`backend=opfs`), `TTFT=2,044 ms`, `7.92 tok/s`, `total=3,033 ms`.
- **GPU spend:** `$0.00` (`Running: 0 pod(s), $0/h of $8.00/h`).

## Checks run at the head

- `bash web/studio/build_foundry_studio.sh`: **PASS**.
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)`: **PASS**.
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` & `shell_parse.mjs`: **PASS**.
- `node --test web/studio/relay/relay.test.mjs web/studio/foundry/studio/tools/{prism_r4_route,command_ownership_registry,canonical_generation_ownership,preview_permissions,custom_api_endpoint,generation_patch,studio_phase1_3,studio_phase4a,stage3_os_github_stitch}.test.mjs`: **`71/71` PASS, `0` failed, `0` skipped**.
- `node web/studio/verify_studio_worker.mjs`: **PASS** (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`, `saveRestoreBitIdentical: true`).
- `bash scripts/studio/runtime-files.sh --verify web/studio/dist --typed`: **PASS** (`485` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap`: **PASS** (`ok 1..7`, `float=0`, `unclassified=0`).
- `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/foundry/studio/tools/command_ownership_registry_browser.mjs`: **PASS** (`commands: 368`, `handlers: 179`, `inputs: 310`, `outputs: 306`, `pageExceptions: 0`).
- `CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/verify_foundry_studio.mjs web/studio/dist`: **PASS** (`ALL FOUNDRY STUDIO CDP CHECKS PASSED` on `/` and `/uor-r4/studio/`).
- Headless Chrome `window.__uorR4.getWasmCapabilities()` & `window.__uorR4.getStatus().wasmCapabilities` CDP check: **PASS** (`simd128IntegerSupported: true`, `activeIntegerKernel: "wasm-scalar-integer"`, `vtensorQuarantined: true`).

## Skills used

- None outside standard harness and Studio verification scripts.

## Still open

- Post-polish `uor-harness` checks, fresh-context `pro` R2 exact-head adversarial review (`APPROVE`), `[antigravity] uor-review v1`, CI + `loop pr-check`, squash-merge, and live `--typed` `gh-pages` Studio deploy (`$0.00` GPU spend).

## Plain summary

`antigravity-40` completes Stage A5 (and all 5 Phase A integration stages) and resolves all 3 R1 `MINOR` polish findings at `$0.00` GPU spend: audits and formally quarantines `afflom/vTensor` (`b0ce3f6b`, floating-point GGML backend over `verdant-tensor`) outside UOR-R4 under CHARTER §3, publishes the Stage B4 pure-integer WASM SIMD128 roadmap (`docs/integration/vtensor-quarantine-and-wasm-simd.md`), adds a 64-byte zero-float `i32x4.dot_i16x8_s` runtime capability probe (`detectWasmCapabilities()`) with guarded `opts` and `wasmSupported` gating to `uor-r4-runtime.mjs` / `getStatus()` / `window.__uorR4.getWasmCapabilities`, appends section #38 to `standards-index.json` (`59,009` B), and passes all build, `71/71` Node tests, worker, `float=0` WASM audit, command-registry (`368/179/310/306`), and dual-origin Chrome CDP checks.
