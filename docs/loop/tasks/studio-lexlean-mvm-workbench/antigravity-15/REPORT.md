# REPORT for Foundry Studio Phase 4A: V13 Speculative Decode, Prewarm, MessageChannel Yield, Auto-RAG & Cold-SW Hardening (`antigravity-15`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `ca165e41e` (PR #4207, `ORDER.md` committed at `128fe92b3`).

## What did not work

- **Speculative decode speedup vs single-threaded CPU WASM (`defaultSpeculative = false`):** While `HoloTok13SpeculativeDrafter` + `step_speculative_tile` achieved exact `0-LSB` byte-for-byte parity in `uor-foundry-v13-wasm`, apples-to-apples benchmarking in Node (`prefill_slice_tokens: 32`, `slice_tokens: 4`, KV flushed to the 285-token prewarmed system prefix before each turn) showed `speculative: true` ran slower on single-threaded CPU WASM than exact decode (`5034 ms` vs `3724 ms` / `8.34` vs `11.28 tok/s` on a 42-token chat prompt with `15` proposed / `0` accepted; `3012 ms` vs `1957 ms` / `6.31` vs `9.71 tok/s` on a 19-token tool prompt with `25` proposed / `2` accepted; `10909 ms` vs `9296 ms` / `11.28` vs `13.23 tok/s` on a 123-token app-edit prompt with `39` proposed / `4` accepted) because verifying unaccepted draft tokens across 28 dense integer layers in scalar/SIMD128 WASM adds per-layer work without GPU tensor parallelism. Kept `speculative` defaulted to `false` on CPU WASM while exposing opt-in `speculative: true` and `configure({ speculative })` with `0-LSB` parity verified for Phase 5 WebGPU.
- **Offline manifest caching gap caught during plan review:** Prior `uor-v13-worker.mjs` cached `pack_00`..`pack_09` in OPFS/IDB but fetched `manifest.json` and `manifest.stitch.json` over HTTP, which would trigger remote fetches when `prewarmIfCached` mounts on `Page.reload` if cache checks looked only at `pack_09.pack`. Added OPFS/IDB caching for `manifest.json` and `manifest.stitch.json` with SHA-256 verification and added `hasOfflineV13Cache` checking all three before `prewarmIfCached` mounts.
- **PR #4213 fresh-context exact-head review (`684764cc-2724-4379-b1cf-1ed49628b4c9`) findings, remediated in v3:**
  1. `configure({ mlpMode })` (`web/studio/foundry/studio/shell/uor-r4-runtime.mjs`) sent `cfg.mlp_mode`, whereas `ConfigureRequest` in `uor-foundry-v13-wasm/src/lib.rs` deserialized `mlp_policy`. Fixed `uor-r4-runtime.mjs` to send `cfg.mlp_policy = mlpMode` and added `#[serde(alias = "mlp_mode")]` on `ConfigureRequest::mlp_policy`, with `cfg.mlp_policy = mlpMode` asserted in `studio_phase4a.test.mjs` and `mlp_verdict: "DenseExact"` asserted in `smoke_v13_wasm.mjs` and `verify_foundry_studio.mjs`.
  2. In `v13_step_turn`'s speculative loop (`publish-wasm/src/lib.rs:827`), `model.kv_cache.tokens.len() >= model.max_seq` fired prematurely at `idx = 0` after `step_speculative_tile` appended multi-token accepted batches ending at `max_seq`. Fixed to `kv_before + idx + 1 >= model.max_seq`.
  3. Remediated all 5 non-blocking observations: set `FoundryV13Engine::default().speculative_default = false`; skipped redundant `cacheWrite` on `manifest.json` cache hit and inspected `res.message || res.detail || res.error` in `uor-v13-worker.mjs`; aligned `BARE_TOOL_PROMPT_RE` in `host.js` and `foundry-tools-and-holo` in `scripts/studio/build-standards-index.mjs` to the exact 18 `FOUNDRY_TOOL_NAMES`; and corrected `uor-r4-v13-packs` OPFS directory name in `build-standards-index.mjs`.
- **Vendored upstream workflow tests & Python skip:** Upstream workflow tests under `web/studio/foundry/studio/tools/` remain unrunnable here due to missing vendored `.github/workflows/studio.yml`; all 11 UOR-R4 Foundry unit suites passed (`63/63`). `python3 -m unittest discover -s scripts/tests` passed `35` tests with `1` pre-existing skipped test.

## What changed

- **External `uor-foundry-v13-wasm` v3 (`Casey-allard/lexlean-r4@wasm-v13-foundry`, commit `0b5b6d30a019286269d432a625eb7d9c7a339b1b`):**
  - Exposed `0-LSB` speculative decoding (`HoloTok13SpeculativeDrafter::propose_adaptive_draft_tokens` + `HoloV13Model::step_speculative_tile` with exact stop/max_new/max_seq per-index KV rewind), `v13_prewarm_prefix`, and `v13_configure` (`mlp_policy` + `mlp_mode` alias, `speculative_default: false`), compiled with `RUSTFLAGS="-C target-feature=+simd128"`.
  - Pinned v3 `uor_foundry_v13_wasm.js` and `uor_foundry_v13_wasm_bg.wasm` hashes/sizes in [`web/studio/foundry-assets.lock.json`](../../../../../web/studio/foundry-assets.lock.json).
- **Worker & Runtime ([`web/studio/foundry/studio/shell/uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs), [`web/studio/foundry/studio/shell/uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs)):**
  - Replaced `setTimeout(r, 0)` loop yields with a race-free `MessageChannel` `yieldMacrotask` helper, raised `mapLimit(manifest.packs, 4, ...)`, updated turn slicing defaults (`maxNewTokens = 768`, `prefillSliceTokens = 32`, `sliceTokens = 4`), cached `manifest.json` (on miss) and `manifest.stitch.json` offline in OPFS/IDB, added `prewarm_v13` / `configure_v13` worker actions (`res.message || res.detail || res.error`), exported cache-gated `prewarmIfCached` (`hasOfflineV13Cache`) and `configure` (`defaultSpeculative = false`, `cfg.mlp_policy`), and forwarded `prefixReusedTokens`, `draftProposed`, `draftAccepted`, and `mlpMode` telemetry.
- **Studio Self-Knowledge, Auto-RAG Grounding & Cold-SW Hardening ([`scripts/studio/build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs), [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs), [`web/studio/foundry/studio/shell/host.js`](../../../../../web/studio/foundry/studio/shell/host.js)):**
  - Added `foundry-studio-architecture`, `foundry-tools-and-holo` (exact 18 `FOUNDRY_TOOL_NAMES`), and `foundry-browser-apis-and-vm` (`uor-r4-v13-packs`) sections to [`scripts/studio/build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs).
  - Updated [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs) (`maxNewTokens = 768`, optional `groundingContext` appended after the user prompt on round 0 and preserved across multi-round tool calls so `FOUNDRY_SYSTEM_PROMPT` stays byte-identical for KV prefix reuse, plus telemetry forwarding).
  - Added `buildTurnGroundingContext` (skipping slash/bare 18-tool commands, including Build/Agent file lists and `readSafeAppEnvVars` keys only without secret values, plus bounded `[Standards]` RAG snippets, never written to persisted `messages`), wired `groundingContext` and `768` token limit in `chatTurn` and `home-chat:send`, added idle `uorR4.prewarmIfCached`, exposed helpers on `window.__uorR4`, and added `await workerActive` in `buildPreview` and `compileTailwind`.
- **Tests & E2E ([`web/studio/foundry/studio/tools/studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs), [`web/studio/verify_foundry_studio.mjs`](../../../../../web/studio/verify_foundry_studio.mjs)):**
  - Added unit tests for the new standards sections, prompt/KV-prefix/grounding invariants, and worker/runtime/host contracts, plus CDP checks for `configure({ mlpMode: 'dense' })` -> `DenseExact`, `prewarmIfCached` + live `0-LSB` parity pair, `standards:lookup`, keys-only `buildTurnGroundingContext`, and zero-network reload prewarm.

## Measurements

- External V13 WASM v3 (`0b5b6d30a019286269d432a625eb7d9c7a339b1b`):
  - `uor_foundry_v13_wasm.js`: `73,505` bytes, sha256 `e4ea92b79bea247f51366d77b0c630a8defef744f944dee028d3bebb8b9ad63f`.
  - `uor_foundry_v13_wasm_bg.wasm`: `11,431,737` bytes, sha256 `855213514627bf885a38b0e1799d68a3d1f3b971366358700f67b2621e710ded`, `50` inference-marker functions audited (`compute_float_ops: 0`, `float_ops: 0`, `passed: true`), `11,226` SIMD instructions (`wasm-tools print` matching `^\s*(v128|i8x16|i16x8|i32x4|i64x2)\.`, `11,377` total SIMD type+instruction lines with `151` `(local ... v128)` lines).
- Generated `standards-index.json` (on disk after final `unittest`): `49,511` bytes (`< 110 KB`), `sha256: 65a73901894fe153d46687e9f31dd542c23466d47d39422318d087f3fefde2d0`, `31` sections.
- Built Studio bundle (`web/studio/dist/`): `485` files staged (`482` closure files in `manifest.json`), closure `4de00d803a10fdc9138a909a6160b34d61e0a23b9bce3cba7025e24490490cfc` from `build_foundry_studio.sh` (updated on disk to `5a13cb3f4675210e15d77ee453b73ef899c3974847c090c2753cd71ca0806b3a` after `test_tauri_runtime.py` re-ran `prepare-tauri-runtime.sh`, refreshing `standards-index.json` `generated_at`).
- `host.bundle.js`: `102` inputs, `1,032,365` bytes, sha256 `f49428dad45fb10af81bee5fbd7e42541940c5eab741188688976daf0f8b965b`.
- `stage-studio-site.sh --typed`: `485` file(s), `75,759,616` byte(s), `482` closure files verified.
- Serving WASM (`web/studio/pkg/uor_r4_integer_bg.wasm`): `1,274,400` bytes, sha256 `6c6dd3e5a67794bbd0b4c0dd7f208139a271ad57005af396f866f0e84bd0dd07`, `7/7` D80/D81 entrypoints PASS.
- Ratchet baselines (`uor-harness ratchet`): `d23` `4451` across `165` files, `d25` `93` across `26` files (`0` tensor-core intrinsics), `d32` `31` across `1` files.

## Checks run at the head

- `eval_foundry_v13.py` & `smoke_v13_wasm.mjs` (`task-7994`, external `uor-foundry-v13-wasm` v3) — PASS (`compute_float_ops: 0`, `MOUNT OK`, `STITCH RECOVERY OK`, `PREWARM & CONFIGURE OK` for both `mlp_policy` & `mlp_mode`, `0-LSB PARITY OK` on chat & tool turns, `draftAccepted: 2`).
- `node web/studio/foundry/studio/tools/shell_parse.mjs` — PASS (`every shell script parses under the goal the browser uses for it`).
- `node --test` across all 11 UOR-R4 Foundry unit suites (10 existing + `studio_phase4a.test.mjs`) — `63/63` PASS, `0` failed.
- `bash web/studio/build_foundry_studio.sh` — PASS (`Built Foundry Studio in .../web/studio/dist (485 files)`).
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)` — PASS (`adds no type error to upstream's own (28), every ui call is rewritten for Radix (37992 ms)`).
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` — PASS (`102 inputs, 1032365 bytes`, also re-verified after `unittest`).
- `node web/studio/verify_studio_worker.mjs` — PASS (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`).
- `node web/studio/verify_foundry_studio.mjs web/studio/dist` — PASS on `/` and `/uor-r4/studio/` (`ALL FOUNDRY STUDIO CDP CHECKS PASSED`, `configure({ mlpMode: 'dense' }) -> DenseExact`, `prewarm 304 tok`, `0-LSB parity OK`, `standards:lookup`, keys-only `buildTurnGroundingContext`, zero-network reload, `.holo` publish/inspect/import/export, `liveAfterClearOk`, `holoCacheStillPresent`, zero forbidden IPC errors).
- `bash scripts/stage-studio-site.sh --typed` & `bash scripts/studio/runtime-files.sh --verify` — PASS (`485` files, `482` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap` — PASS (`ok 1..7`).
- `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` — `Ran 35 tests in 113.242s`, `OK (skipped=1)`.
- `uor-harness ratchet d23`, `d25`, `d32` — PASS (`4451`, `93`, `31` unchanged).

## Skills used

- Standard git, `wasm-pack`, `wasm-tools`, Chromium CDP verification harness, and `uor-harness` loop workflow.

## Still open

- Staged checks (`uor-harness check --staged`, `uor-harness loop pr-check --local`), updated push to PR #4213, final fresh-context `pro` exact-head approval, CI & `pr-check --pr 4213`, merge into `main`, worktree/branch cleanup, `--typed` deploy to `gh-pages`, live Chrome CDP check on `https://uor-foundation.github.io/uor-r4/studio/`, and planning Phase 4D (`0.6B v2` SFT/GPTQ) + Phase 5 (`Qwen3-1.7B` WebGPU WGSL + 4-worker WASM SIMD `+atomics`).

## Plain summary

Remediated PR #4213's review findings (`configure` `mlp_policy` wiring + `mlp_mode` alias, speculative `max_seq` per-index KV truncation, default `speculative_default: false`, conditional `manifest.json` cache writes, and exact 18-tool / `uor-r4-v13-packs` alignment) on top of Foundry Studio Phase 4A, republished `uor-foundry-v13-wasm` v3 (`0b5b6d30`), and re-verified all unit, build, typecheck, worker, Python, WASM, ratchet, and dual-mount Chromium CDP tests cleanly.
