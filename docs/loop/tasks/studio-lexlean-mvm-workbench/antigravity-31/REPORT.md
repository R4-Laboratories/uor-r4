# REPORT for Foundry Studio V13 Dual-Model Recovery (`antigravity-31`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `3d1565c25` (PR `#4237`, `ORDER.md` committed alone first at `504610246`). GPU spend: `$0.00`.

## What did not work

- **Retirement of Run 5 `0.6B` Foundry LoRA (`weights-v13-qwen3-0.6b-foundry`, `3b04cd46`):** Previous Studio (`gh-pages` @ `PR #4213`) served the Run 5 Foundry LoRA which collapsed casual chat onto Foundry capability boilerplate and failed to close/update build apps reliably. Replacing it with the already-published uncorrupted pure-integer `UORR4V13` models (`uor-r4:qwen3-0.6b` @ `081cb4f9` default + `uor-r4:qwen3-1.7b` @ `b35a6dcc`, `0` float/matmul ops) restored coherent general chat and code generation without GPU retraining.
- **Node WASM Build Turn 1–2 truncation & `<tool_call>` hijacking (`task-12517`, `task-12527`, `task-12825`, `task-12830`):**
  - On base `0.6B` and `1.7B`, Build Turn 1 originally hit `stop=length` at `512` tokens mid-JSX, wrapped `<prism-write>` code in triple backticks, and `0.6B` emitted `const App = () =>` rather than `export default function App()`.
  - After putting `<tool_call>` and all 18 tool signatures at the end of `FOUNDRY_SYSTEM_PROMPT`, Chrome CDP `task-12820` / Node reproduction `task-12825` failed on `"build a todo list app"` because `0.6B` emitted a spurious `<tool_call>{"name": "Todo List App", "arguments": {"version": "1.0"}}</tool_call>` (`toks=22`, `updatedFiles=false`). Splitting pure app-authoring (`FOUNDRY_BUILD_SYSTEM_PROMPT`) from explicit tool/`local-agent` turns (`FOUNDRY_SYSTEM_PROMPT` via `BUILD_TOOL_INTENT_RE`) resolved this (`toks=338, stop=eos`).
  - In Turn 2 (`task-12830`, `toks=473, stop=eos`), wrapping `Current src/App.tsx` inside triple backticks in `[Foundry Context]` primed `0.6B` to close `<prism-write>` with ```` ``` ```` instead of `</prism-write>` and omit `import { useState } from "react";`. Switching `[Foundry Context]` to `<current-file path="src/App.tsx">`, adding unclosed/`\`\`\`\``-ended component fallback recovery in `extractFallbackCodeWrites`, and adding React hook / `React` / `export default` fallbacks via `transformProjectSource` + `stamp_shell.py` resolved compilation and runtime hook errors.
- **Initial Chrome CDP Check 5c test script fixes (`task-12859`, `task-12872`, `task-12891`):** Wait selectors in `verify_foundry_studio.mjs` initially clicked `"Don't allow"` on `/Allow|Approve/i`, polled `chat_stream` `'idle'` before `get-chat` recorded an assistant reply, and dispatched a duplicate `app_run` `START` concurrently with `PreviewPanel` auto-start (`previewPermissionChanged`). Aligning Check 5c to exact `'Allow this preview'` click, assistant message readiness, and `previewReloadEpoch` wait resolved all three.
- **PR #4241 fresh-context exact-head review R1 (`5e8c6920-3c4d-46d2-95a5-0fb179d4314d`, commit `3dc0ff366`) blocking findings, all remediated:**
  1. `uor-v13-worker.mjs` cleared WASM mount state inside `v13_begin_mount` without resetting JS `mountedContainerSha` / `mountedModelId`, `uor-r4-runtime.mjs` left `loadedModelId` / `loadedCapabilities` intact before/on failed swaps and ran `ensureModelLoaded` outside `generate`'s `serializeOp`, and `reusedMount: true` omitted `phase: 'ready'` progress. Fixed all three (clearing JS mount/loaded state before `v13_begin_mount` / `loadV13Entry`, serializing `ensureModelLoadedQueued` inside `generate`'s `serializeOp`, emitting `'ready'` + reporting `'memory'` if any pack hits memory fallback).
  2. `r4-agent-loop.mjs` checked `BUILD_TOOL_INTENT_RE.test(round0Prompt)` instead of `prompt`, allowing `[Foundry Context]` standards snippets to flip UI builds to the 18-tool prompt. Fixed to test `prompt` only.
  3. `createProjectResolverPlugin` matched single-line `import` regexes (duplicating multi-line React imports and destructured hooks) and `buildPortable` omitted the fallback transforms. Extracted shared `transformProjectSource` parsing multi-line `import` / destructured declarations without comments and wired it into both `createProjectResolverPlugin` and `buildPortable`. Also tightened `FOUNDRY_DOMAIN_RE`, split `filterStopWords` away from explicit `standards_lookup`, fixed `getUorR4Models` `contextWindow` (`m.context_window || 2048`), rejected traversal paths in `extractFallbackCodeWrites`, updated `ModelPicker.tsx` `"hydrating"` pill scoping, and updated `build-standards-index.mjs` tool count heading (`18`).
  4. Corrected `REPORT.md` (`stage3_os_github_stitch.test.mjs`, exact `34/34` pass count across all 7 Foundry suites, and all checks re-run directly on the final code head).
- **Legacy upstream Foundry test fixtures & Python skipped test:** Outside UOR-R4's 7 Foundry unit suites (`34/34` pass), legacy upstream Foundry test files on `main` fail on missing vendored `.github/workflows/studio.yml`, `foundry/verification/`, `studio/model/host-read-scope.json`, or `try/vendor/prism-code/src/blake3.mjs`. `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` ran `35` tests (`1` pre-existing skipped).

## What changed

- **Dual-Model Catalog & Standards Index ([`scripts/studio/catalog.mjs`](../../../../../scripts/studio/catalog.mjs), [`web/studio/foundry/studio/shell/app.json`](../../../../../web/studio/foundry/studio/shell/app.json), [`scripts/studio/build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs), [`scripts/tests/test_typed_runtime.py`](../../../../../scripts/tests/test_typed_runtime.py)):**
  - Configured primary `qwen3-0.6b` (`081cb4f9cdf0e4e9dd1a160fa014615cb81db10e`, `9` packs `pack_00`..`pack_08`, `410` pieces, `429,484,190` bytes, `container_sha256: 617ea77fd9c3d15285e351bd4f094cc0845cf5bc038e042a3fb5aadf0e406bda`) and secondary `qwen3-1.7b` (`b35a6dcce1c527d91402a07dfa9ebd0928ee28f7`, `23` packs `pack_00`..`pack_22`, `1,088` pieces, `1,140,250,782` bytes, `container_sha256: 2c362d73046851ff69dfa7e1de197d4f0fee14031e9e53e29bf278c41b6bc8bf`).
- **Routing, Runtime & Pack-Streaming OPFS/IDB Mount ([`web/studio/foundry/studio/shell/chat-model-route.mjs`](../../../../../web/studio/foundry/studio/shell/chat-model-route.mjs), [`web/studio/foundry/studio/shell/uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs), [`web/studio/foundry/studio/shell/uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs)):**
  - Updated default/allowed UOR-R4 model routing to `uor-r4:qwen3-0.6b` and `uor-r4:qwen3-1.7b` with transparent legacy normalization from `qwen3-0.6b-foundry` -> `uor-r4:qwen3-0.6b`.
  - Added per-model `last_pack` cache checks, per-model local mirrors (`__UOR_V13_MIRRORS__` / `uor.v13.mirrors`), serialized `ensureModelLoadedQueued` + `generate`, safe reset of mounted/loaded state on swap failure, and `'ready'` progress signaling on cache/mount reuse.
  - Replaced `pieceBuffers` simultaneous RAM allocation in [`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs) with pack-by-pack OPFS/IDB streaming into `wasm.v13_push_piece` (preserving Stitch 2-of-3 recovery, `MessageChannel` yields, and sequential piece validation).
- **Prompts, Auto-RAG Domain Gating, Build Grounding & Preview Compat ([`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs), [`web/studio/foundry/studio/shell/host.js`](../../../../../web/studio/foundry/studio/shell/host.js), [`web/studio/foundry/studio/scripts/stamp_shell.py`](../../../../../web/studio/foundry/studio/scripts/stamp_shell.py)):**
  - Split `FOUNDRY_CHAT_SYSTEM_PROMPT`, `FOUNDRY_BUILD_SYSTEM_PROMPT`, and `FOUNDRY_SYSTEM_PROMPT` (`BUILD_TOOL_INTENT_RE` checks user `prompt`), with Build `maxNewTokens: 1024` and Chat `maxNewTokens: 512`.
  - Added `STANDARDS_STOP_WORDS` (`filterStopWords: true` in `buildTurnGroundingContext` only) + delimiter-aware matching + `FOUNDRY_DOMAIN_RE` gating in [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js).
  - Added `Current src/App.tsx` (`<current-file>`), `Last preview build error`, and `compactBuildHistory` for Build turns, `extractFallbackCodeWrites` (refusing traversal paths, supporting fragment `<>...</>` and trailing-prose closed fences), shared `transformProjectSource` across both `createProjectResolverPlugin` and `buildPortable`, `contextWindow` catalog propagation, and [`stamp_shell.py`](../../../../../web/studio/foundry/studio/scripts/stamp_shell.py) `src/main.tsx` default/named `App` compat.
- **UI Pickers & Test Suites ([`hologram.ModelPicker.tsx`](../../../../../web/studio/foundry/studio/vendor/prism/hologram.ModelPicker.tsx), [`hologram.CustomApiEndpointSettings.tsx`](../../../../../web/studio/foundry/studio/vendor/prism/hologram.CustomApiEndpointSettings.tsx), [`prism_r4_route.test.mjs`](../../../../../web/studio/foundry/studio/tools/prism_r4_route.test.mjs), [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs), [`web/studio/verify_foundry_studio.mjs`](../../../../../web/studio/verify_foundry_studio.mjs)):**
  - Updated picker defaults, scoped `hydrating`/`downloading`/`ready` pill status, unit contracts (including multi-line imports, destructured hooks, `BUILD_TOOL_INTENT_RE`, traversal blocking, worker/runtime queue guards), and Chrome CDP E2E coverage (casual Home Chat `"hi how are you"`, Todo Build Turn 1 + Turn 2 follow-up filter + live preview ready, and `uor-r4:qwen3-1.7b` pack-streamed mount + chat turn).

## Measurements

- Built `web/studio/dist/`: `485` files (`482` closure files in `manifest.json`, closure `27a26f963e02b47058ac1f4371a3d70ca9b156009caadfb915184d21964a835c`).
- `models.json`: `["qwen3-0.6b", "qwen3-1.7b"]`.
- `standards-index.json`: `49,644` bytes (`< 110 KB`), `31` sections, `sha256: 1b2b3710fcf9ca12eef66b0c478d5ddb7daed96a1d2538efcfad9d47b77d93e1`.
- `host.bundle.js`: `102` inputs, `1,041,082` bytes, `sha256: 346d64eefafe154f15c37e04ebf04c946ccaa0eab968b35dab5f828f73bee649`.
- Serving WASM (`web/studio/pkg/uor_r4_integer_bg.wasm`): `1,293,870` bytes, `sha256: 7ee2e36858e3a9bd327f2051c9c833792b7857e9b7f9fb8407f49e7246bad389`, `7/7` D80/D81 entrypoints PASS (`float=0`).
- `uor_foundry_v13_wasm_bg.wasm`: `11,431,737` bytes, `sha256: 855213514627bf885a38b0e1799d68a3d1f3b971366358700f67b2621e710ded` (`float_ops: 0`, `dense_matmul_ops: 0` observed on both `0.6B` and `1.7B` Chrome CDP turns).
- Ratchet baselines (`uor-harness ratchet`): `d23` `4451` across `165` files, `d25` `93` across `26` files (`0` tensor-core intrinsics), `d32` `31` across `1` files.

## Checks run at the head

All checks below ran cleanly after the final code edit on the worktree:
- `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` (`task-12963`): `Ran 35 tests in 114.664s`, `OK (skipped=1)`.
- `bash web/studio/build_foundry_studio.sh` (`task-12963`): PASS (`485` files, closure `27a26f963e02b47058ac1f4371a3d70ca9b156009caadfb915184d21964a835c`).
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)` (`task-12963`): PASS (`adds no type error to upstream's own (28)`, `35620 ms`).
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` (`task-12963`): PASS (`102 inputs, 1041082 bytes`).
- `node web/studio/foundry/studio/tools/shell_parse.mjs` (`task-12963`): PASS.
- `node --test` across the 7 UOR-R4 Foundry unit suites (`prism_r4_route`, `command_ownership_registry`, `custom_api_endpoint`, `generation_patch`, `studio_phase1_3`, `studio_phase4a`, `stage3_os_github_stitch`) (`task-12963`): **`34/34` PASS, `0` failed**.
- `node web/studio/verify_studio_worker.mjs` (`task-12963`): PASS (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`).
- `bash scripts/stage-studio-site.sh --typed` & `bash scripts/studio/runtime-files.sh --verify` (`task-12963`): PASS (`485` files, `482` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap` (`task-12963`): PASS (`ok 1..7`).
- `node web/studio/verify_foundry_studio.mjs web/studio/dist` (`task-12968`, Chromium CDP on `/` & `/uor-r4/studio/`): PASS (`ALL FOUNDRY STUDIO CDP CHECKS PASSED`; `Home Chat: "Hi! I'm here to help! How can I assist you today?"`; `Todo Build Turn 1 + Turn 2 follow-up filter edit & live preview OK`; `Qwen3-1.7B pack-streamed mount + chat turn OK: "Hello, I'm UOR-R4 Foundry Assistant."`; Linux 6.18.7 VM, MVM, GitHub, Stitch, Custom API, `.holo`).
- `uor-harness ratchet d23`, `d25`, `d32`: PASS (`4451`, `93`, `31` unchanged).

## Skills used

- Standard git, `wasm-pack`, Node/Python verification suites, headless Chromium CDP, and `uor-harness` loop workflow.

## Still open

- Staged harness checks, push update to PR `#4241`, fresh-context `pro` exact-head R2 approval, `[antigravity] uor-review v1` + CI merge to `main`, `--typed` deploy to `gh-pages`, live Chrome CDP check on `https://uor-foundation.github.io/uor-r4/studio/`, and worktree/branch cleanup (`$0.00` GPU spend).

## Plain summary

Replaced the broken Run 5 `0.6B` Foundry LoRA in Foundry Web Studio with both uncorrupted pure-integer `UORR4V13` models (`uor-r4:qwen3-0.6b` default + `uor-r4:qwen3-1.7b`), separated casual Chat, clean Build `<prism-write>`, and tool-intent prompts, gated Auto-RAG behind Foundry domain keywords, streamed `1.7B` packs incrementally from OPFS/IndexedDB into WASM, and remediated all PR `#4241` R1 review findings (mount/queue state reset on failed swaps, `prompt`-only tool-intent detection, shared multi-line React import transform across preview & portable `.holo` builds, and exact check reporting) — passing all post-remediation unit (`34/34`), Python (`35`), build, typecheck, WASM, ratchet, and dual-origin Chromium CDP checks (`0` float/matmul ops, `$0.00` GPU spend).
