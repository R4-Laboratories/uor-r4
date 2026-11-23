# REPORT for Foundry Studio Build Chat Sync, `.holo` E2E & UI Cleanup (`antigravity-14`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `89b79ba3d` (`ORDER.md` committed at `27a5909ad`).

## What did not work

- **First-visit Service Worker reload race on subpath `/uor-r4/studio/`:** Initial `verify_foundry_studio.mjs` passed all checks on `/`, then exited 1 on `/uor-r4/studio/` (`CDP -32000: Inspected target navigated or closed`) because `host.js` marked `shellUpdate.waiting = true` on `shell-updated` from first-visit SW activation even when `!navigator.serviceWorker.controller` (`drawnByWorker === false`) and called `location.reload()` 1.5s after the chat turn finished. Guarded `shell-updated` with `drawnByWorker` (`Boolean(navigator.serviceWorker && navigator.serviceWorker.controller)`), matching `controllerchange`.
- **Vite `unapplied()` patch check & TS hook order caught during build:** Initial `hologram.screens.mjs` patches for `src/components/ChatModeSelector.tsx` (replaced whole by `hologram.ChatModeSelector.tsx`) and `src/components/LibraryList.tsx` (replaced via `AppSidebar.tsx`) were unapplied, and `handleHomeModeChange` referenced `updateSettings` before `useSettings()` in `src/pages/home.tsx`. Fixed hook ordering in `home.tsx`, dropped the dead `ChatModeSelector.tsx` / `LibraryList.tsx` patches, and removed the `Media` filter option in `LibraryFilterTabs.tsx`.
- **PR #4207 fresh-context exact-head review (`30cb558f-55c5-4008-85c7-3b31a6c23744`) & plan-review findings:**
  1. `clear-session-data` (`web/studio/foundry/studio/shell/host.js:3292`) deleted `"previews"` CacheStorage without clearing `liveFrameReady`, `liveFrameUrl`, and `coverSources`, causing subsequent `showLive()` / preview reloads / cover captures for projected apps to return a stale URL whose `/p/<id>/index.html` entry had been removed (`404 no preview`). It also deleted `"holo"` CacheStorage, which stores installed `.holo` application files served by `sw.template.js:185`. Updated `clear-session-data` to delete only `"previews"` (preserving `"holo"` and `"shell-*"`), reset `liveFrameReady = null; liveFrameUrl = null; coverSources.clear();`, and re-warm `liveFrame()` when `caches`, `fetch`, and `workerActive` are active. Added `liveAfterClearOk` and `holoCacheStillPresent` E2E assertions in `web/studio/verify_foundry_studio.mjs`. (Note: an initial regex literal inside `verify_foundry_studio.mjs`'s `evalAsync` template string triggered a `SyntaxError` on backslash escaping and was replaced with explicit `.endsWith('/')` checks.)
  2. Initial `REPORT.md` measurements carried stale `host.bundle.js` size and `web/studio/dist/` closure digest after a follow-up edit; all measurements and check outputs below were refreshed from the final rebuilt head.
- **Vendored upstream workflow tests:** Vendored Foundry tests under `web/studio/foundry/studio/tools/` that read `web/studio/foundry/.github/workflows/studio.yml` (`application_main_integration.test.mjs`, `chat_provider_routing.test.mjs`, `read_only_chat_execution.test.mjs`, `verified_host_bundle.test.mjs`, `verified_shell_cache.test.mjs`) fail `ENOENT` in `uor-r4` because that upstream workflow file does not live in this repository. All 10 UOR-R4 Foundry test suites passed (`60/60`).
- **Python test suite skip:** `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` ran `35` tests with `1` pre-existing skipped test and `0` failures.

## What changed

- **Phase 1 — Build Chat, Tool Loop & Workspace Sync:**
  - [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs): Added `holo_publish` / `holo_inspect` to `FOUNDRY_SYSTEM_PROMPT` and `FOUNDRY_TOOL_NAMES`, `repairToolJson` + fenced/alias normalization and structured `{ parseError }` retry rounds, configurable `maxToolRounds`, and non-build read-only `workspace_write` enforcement.
  - [`web/studio/foundry/studio/shell/linux-vm-host.mjs`](../../../../../web/studio/foundry/studio/shell/linux-vm-host.mjs): Returned `updatedFiles` from `execCommand`, excluded `.env` in `walk9pFiles` so env secrets stay out of workspace exports/commits, and added `writeEnvIntoVm` / `syncEnvToVm` invoked from `syncWorkspaceToVm` and `set-app-env-vars`.
  - [`web/studio/foundry/studio/shell/host.js`](../../../../../web/studio/foundry/studio/shell/host.js): Fixed duplicate user prompt (`messages.slice(0, -2)`), passed `maxToolRounds`, surfaced `unclosedWrites` warnings, removed the blocking `await showBuilt` / `await refreshPreview` before `chat:stream:end`, normalized/deduplicated `<prism-write>` paths via `parsePrismTags`, extended `version-created` + `applyWorkspacePatch` / `applyWorkspaceChanges` invalidations (`apps`, `app`, `versions`, `problems`), cleared stale `liveShown`, enforced non-build read-only guards on mutating tools, added `holo_publish` / `holo_inspect` tools, and guarded `shell-updated` reloads with `drawnByWorker`.
  - [`web/studio/foundry/studio/vendor/prism/hologram.screens.mjs`](../../../../../web/studio/foundry/studio/vendor/prism/hologram.screens.mjs): Invalidated `["app-files", "content", scope.appId]` in `src/window_infrastructure/query_keys.ts` so `FileEditor` refreshes on AI/VM/GitHub edits, and unified Home `/` `Chat` vs `Build App` mode controls across `home.tsx`, `HomeChatInput.tsx`, and `ChatInputControls.tsx`.
- **Phase 2 — `.holo` Publish, Download, Inspect, Import & Export:**
  - [`web/studio/foundry/studio/generated/upstream-provenance.json`](../../../../../web/studio/foundry/studio/generated/upstream-provenance.json) & [`web/studio/build_foundry_studio.sh`](../../../../../web/studio/build_foundry_studio.sh): Validated local `generated/` SHA-256 digests (`kernel.ir`, `coverage.json`, `roots.json`, `hologram_forge_core.rs`) and deterministically computed `lcnf_manifest_sha256`, `target_profile_id`, PAX gzip `cargo_crate_sha256`, `core_wasm_sha256`, and `browser_projection_sha256` into `provenance.json` before sealing.
  - [`web/studio/foundry/studio/shell/host.js`](../../../../../web/studio/foundry/studio/shell/host.js), [`web/studio/foundry/studio/shell/command-registry-data.mjs`](../../../../../web/studio/foundry/studio/shell/command-registry-data.mjs), [`web/studio/foundry/studio/tools/command_ownership_registry.test.mjs`](../../../../../web/studio/foundry/studio/tools/command_ownership_registry.test.mjs): Wired `holo:inspect`, `holo:import`, and `forge:export` adapters (`368` total registry entries, `179` active adapters).
  - [`web/studio/foundry/studio/vendor/prism/src/components/preview_panel/HologramSection.tsx`](../../../../../web/studio/foundry/studio/vendor/prism/src/components/preview_panel/HologramSection.tsx) & [`web/studio/foundry/studio/vendor/prism/hologram.ImportAppDialog.tsx`](../../../../../web/studio/foundry/studio/vendor/prism/hologram.ImportAppDialog.tsx): Wired direct `holo:publish`, per-archive `Inspect`, standalone `./holo.html` runner link, `Export Project (.json)`, and binary `.holo` import via `holo:import`.
- **Phase 3 — `Problems`, `Configure` Env Vars, Browser Adapters & Dead UI Cleanup:**
  - [`web/studio/foundry/studio/shell/host.js`](../../../../../web/studio/foundry/studio/shell/host.js): Extracted shared `createProjectResolverPlugin`, implemented real `esbuild` `check-problems` + `prism/spec.json` checks, injected safe `VITE_*` `appEnvDefine` into `buildPreview` / `buildPortable` / `check-problems`, wired `set-app-env-vars` invalidations + VM `.env` sync, updated `clear-session-data` (clears ephemeral `"previews"` + resets `liveFrameReady`/`liveFrameUrl`/`coverSources` and re-warms `liveFrame()`, preserves `"holo"` + `"shell-*"`), and implemented `set-chat-favorite` and `remove-chat-referenced-app`.
  - [`web/studio/foundry/studio/vendor/prism/hologram.screens/AppSidebar.tsx`](../../../../../web/studio/foundry/studio/vendor/prism/hologram.screens/AppSidebar.tsx) & [`web/studio/foundry/studio/vendor/prism/hologram.screens.mjs`](../../../../../web/studio/foundry/studio/vendor/prism/hologram.screens.mjs): Replaced `/plugins` with a read-only notice and removed `/plugins` from sidebar nav, removed `Basic Agent` (`local-agent`), replaced `dyad.sh`/`dyad-sh` URLs, and removed `GithubCollaboratorManager`, Visual Editor Pro lock box, `Generate Image` menu items, `analyzeComponent` call, `Rename master to main`/`UncommittedFilesBanner`, `CommitMenu`, `Show in folder`, and the `Media` filter tab.
- **Tests & E2E checks:**
  - Added [`web/studio/foundry/studio/tools/studio_phase1_3.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase1_3.test.mjs) and Check 10 (including `liveAfterClearOk` and `holoCacheStillPresent`) in [`web/studio/verify_foundry_studio.mjs`](../../../../../web/studio/verify_foundry_studio.mjs).

## Measurements

- Built Studio bundle (`web/studio/dist/`): `485` files staged (`482` closure files in `manifest.json`), closure digest `6d418bb10c806b858955a96d3721fbdfdbfdd401af2bc310ea3f735cfb5d687a` from `build_foundry_studio.sh` (updated on disk to `008f663a69b8b842f0a67e31a1ed19aaeb8ba488b60dc40efd45dc17bebb8674` when `test_tauri_runtime.py` re-ran `prepare-tauri-runtime.sh`, since `standards-index.json` carries a build-time `generated_at`).
- `stage-studio-site.sh --typed`: `485` file(s), `75,722,752` byte(s), `482` closure files verified.
- `host.bundle.js`: `102` inputs, `1,028,616` bytes, sha256 `de0313d01b976c014af702cefdde665935783877f789baeac2fcc21426176830`.
- Serving WASM (`web/studio/pkg/uor_r4_integer_bg.wasm`): `1,274,400` bytes, sha256 `6c6dd3e5a67794bbd0b4c0dd7f208139a271ad57005af396f866f0e84bd0dd07`, `7/7` entrypoints clean under D80/D81 audit.
- Command registry: `368` total entries, `179` active browser adapters.
- Ratchet baselines (`uor-harness ratchet`): `d23` `4451` across `165` files, `d25` `93` across `26` files (`0` tensor-core intrinsic sites), `d32` `31` across `1` files.

## Checks run at the head

- `node web/studio/foundry/studio/tools/shell_parse.mjs` — PASS (`every shell script parses under the goal the browser uses for it`).
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)` — PASS (`adds no type error to upstream's own (28), every ui call is rewritten for Radix (39408 ms)`).
- `bash web/studio/build_foundry_studio.sh` — PASS (`485` files staged, closure `6d418bb10c806b858955a96d3721fbdfdbfdd401af2bc310ea3f735cfb5d687a`).
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` — PASS (`102 inputs, 1028616 bytes, sha256 de0313d01b976c014af702cefdde665935783877f789baeac2fcc21426176830`).
- 10 UOR-R4 Foundry unit test suites (`command_ownership_registry.test.mjs`, `host_boundary.test.mjs`, `host_input_boundary.test.mjs`, `custom_api_endpoint.test.mjs`, `stage3_os_github_stitch.test.mjs`, `portable_build_permission.test.mjs`, `preview_permissions.test.mjs`, `publication_identity.test.mjs`, `bounded_build_modules.test.mjs`, `studio_phase1_3.test.mjs`) — `60/60` PASS, `0` failed.
- `node web/studio/verify_studio_worker.mjs` — PASS (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`).
- `node web/studio/verify_foundry_studio.mjs web/studio/dist` — PASS on both `/` and `/uor-r4/studio/` (`ALL FOUNDRY STUDIO CDP CHECKS PASSED`, including `liveAfterClearOk` and `holoCacheStillPresent`).
- `bash scripts/stage-studio-site.sh --typed web/studio /tmp/uor-stage-verify-p13 && bash scripts/studio/runtime-files.sh --verify /tmp/uor-stage-verify-p13 --typed` — PASS (`485` files, `482` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap` — PASS (`ok 1..7`).
- `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` — `Ran 35 tests in 130.631s`, `OK (skipped=1)`.
- `uor-harness ratchet d23`, `d25`, `d32` — PASS (`4451`, `93`, `31` unchanged).
- `uor-harness check --staged` & `uor-harness loop pr-check --local` — PASS (`verdict: PASS`).

## Skills used

- Standard git, Chrome CDP verification harness, and `uor-harness` loop workflow.

## Still open

- Final fresh-context `pro` exact-head approval on updated HEAD, PR #4207 merge into `main`, worktree cleanup, `--typed` deploy to `gh-pages`, live Chrome CDP check on `https://uor-foundation.github.io/uor-r4/studio/`, followed by planning Phase 4 (Option A + D) and Phase 5 (Option B).

## Plain summary

Unblocked Foundry Web Studio Build Chat from its post-turn preview deadlock and closed IDB/FileEditor/Linux-VM/GitHub sync gaps (`Phase 1`), wired full `.holo` publish/inspect/import/export with verified provenance (`Phase 2`), implemented real `esbuild` `check-problems`, `VITE_*` `Configure` env-var injection, browser adapters (`clear-session-data` preserving `.holo` caches and re-warming `liveFrame()`, `set-chat-favorite`, `remove-chat-referenced-app`), and dead UI cleanup (`Phase 3`), and verified all unit, worker, Python, WASM, ratchet, and dual-mount Chrome CDP checks cleanly.
