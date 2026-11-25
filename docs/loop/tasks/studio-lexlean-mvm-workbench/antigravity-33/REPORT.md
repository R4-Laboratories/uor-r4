# REPORT: studio-lexlean-mvm-workbench, antigravity-33

Lab: `antigravity`. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7). Order: [ORDER.md](ORDER.md). Base: `80b856c91` (`ORDER.md` committed alone first at `1d824a997`). GPU spend: `$0.00`.

## What did not work, first

- **First `/` Check 2 transient failure before bounding streaming buffer (`task-15188` vs `task-15254`):** The first `verify_foundry_studio.mjs` run after introducing `pendingMountPacks` threw once on `/` Check 2 (`The browser operation failed: uor-r4:select-model`). We added a bounded 3-pack RAM window (`MAX_STREAM_MOUNT_BUFFERED_PACKS = 3`) plus `pendingOutOfOrderPacks` fallback to OPFS/IDB in [`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs) so concurrent downloads never hold more than 3 packs (~150 MB) in JS heap simultaneously; after rebuilding, unmodified `verify_foundry_studio.mjs` passed both `/` and `/uor-r4/studio/` completely (`task-15254`).
- **Pre-existing missing vendored upstream fixtures in `tools/*.test.mjs`:** Unchanged from `antigravity-32` (`61/95` upstream+UOR scripts depend on absent vendored fixtures or `.github/workflows/studio.yml`); all 9 targeted Foundry suites pass (`48/48`).
- **Cold-cache CDP run vs warm OPFS reload:** `verify_foundry_studio.mjs` creates a fresh ephemeral Chrome profile per run (`mkdtemp`), so the timed `45.68s`/`47.16s` `1.7B` runs exercise the pipelined cold download + OPFS write + direct-RAM push path (`streamMountActive`); the single-pass warm OPFS fast-path (`sealCoversAll` + `cacheStat`) is verified statically in `studio_phase4a.test.mjs` and shares the exact `v13_finalize_mount()` container SHA-256 verification.

## What changed

- **Single-pass mount, bounded streaming, zero redundant piece hashing & live telemetry ([`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs)):**
  - Added zero-read `cacheStat()` (`createSyncAccessHandle().getSize()` / `getFile().size`) and `sealCoversAll` fast-path reading each OPFS pack once when `pack_seal.json` and file sizes match (cutting warm `1.7B` disk reads from `2.28 GB` to `1.14 GB`).
  - Removed redundant per-piece `sha256Hex()` hashing on already-verified packs (`if (packVerified && packBytes)`), keeping per-piece inspection exclusively for damaged packs heading into Stitch 2-of-3 recovery.
  - Added contiguous pipelined streaming mount bounded by `MAX_STREAM_MOUNT_BUFFERED_PACKS = 3` and `pendingOutOfOrderPacks` so cold packs mount straight from RAM while downloading/writing to OPFS without unbounded heap growth.
  - Updated `handleGenerateV13` to default `prefill_slice_tokens: payload.prefillSliceTokens || 64` and emit live `v13_progress` updates during `prefilling` (`prefillDone`, `prefillTotal`) and `generating` (`ttftMs`, `tokensPerSec`, resetting to `ready` in `finally`).
- **Prewarm isolation, dynamic Home Chat budget & UI status bar ([`uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs), [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js), [`HomeChatSurface.tsx`](../../../../../web/studio/foundry/studio/vendor/prism/hologram.screens/HomeChatSurface.tsx), [`build_foundry_studio.sh`](../../../../../web/studio/build_foundry_studio.sh)):**
  - Added `allowModelSwitch = false` / `skipped: 'other_model_mounted'` in `prewarmIfCached` so background `0.6B` prewarm never unmounts `Qwen3-1.7B`.
  - Added `resolveSelectedUorR4ModelId()` in `host.js` for `uor-r4:status` and `runIdlePrewarm`, triggered non-blocking prewarm immediately inside `uor-r4:select-model`, computed dynamic `max_seq`-aware `computedHomeMaxNew` (`512..1024`) in `home-chat:send`, surfaced `Prefilling ...` and `Generating ... (tok/s · TTFT ms)` in `StudioModelStatusBar`, and updated the stale `stamp_shell.py` comment in `build_foundry_studio.sh`.
- **Tests ([`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs), [`verify_foundry_studio.mjs`](../../../../../web/studio/verify_foundry_studio.mjs)):** Added Stage 4 contract assertions and Chrome CDP checks for `other_model_mounted` non-eviction + `1.7B` mount/TTFT/`tok/s`.

## Studio Python Audit

| Location | Usage in shipped Studio or build | Status |
|---|---|---|
| `web/studio/build_foundry_studio.sh`, `build_serve_wasm.sh`, `build_mvm_wasm.sh` | Main web bundle + WASM builds | **0** Python invocations (uses Rust `stamp-shell`, `project-shell`, `wasm-pack`, `vite`) |
| `scripts/studio/catalog.mjs:45` | Optional `--trained-manifest` / `--receipt-gate` path only | Not invoked by default `build_foundry_studio.sh` or `stage-studio-site.sh --typed` |
| `scripts/studio/build-and-publish.sh:187` | Repo-wide D11 WASM instruction audit (`scripts/audit_zero_matmul_wasm.py`) | Build/release gate only (verifies `float=0` across `uor_r4_integer_bg.wasm`), not shipped in browser dist |
| `web/studio/foundry/studio/{scripts,tools}/*.py` (`22` files) | Upstream Foundry authoring/vendor/projection/test utilities (`author.py`, `brand_kit.py`, `corpus*.py`, `model_*.py`, `stamp_shell.py`, `editor_assets*.py`, `compiler_assets*.py`, etc.) | Standalone developer/upstream utilities; **0** shipped in `web/studio/dist/` or invoked by `build_foundry_studio.sh` |

## Measurements (`$0.00` pod spend)

- Built `web/studio/dist/`: `485` files (`Foundry runtime + 0 model artifacts + 482 closure files verified`), `0` Python invocations.
- Serving WASM (`web/studio/pkg/uor_r4_integer_bg.wasm`): `sha256: 414179c08df88d38fa12f26c40afbb8c86e0c18d1bbc52d64f3471a7ba78161c`, `7/7` D80/D81 entrypoints PASS (`float=0`).
- Dual-origin Chrome CDP E2E (`verify_foundry_studio.mjs` on `/` & `/uor-r4/studio/`):
  - **Root `/`:** `Qwen3-1.7B` pipelined mount `45,679 ms` (`opfs`), `nonEvictPrewarm.skipped === 'other_model_mounted'`, `TTFT=2,152 ms`, `11.42 tok/s`, `total=3,204 ms` (`float_ops=0`, `dense_matmul_ops=0`).
  - **Subpath `/uor-r4/studio/`:** `Qwen3-1.7B` pipelined mount `47,156 ms` (`opfs`), `nonEvictPrewarm.skipped === 'other_model_mounted'`, `TTFT=2,612 ms`, `8.86 tok/s`, `total=3,969 ms` (`float_ops=0`, `dense_matmul_ops=0`).

## Checks run at the head

- `bash web/studio/build_foundry_studio.sh` (`task-15254`) — **PASS** (`485` files, `0` Python invocations).
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)` (`task-15254`) — **PASS** (`adds no type error to upstream's own (28)`).
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` & `node web/studio/foundry/studio/tools/shell_parse.mjs` — **PASS**.
- `node --test` across the 9 targeted Foundry suites (`prism_r4_route`, `command_ownership_registry`, `canonical_generation_ownership`, `preview_permissions`, `custom_api_endpoint`, `generation_patch`, `studio_phase1_3`, `studio_phase4a`, `stage3_os_github_stitch`) — **`48/48` PASS, `0` failed**.
- `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/foundry/studio/tools/command_ownership_registry_browser.mjs` (`task-15254`) — **PASS** (`368` commands, `179` handlers, `310` inputs, `306` outputs, `0` page exceptions).
- `node web/studio/verify_studio_worker.mjs` (`task-15254`) — **PASS** (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`).
- `bash scripts/studio/runtime-files.sh --verify web/studio/dist --typed` (`task-15254`) — **PASS** (`482` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap` (`task-15254`) — **PASS** (`ok 1..7`, `float=0`).
- `CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/verify_foundry_studio.mjs web/studio/dist` (`task-15254`) — **PASS** (`ALL FOUNDRY STUDIO CDP CHECKS PASSED` on `/` & `/uor-r4/studio/`).
- Relative markdown link check, `uor-harness check --staged`, `uor-harness ratchet d23|d25|d32`, `uor-harness claim-wording`, `uor-harness loop report --check --order docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-33/ORDER.md`, `uor-harness loop pr-check --local`, `git diff --check --cached`, and `scripts/pod/uor-pod status` (`Running: 0 pod(s)`) — **PASS**.

## Skills used

- Standard git, Cargo (`--locked`), `wasm-pack`, Node test runner, headless Chromium CDP, and `uor-harness` loop workflow.

## Still open

- Fresh-context `pro` exact-head adversarial review, `[antigravity] uor-review v1`, CI + merge queue into `main`, `--typed` `gh-pages` deployment + live `https://uor-foundation.github.io/uor-r4/studio/` check, and worktree/branch cleanup (`$0.00` GPU spend).

## Plain summary

`antigravity-33` (Stage 4) eliminates redundant warm OPFS pack reads (`cacheStat` + single-pass `sealCoversAll`), removes redundant per-piece SHA-256 hashing on verified packs, adds bounded pipelined streaming mount (`MAX_STREAM_MOUNT_BUFFERED_PACKS = 3`), halves prefill WASM round-trips (`prefillSliceTokens = 64`), exposes live prefill progress and `TTFT`/`tok/s` telemetry in `StudioModelStatusBar`, prevents background `0.6B` prewarm from unmounting `Qwen3-1.7B` (`other_model_mounted`), gives Home Chat a dynamic `512..1024` output token budget, and audits all Studio Python scripts — passing `48/48` targeted Foundry tests, `command_ownership_registry_browser.mjs`, `audit_zero_matmul_wasm.py`, and full dual-origin Chrome CDP E2E (`1.7B` mount `45.7–47.2s`, `TTFT=2.15–2.61s`, `8.86–11.42 tok/s`, `$0.00` GPU spend).
