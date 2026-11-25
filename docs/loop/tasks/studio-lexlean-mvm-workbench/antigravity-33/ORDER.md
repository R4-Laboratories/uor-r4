# ORDER: studio-lexlean-mvm-workbench, antigravity-33

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (Stage 4: Browser Studio Acceleration, 1.7B Prewarm Fix, Live Telemetry & Python Audit)

Following completed stages `1A` (`#4248`), `1B` (`#4249`), and `1C` (`#4250`), deliver Stage 4 browser Studio acceleration, bug fixes, and standalone Studio Python audit (`$0.00` GPU spend):

1. **Single-pass streaming mount & zero redundant piece hashing (`uor-v13-worker.mjs`):**
   - Add zero-read `cacheStat()` (`createSyncAccessHandle().getSize()` / `getFile().size`); on warm OPFS cache with verified `pack_seal.json`, mount every pack in one OPFS read pass instead of two (halving `1.7B` disk reads from `2.28 GB` to `1.14 GB`).
   - Skip redundant per-piece SHA-256 slicing whenever `packVerified === true` (already hash-verified as a pack; only inspect slices on damaged packs before Stitch 2-of-3 recovery).
   - Pipeline cold streaming mount directly from downloaded RAM pack buffers in contiguous pack order while persisting asynchronously to OPFS/IDB, falling back cleanly if any pack fails verification.
2. **Live prefill progress & TTFT / `tok/s` telemetry (`uor-v13-worker.mjs`, `uor-r4-runtime.mjs`, `HomeChatSurface.tsx`):**
   - Default `prefillSliceTokens = 64` (halving prefill WASM boundary round-trips while keeping cancellation responsive) and emit `prefilling` (`prefillDone/prefillTotal`) and live `generating` (`tokensPerSec`, `ttftMs`) progress events into `StudioModelStatusBar`.
3. **Fix `1.7B` prewarm / silent-unmount bug & dynamic Home Chat budget (`uor-r4-runtime.mjs`, `host.js`):**
   - Guard `prewarmIfCached` with `other_model_mounted` unless `allowModelSwitch` is true so background `0.6B` prewarm never evicts an active `1.7B` mount.
   - Resolve `selectedModel` in `host.js` for `uor-r4:status`, `runIdlePrewarm`, and `uor-r4:select-model` (`prewarmIfCached` immediately after explicit model switch).
   - Compute dynamic `max_seq`-aware `maxNewTokens` (`512..1024`) in `home-chat:send` (instead of fixed `512`).
4. **Foundry Studio Python audit & cleanups (`web/studio/build_foundry_studio.sh`, `REPORT.md`):**
   - Update stale `stamp_shell.py` comment in `build_foundry_studio.sh`, verify zero Python in shipped Studio runtime/build path (`build_foundry_studio.sh`), and catalog all standalone Studio test/vendor Python files in `REPORT.md`.

## Cancel probe

- Any edit touches `crates/**`, `src-tauri/**`, `Cargo.lock`, `CHARTER.md`, `DECISIONS.md`, or `docs/harness/native-space.json`.
- Chrome CDP E2E (`verify_foundry_studio.mjs`), worker checks (`verify_studio_worker.mjs`), `command_ownership_registry_browser.mjs` (`368`/`179` inventory), `hologram.typecheck.mjs`, or any of the 9 Foundry suites regress.

## Scope

Writes strictly inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-33/**`, `web/studio/build_foundry_studio.sh`, `web/studio/verify_foundry_studio.mjs`, and `web/studio/foundry/studio/**`.

## Done check

`bash web/studio/build_foundry_studio.sh` (0 Python calls), `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `shell_parse.mjs`, 9 Foundry suites (`node --test`), `command_ownership_registry_browser.mjs`, `verify_studio_worker.mjs`, `runtime-files.sh --verify web/studio/dist --typed`, `audit_zero_matmul_wasm.py`, Chrome CDP `verify_foundry_studio.mjs` (measuring `0.6B` & `1.7B` mount, prewarm, TTFT, tok/s), `uor-harness` checks/ratchets/`loop pr-check`, `pro` exact-head review, merge, and live `gh-pages` deploy PASS at `$0.00` GPU spend.
