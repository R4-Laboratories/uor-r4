# REPORT: studio-lexlean-mvm-workbench, antigravity-34

Lab: antigravity. Order: [`ORDER.md`](ORDER.md). Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (`#3939`, M7).

## What did not work

- In Chromium CDP verification (`verify_foundry_studio.mjs`), the test creates an ephemeral Chrome profile (`mkdtemp`) per run, so cold streaming mount (`Qwen3-1.7B`) runs directly in CDP while warm `sealCoversAll` invalidation on corrupted OPFS (`sealMap = null;` inside `catch (sealedErr)`) is verified via unit assertions in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs) rather than synthetic disk corruption inside Chrome.
- `command_ownership_registry_browser.mjs` on macOS requires `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"` when Playwright's standalone Chromium binary is absent.

## What changed

- [`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs): Added `sealMap = null;` inside `catch (sealedErr)` so failed warm `sealCoversAll` mounts fall back cleanly to SHA-256 verification and Stitch 2-of-3 recovery; removed unused `statHits`; tracked `decodeStartMs = stepStartMs` on the first `last.delta` slice; and reset `ttftMs: null, tokensPerSec: null` when returning to `phase: 'ready'` in `finally`.
- [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs): Added Stage 4 regression assertions for `sealMap = null;` fallback invalidation, removal of `statHits`, `decodeStartMs` tracking, and `phase: 'ready'` telemetry reset.

## Measurements

- **Root (`/`):** `Qwen3-1.7B` pack-streamed mount `45,063 ms` (`opfs`), `TTFT=2,243 ms`, decode `6.93 tok/s`, `total=3,385 ms`; `float_ops=0`, `dense_matmul_ops=0`.
- **Subpath (`/uor-r4/studio/`):** `Qwen3-1.7B` pack-streamed mount `52,080 ms` (`opfs`), `TTFT=2,406 ms`, decode `7.22 tok/s`, `total=3,486 ms`; `float_ops=0`, `dense_matmul_ops=0`.
- **GPU spend:** `$0.00` (`Running: 0 pod(s), $0/h`).

## Checks run at the head

- `bash web/studio/build_foundry_studio.sh`: PASS (485 staged files, `standards-index.json` 51,078 B / 32 sections, `0` Python invocations).
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)`: PASS (`adds no type error to upstream's own (28)`).
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` & `node web/studio/foundry/studio/tools/shell_parse.mjs`: PASS.
- 9 Foundry suites (`prism_r4_route`, `command_ownership_registry`, `canonical_generation_ownership`, `preview_permissions`, `custom_api_endpoint`, `generation_patch`, `studio_phase1_3`, `studio_phase4a`, `stage3_os_github_stitch`): `48/48` PASS, `0` failed.
- `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/foundry/studio/tools/command_ownership_registry_browser.mjs`: PASS (`commands: 368`, `handlers: 179`, `inputs: 310`, `outputs: 306`, `pageExceptions: 0`).
- `node web/studio/verify_studio_worker.mjs`: PASS (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`, `saveRestoreBitIdentical: true`).
- `bash scripts/studio/runtime-files.sh --verify web/studio/dist --typed`: PASS (`482` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap`: PASS (`ok 1..7`, `float=0`, `sha256: 414179c08df88d38fa12f26c40afbb8c86e0c18d1bbc52d64f3471a7ba78161c`).
- `node web/studio/verify_foundry_studio.mjs web/studio/dist`: PASS (`ALL FOUNDRY STUDIO CDP CHECKS PASSED` across `/` and `/uor-r4/studio/`).

## Skills used

- None outside standard harness and Studio verification scripts.

## Still open

- Fresh-context `pro` exact-head R2 adversarial review on PR `#4251`, merge queue squash merge, and live `gh-pages` Studio deployment.

## Plain summary

1. **Invalidated `sealMap` on sealed-mount fallback ([`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs)):** Added `sealMap = null;` inside `catch (sealedErr)` so failed warm `sealCoversAll` mounts fall back cleanly to full SHA-256 verification and Stitch 2-of-3 recovery, and removed unused `statHits`.
2. **Fixed single-slice `tok/s` clamp spike & `ready` telemetry reset ([`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs)):** Tracked `decodeStartMs = stepStartMs` on the first `last.delta` slice and reset `ttftMs: null, tokensPerSec: null` when returning to `phase: 'ready'` in `finally`.
3. **Added regression assertions ([`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs)):** Covered `sealMap = null`, removal of `statHits`, `decodeStartMs`, and `phase: 'ready'` telemetry reset, then rebuilt and re-verified all host, worker, WASM, and Chrome CDP suites.
