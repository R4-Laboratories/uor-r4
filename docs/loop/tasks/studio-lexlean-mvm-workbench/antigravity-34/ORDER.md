# ORDER: studio-lexlean-mvm-workbench, antigravity-34

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (PR #4251 R1 Review Fix: `sealMap` Invalidation & Decode Telemetry Cleanup)

Address the blocking finding and non-blocking observations from PR `#4251` exact-head R1 review at `7e48892d4004e826ee6984fb677383d1b4617b20`:

1. Clear `sealMap = null;` inside `catch (sealedErr)` in [`web/studio/foundry/studio/shell/uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs) so falling back from a failed `sealCoversAll` sealed mount actually runs full SHA-256 verification and Stitch 2-of-3 recovery instead of short-circuiting on stale `sealMap` entries, and remove dead `statHits`.
2. Measure `decodeStartMs` from `stepStartMs` on the first `last.delta` slice (avoiding the `0.001 s` clamp spike on short turns) and reset `ttftMs: null, tokensPerSec: null` when returning to `phase: 'ready'` in `finally`.
3. Add a unit assertion in [`web/studio/foundry/studio/tools/studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs) covering `sealMap = null` fallback invalidation, rebuild/re-verify Studio, write `REPORT.md`, and run fresh `pro` exact-head R2 review.

## Cancel probe

- Any edit touches `crates/**`, `src-tauri/**`, `Cargo.lock`, `CHARTER.md`, `DECISIONS.md`, or `docs/harness/native-space.json`.

## Scope

Writes strictly inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-34/**`, `web/studio/foundry/studio/shell/uor-v13-worker.mjs`, `web/studio/foundry/studio/tools/studio_phase4a.test.mjs`, and rebuilt `web/studio/dist/**`.

## Done check

Rebuilt `web/studio/dist/**` via `build_foundry_studio.sh`, `host_bundle.mjs --check`, `shell_parse.mjs`, `hologram.typecheck.mjs`, 9 Foundry suites, `verify_studio_worker.mjs`, `runtime-files.sh --verify web/studio/dist --typed`, `audit_zero_matmul_wasm.py`, Chrome CDP `verify_foundry_studio.mjs`, `uor-harness` checks/ratchets/`loop pr-check`, and fresh `pro` exact-head `APPROVE`.
