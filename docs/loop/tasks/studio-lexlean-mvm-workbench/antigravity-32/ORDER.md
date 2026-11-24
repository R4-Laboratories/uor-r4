# ORDER for Foundry Studio Option A Usability, Speed & Rust Port (`antigravity-32`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `ecda1987e` (`PR #4241`). Executes Casey's Option A Studio Sprint (`$0.00` GPU spend, zero edits to `crates/**`, `src-tauri/**`, `CHARTER.md`, or `DECISIONS.md`): fixes `#/chat` App Build usability (`chat:cancel` boolean contract, post-cancel/error queue auto-resume, early-mount draft retention, `StudioModelStatusBar`, non-stealing permission dialog focus, duplicate mode UI inspection), adds surgical `<prism-search-replace>` patch editing + dynamic `max_seq`-aware output budgets + 1-step compile self-repair, accelerates in-browser V13 execution via OPFS `createSyncAccessHandle()` + verified pack seal + default 0-LSB speculative decoding + Build-mode prewarm, adds concise UOR-R4 self-awareness and architecture grounding, and ports `web/studio/build_foundry_studio.sh`'s Python build steps into `hologram-forge-core`'s locked Rust binary `stamp-shell`.

## Cancel probe

Verify zero `python3` invocations in `web/studio/build_foundry_studio.sh`, `--locked` `stamp-shell` Rust build + exact `provenance.json` & editor/compiler receipts, `shell_parse.mjs`, all Foundry `tools/*.test.mjs` + new unit tests in `studio_phase4a.test.mjs`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `verify_studio_worker.mjs`, Chrome CDP `verify_foundry_studio.mjs` (real `#/chat` typing/Enter/Cancel/re-send/`Ask` mode, `StudioModelStatusBar`, UOR self-awareness `"are you a transformer?"`, Home Chat, Todo build Turn 1 + Turn 2 + preview on `/` & `/uor-r4/studio/`), `stage-studio-site.sh --typed`, `runtime-files.sh --verify`, `audit_zero_matmul_wasm.py`, `uor-harness check --staged`, `ratchet d23/d25/d32`, `uor-harness loop pr-check --local`, zero edits outside scope, `$0.00` GPU spend, and fresh-context `pro` exact-head `VERDICT: APPROVE` prior to merge.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-32/**`, `web/studio/build_foundry_studio.sh`, `web/studio/verify_foundry_studio.mjs`, `scripts/studio/build-standards-index.mjs`, and `web/studio/foundry/studio/**`:
- `web/studio/foundry/studio/shell/chat-queue-controller.mjs`, `host.js`, `preview-permissions.mjs`, `vendor/prism/hologram.screens.mjs`, `hologram.screens/HomeChatSurface.tsx`: fix `chat:cancel` boolean return, auto-resume paused queue on `SUBMIT`, preserve pre-`chatId` typing, add `StudioModelStatusBar` (`uor-r4:progress`) to `#/chat`, guard preview focus stealing, add `<prism-search-replace>` parsing/application + dynamic token budget + 1-step compile self-repair, and add `uor-r4:prewarm` (`chat`/`build`).
- `web/studio/foundry/studio/shell/r4-agent-loop.mjs`, `uor-r4-runtime.mjs`, `uor-v13-worker.mjs`, `scripts/studio/build-standards-index.mjs`: enable OPFS `createSyncAccessHandle()` + `pack_seal.json` + concurrency 6 + default 0-LSB `speculative: true` + sparse shortlist probe; split/teach concise UOR-R4 self-awareness and `<prism-search-replace>` edits.
- `web/studio/foundry/studio/core/Cargo.toml`, `core/src/bin/stamp_shell.rs`, `web/studio/build_foundry_studio.sh`: replace Python build calls in `build_foundry_studio.sh` with `--locked` Rust binary `stamp-shell`.
- `web/studio/foundry/studio/tools/studio_phase4a.test.mjs`, `web/studio/verify_foundry_studio.mjs`, `REPORT.md`: unit/E2E tests and closeout report.

## Done check

- All Foundry node tests, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `build_foundry_studio.sh` (Rust `stamp-shell`, zero `python3`), `verify_studio_worker.mjs`, Chrome CDP `verify_foundry_studio.mjs`, `audit_zero_matmul_wasm.py`, `uor-harness` checks/ratchets/pr-check, `pro` exact-head review, `[antigravity] uor-review v1`, and CI pass before merge.

## Stop and decide if

Stop and hand back to plan mode if `stamp-shell` requires unlocking `core/Cargo.lock`, if `speculative: true` breaks 0-LSB/stability in CDP, or if an unpatched Prism modal/input blocker persists on `#/chat`.
