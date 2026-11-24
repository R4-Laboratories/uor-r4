# ORDER for Foundry Studio V13 Dual-Model Recovery (`antigravity-31`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `3d1565c25` (`PR #4237`). Replaces the broken Run 5 `0.6B` Foundry LoRA (`weights-v13-qwen3-0.6b-foundry`, `3b04cd46`) with both published uncorrupted pure-integer `UORR4V13` models (`uor-r4:qwen3-0.6b` @ `081cb4f9` default + `uor-r4:qwen3-1.7b` @ `b35a6dcc`, `0` float/matmul ops, `$0.00` GPU spend), eliminates capability-list collapse and Auto-RAG poisoning on casual chat, adds pack-streaming OPFS/IDB worker mount for `1.7B`, fixes Build-mode `src/App.tsx` generation/follow-up editing and preview compilation, verifies E2E in Chrome CDP across `/` and `/uor-r4/studio/`, and prepares `main` for live `gh-pages` publication.

## Cancel probe

Verify `shell_parse.mjs`, all Foundry `tools/*.test.mjs`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `scripts/tests/test_typed_runtime.py`, full `web/studio/build_foundry_studio.sh`, `verify_studio_worker.mjs`, Chrome CDP `verify_foundry_studio.mjs` (`0.6B` + `1.7B`, natural Home Chat without capability collapse, Todo build + preview `app_run === 'ready'` + follow-up filter edit, both `/` and `/uor-r4/studio/`), `stage-studio-site.sh --typed`, `runtime-files.sh --verify`, `audit_zero_matmul_wasm.py`, `uor-harness check --staged`, `ratchet d23/d25/d32`, `uor-harness loop pr-check --local`, zero edits outside Studio/loop scope (`crates/**`, `src-tauri/**`, `CHARTER.md`, `DECISIONS.md`), `$0.00` GPU spend, and fresh-context `pro` exact-head `VERDICT: APPROVE` prior to merge.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-31/**`, `scripts/studio/**`, `scripts/tests/test_typed_runtime.py`, `web/studio/verify_foundry_studio.mjs`, and `web/studio/foundry/studio/**`:
- `scripts/studio/catalog.mjs`, `build-standards-index.mjs`, `web/studio/foundry/studio/shell/app.json`: configure dual `UORR4V13` catalog (`qwen3-0.6b` primary @ `081cb4f9` + `qwen3-1.7b` @ `b35a6dcc`) and remove `qwen3-0.6b-foundry`.
- `web/studio/foundry/studio/shell/chat-model-route.mjs`, `uor-r4-runtime.mjs`, `uor-v13-worker.mjs`, `vendor/prism/hologram.ModelPicker.tsx`, `hologram.CustomApiEndpointSettings.tsx`: allow both models + legacy migration, reset per-model hydration progress, per-model mirrors, and pack-by-pack OPFS/IDB streaming mount (`~50 MiB` peak JS RAM for `1.7B`).
- `web/studio/foundry/studio/shell/r4-agent-loop.mjs`, `host.js`, `scripts/stamp_shell.py`: split `FOUNDRY_CHAT_SYSTEM_PROMPT` vs `FOUNDRY_SYSTEM_PROMPT` (`maxNewTokens: 1024` in Build), gate Auto-RAG with stop-words + Foundry domain terms, inject `src/App.tsx` + `operationError` + history compaction in Build, support default/named `App` via `main.tsx` + `createProjectResolverPlugin`, and add fenced/truncated fallback code extraction.
- `web/studio/foundry/studio/tools/*.test.mjs`, `scripts/tests/test_typed_runtime.py`, `web/studio/verify_foundry_studio.mjs`: update assertions and verify Chat, Build Todo + follow-up edit + preview, and `1.7B` switch in Chrome CDP.
- `REPORT.md`: negatives-first closeout report.

## Done check

- All Foundry node tests, Python tests, `build_foundry_studio.sh`, `verify_studio_worker.mjs`, Chrome CDP `verify_foundry_studio.mjs` (`0.6B` + `1.7B`), `audit_zero_matmul_wasm.py`, `uor-harness check --staged`, `ratchet d23/d25/d32`, `uor-harness loop pr-check --local`, `pro` exact-head review, `[antigravity] uor-review v1`, and GitHub CI pass before merge.

## Stop and decide if

Stop and hand back to plan mode if Chrome CDP reveals a runtime/model failure unresolvable inside `web/studio/**` + `scripts/studio/**` without retraining or editing `crates/**`.
