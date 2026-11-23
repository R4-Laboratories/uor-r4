# ORDER for Foundry Studio Phase 4A: V13 Speculative Decode, Prewarm, MessageChannel Yield, Auto-RAG Grounding & Cold-SW Hardening (`antigravity-15`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `ca165e41e` (PR #4207). Implements Phase 4A (Option A) under D90: upgrades external `uor-foundry-v13-wasm` with pure-integer 0-LSB speculative decoding (`HoloTok13SpeculativeDrafter` + `step_speculative_tile`), `v13_prewarm_prefix`, and `v13_configure` (`wasm_float_audit.json` `compute_float_ops == 0`); updates `uor-v13-worker.mjs` with `MessageChannel` micro-yields and 4-way pack hydration; adds `prewarmIfCached` + `configure` in `uor-r4-runtime.mjs`; adds Studio architecture/tool/runtime self-knowledge to `standards-index.json`, non-persisted workspace + auto-RAG grounding in `r4-agent-loop.mjs` and `host.js` (keeping `FOUNDRY_SYSTEM_PROMPT` byte-identical across turns); and hardens `buildPreview` + `compileTailwind` against cold-visit Service Worker activation races (`await workerActive`).

## Cancel probe

Verify `node web/studio/foundry/studio/tools/shell_parse.mjs`, `node --test` across all 11 UOR-R4 Foundry unit suites (including `studio_phase4a.test.mjs`), `bash web/studio/build_foundry_studio.sh`, `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)`, `node web/studio/foundry/studio/tools/host_bundle.mjs --check`, `node web/studio/verify_studio_worker.mjs`, `node web/studio/verify_foundry_studio.mjs web/studio/dist`, `scripts/stage-studio-site.sh --typed` + `scripts/studio/runtime-files.sh --verify`, `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap`, `python3 -m unittest discover -s scripts/tests`, `uor-harness ratchet d23/d25/d32`, `uor-harness check --staged`, `uor-harness loop pr-check --local`, zero `crates/**` / `src-tauri/**` / `.github/**` edits, zero binaries (§3.13), and fresh-context exact-head `VERDICT: APPROVE`.

## Scope

Under D90 (`docs/integration/DECISIONS.md`), `antigravity-15` edits only:
- `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-15/{ORDER.md,REPORT.md}`
- `web/studio/foundry-assets.lock.json`
- `scripts/studio/build-standards-index.mjs`
- `web/studio/foundry/studio/shell/{uor-v13-worker.mjs,uor-r4-runtime.mjs,r4-agent-loop.mjs,host.js}`
- `web/studio/foundry/studio/tools/studio_phase4a.test.mjs`
- `web/studio/verify_foundry_studio.mjs`

Zero edits to `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/foundry/studio/shell/r4/spec.json`, or owner-paths.

## Done check

- `uor-foundry-v13-wasm` v2 (`compute_float_ops == 0`, SIMD128 enabled) passes byte-identical `0-LSB` parity (`speculative: true` vs `false`), reuses `v13_prewarm_prefix` KV entries on Turn 1, and is published + SHA-256-locked in `foundry-assets.lock.json`.
- `uor-v13-worker.mjs`, `uor-r4-runtime.mjs`, `r4-agent-loop.mjs`, and `host.js` deliver idle cached-only prewarm (`externalReqs.length === 0` on fresh boot), `MessageChannel` yields, workspace + `standards-index.json` auto-RAG grounding without breaking `workspace_list` or mutating `messages` history, and `await workerActive` cold-SW publish safety.
- All unit, build, typecheck, worker, Python, WASM, ratchet, and Chrome CDP checks pass cleanly, followed by live `gh-pages` verification.

## Stop and decide if

Stop and hand back to plan mode if `uor-foundry-v13-wasm` speculative decode diverges from non-speculative output (`0-LSB` parity failure), WAT float audit reports `compute_float_ops > 0`, `verify_foundry_studio.mjs` regresses on Home Chat or Build App (`workspace_list`), or external wasm publication/lockfile verification fails.
