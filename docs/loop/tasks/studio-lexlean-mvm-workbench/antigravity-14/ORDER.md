# ORDER for Foundry Studio Build Chat Sync, `.holo` E2E & UI Cleanup (`antigravity-14`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `89b79ba3d` (PR #4201). Resolves Phases 1–3 of the Studio audit: unblocks Build Chat completion from the 5-minute `authorizePreview` deadlock, fixes duplicate user prompts, normalizes/repairs `<prism-write>` and `<tool_call>` execution, closes IDB/FileEditor/Preview/Linux-VM/GitHub sync gaps, unifies Home `/` `Chat` vs `Build App` controls, enables `.holo` publish/download/inspect/import/export, wires real `Problems` and `Configure` env-var controls, and removes dead/Dyad-Pro UI surfaces.

## Cancel probe

Verify `node web/studio/foundry/studio/tools/shell_parse.mjs`, `node --test web/studio/foundry/studio/tools/*.test.mjs`, `bash web/studio/build_foundry_studio.sh`, `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)`, `node web/studio/foundry/studio/tools/host_bundle.mjs --check`, `node web/studio/verify_studio_worker.mjs`, `node web/studio/verify_foundry_studio.mjs web/studio/dist`, `scripts/stage-studio-site.sh --typed` + `scripts/studio/runtime-files.sh --verify`, `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph`, `python3 -m unittest scripts/tests/test_typed_runtime.py scripts/tests/test_release_served_forward_gate.py scripts/tests/test_tauri_runtime.py`, `uor-harness ratchet d23`, `uor-harness ratchet d25`, `uor-harness ratchet d32`, `uor-harness ratchet claim-wording`, `uor-harness check --staged`, zero `crates/**` / `src-tauri/**` / `.github/**` edits, zero binaries (§3.13), and fresh-context exact-head `VERDICT: APPROVE`.

## Scope

Under D90 (`docs/integration/DECISIONS.md`), `antigravity-14` edits only:
- `web/studio/build_foundry_studio.sh`
- `web/studio/foundry/studio/generated/upstream-provenance.json`
- `web/studio/foundry/studio/shell/{host.js,r4-agent-loop.mjs,command-registry-data.mjs,linux-vm-host.mjs}`
- `web/studio/foundry/studio/vendor/prism/{hologram.screens.mjs,hologram.ImportAppDialog.tsx,hologram.no-remote-model.mjs,hologram.screens/AppSidebar.tsx,hologram.screens/HomeChatSurface.tsx,src/components/preview_panel/HologramSection.tsx}`
- `web/studio/foundry/studio/tools/{shell_parse.mjs,command_ownership_registry.test.mjs,studio_phase1_3.test.mjs}`
- `web/studio/{verify_foundry_studio.mjs,verify_studio_worker.mjs}`
- `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-14/{ORDER.md,REPORT.md}`

Zero edits to `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/foundry/studio/shell/r4/spec.json`, or owner-paths.

## Done check

- Build Chat turns complete to `phase: "idle"` after writing files without blocking on `authorizePreview`, avoid duplicating user prompts, normalize `./` / `/workspace/` paths, repair/guard tool calls, and keep `FileEditor`, `FileTree`, `Preview`, GitHub pull, and Linux VM `/workspace` synchronized.
- Home `/` has one unified `Chat` vs `Build App` selector (`Build App` always creates in `build` mode), and `DefaultChatModeSelector` offers only valid browser modes.
- `holo:publish` returns `Accept` with downloadable `.holo` v4 archives; `holo:inspect`, `holo:import`, `holo.html` links, `forge:export`, and `holo_publish`/`holo_inspect` AI tools work end-to-end.
- `Problems`, `Configure` (`VITE_*`), `clear-session-data`, `set-chat-favorite`, and `remove-chat-referenced-app` work cleanly; dead Pro/desktop controls and `dyad.sh` links are removed.

## Stop and decide if

Stop if `core.wasm` `compose` rejects derived `provenance.json` or requires `web/studio/foundry/studio/core/` Rust edits, or if `host_boundary.test.mjs` / `command_ownership_registry.test.mjs` reveals a boundary conflict.
