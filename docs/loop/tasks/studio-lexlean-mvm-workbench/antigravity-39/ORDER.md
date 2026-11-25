# ORDER: studio-lexlean-mvm-workbench, antigravity-39

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (Stage A4: `afflom/verdant-engine` Sandboxed 3D Game Template Integration)

Execute Stage A4 of the approved 5-repo integration plan at `$0.00` GPU spend under strict CHARTER §3 quarantine (`verdant-tensor` and `uor-matmul` kept out of `crates/uor-r4-*`, zero floats/matmul/softmax in discrete paths, zero edits to `Cargo.toml`/`Cargo.lock`):

1. **Sandboxed `verdant-3d` Studio Template (`verdant-3d-template.mjs`):**
   - Create [`verdant-3d-template.mjs`](../../../../../web/studio/foundry/studio/shell/verdant-3d-template.mjs) (attributing `afflom/verdant-engine` `MIT OR Apache-2.0` `site/prism-bridge-meta.mjs`, `site/scene-renderer.mjs`, `templates/browser-3d-starter/game.mjs`) and register `"verdant-3d-template.mjs": "mjs"` in [`shell_parse.mjs`](../../../../../web/studio/foundry/studio/tools/shell_parse.mjs).
   - Export `VERDANT_3D_TEMPLATE_ID` (`"verdant-3d"`), `VERDANT_3D_PRISM_CONSTANTS`, `VERDANT_3D_TEMPLATE_META`, and `buildVerdant3dProjectFiles(baseScaffoldFiles)` producing an isolated React/TypeScript Orbital Beacon Arena project (`src/engine/verdant-engine.ts` self-contained WebGL2 + Canvas2D fallback 3D engine & Euler-Bernoulli modal chime audio with `dispose()`, `src/components/VerdantViewport.tsx`, `src/pages/Index.tsx`, and `AI_RULES.md`) with zero external dependencies or banned identifiers (`softmax`, `gemm`, `.matmul(`).
2. **Host Template & Standards Index Wiring:**
   - Update [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js) so `get-templates` includes `"verdant-3d"` alongside `"react"` (preserving `"react"` first), `create-app` resolves `templateId ?? settings?.selectedTemplateId ?? "react"`, and `window.__uorR4.verdantEngine` exposes template metadata & `buildProjectFiles()` while leaving Home Chat tool-free and IPC `command-registry` (`368/179/310/306`) unchanged.
   - Add `verdant-engine-3d-template` section to [`build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs) (`< 110 KB`) and live `esbuild` TS/TSX compilation + wiring tests in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs).

## Cancel probe

- Any edit touches `CHARTER.md`, `DECISIONS.md`, `docs/harness/native-space.json`, `Cargo.toml`, `Cargo.lock`, `crates/uor-r4-*`, `src-tauri/**`, or `web/studio/foundry/studio/generated/**`.
- Any `d23`/`d25`/`d32` ratchet (`4451`/`93`/`31`), `audit_zero_matmul_wasm.py` (`float=0`), `command_ownership_registry_browser.mjs` (`368/179/310/306`), or Chrome CDP check regresses.

## Scope

Writes inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-39/**`, `web/studio/foundry/studio/shell/**`, `web/studio/foundry/studio/tools/**`, and `scripts/studio/build-standards-index.mjs`.

## Done check

`bash web/studio/build_foundry_studio.sh`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `shell_parse.mjs`, all 10 relay/Foundry Node test suites, `verify_studio_worker.mjs`, `runtime-files.sh --verify web/studio/dist --typed`, `audit_zero_matmul_wasm.py`, `command_ownership_registry_browser.mjs`, Chrome CDP `verify_foundry_studio.mjs` plus live `"verdant-3d"` template preview smoke check, `uor-harness` checks/ratchets/`loop pr-check`, fresh `pro` exact-head review, merge, and live `gh-pages` Studio deploy PASS at `$0.00` GPU spend.
