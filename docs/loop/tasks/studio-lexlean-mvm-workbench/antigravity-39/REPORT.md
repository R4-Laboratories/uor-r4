# REPORT: studio-lexlean-mvm-workbench, antigravity-39

Lab: `antigravity`. Order: [`ORDER.md`](ORDER.md). Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7). Base: `5c7f71d67` (`ORDER.md` committed alone first at `f43b80741`). GPU spend: `$0.00`.

## What did not work

- Initial `VERDANT_AI_RULES_SUFFIX` in [`verdant-3d-template.mjs`](../../../../../web/studio/foundry/studio/shell/verdant-3d-template.mjs) named `verdant-tensor` and `uor-matmul` inside an instruction telling users not to add them; the Stage A4 forbidden-crate check in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs) caught those strings immediately (`task-17081`). Replaced the phrasing with generic offline guidance without naming forbidden crates; all `69/69` Node tests passed on rebuild (`task-17088`).
- Initial headless Chrome CDP verification run (`task-17093`) failed cold on `/` with transient `uor-r4:select-model` `PrismError`; an isolated CDP check verified `ensureModelLoaded('uor-r4:qwen3-0.6b')` (`cacheBackend: 'opfs'`, `sha256Verified: true`), and clean reruns completed both `/` and `/uor-r4/studio/` without error (`task-17103`, `task-17195`).
- Initial isolated `"verdant-3d"` CDP smoke script (`task-17111`) observed `appRunPhase: "starting"` because Foundry waits for `"Allow this preview"` consent on `dialog[aria-label="Review preview permissions"]` before starting `esbuild-wasm`; adding `clickAllow()` and navigating Chrome directly to the built `/p/` preview URL verified live 60 FPS WebGL2 rendering (`task-17140`, `task-17196`).
- R1 fresh-context `pro` exact-head adversarial review (`97735e33-889c-4347-a1fb-c186636a4c6b` at `82f3f719a`) returned `APPROVE` (0 `CRITICAL`, 0 `MAJOR`, 5 `MINOR`). Resolved all 5 polish items before final merge:
  1. `MINOR-1`: Swapped octahedron triangle vertex order (`b`/`c`) in `buildOctahedronMesh` so all 8 faces have CCW outward-pointing normals matching `WebGL2` / `Canvas2D` lighting.
  2. `MINOR-2`: Updated [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js) `create-app` to `(templateId ?? settings?.selectedTemplateId) || "react"` matching `ORDER.md` precedence (noting that the frozen renderer `CreateAppParamsSchema` strips custom `templateId` over IPC unless set via `selectedTemplateId`).
  3. `MINOR-3`: Reworded negative crate comments in `verdant-3d-template.mjs` and [`build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs) to avoid literal quarantined crate strings.
  4. `MINOR-4`: Reset `keys`, `virtualInput`, `jumpQueued`, `dragging`, `dragYaw`, and `dragPitch` in `resetWorld()`, added `window` `blur` + `pointercancel` handlers, called `e.preventDefault()` on `Space`/Arrows (outside text inputs), and added `onPointerCancel` to all 5 D-pad/jump buttons in `VerdantViewport.tsx`.
  5. `MINOR-5`: Avoided synchronous `dt = 0` `loop()` call in `mount()` so FPS starts cleanly at `60`, added `cleanupGlResources` for partial GL init failure, added a `0.35s` drone deflection cooldown, and added matching vertical sine bobbing to Canvas2D beacons.

## What changed

- **Sandboxed `verdant-3d` 3D game engine template (`MIT OR Apache-2.0` adapted from `afflom/verdant-engine`):**
  - [`web/studio/foundry/studio/shell/verdant-3d-template.mjs`](../../../../../web/studio/foundry/studio/shell/verdant-3d-template.mjs) (`43,558` B, SHA-256 `c7a622355d28d44389a33ad49e098a509c5d5c68d354c061ec5b612468a1557d`) & [`shell_parse.mjs`](../../../../../web/studio/foundry/studio/tools/shell_parse.mjs): Exports `VERDANT_3D_TEMPLATE_ID` (`"verdant-3d"`), `VERDANT_3D_PRISM_CONSTANTS`, `VERDANT_3D_TEMPLATE_META`, and `buildVerdant3dProjectFiles(baseScaffoldFiles)` overlaying without mutating `baseScaffoldFiles`:
    - `src/engine/verdant-engine.ts`: Self-contained offline TypeScript 3D engine (perspective/lookAt/`mul4x4` helpers with zero banned identifiers, `heightAt` coastal valley heightfield & indexed terrain mesh, outward CCW octahedron actors, 3 crystalline relay beacons, patrolling sentinel drone with 0.35s cooldown, physics kinetic sphere controller with `WASD`/Arrow/`Space` + drag-orbit camera + `chase`/`first-person`/`orbit` camera modes, WebGL2 directional-sun/sky-fill depth-buffered shader pipeline with automatic Canvas2D painter fallback, Web Audio Euler-Bernoulli modal bar chimes `[9, 25, 49, 81, 121]/9` gated on user toggle, deterministic FNV-1a world state hash, `blur`/`pointercancel` & `dispose()` cleanup).
    - `src/components/VerdantViewport.tsx` & `src/pages/Index.tsx`: Hologram-themed React HUD (`Beacons 0/3`, backend & FPS pill, state hash, camera/audio/pause/restart buttons, on-screen directional controls with `onPointerCancel`) mounting `VerdantEngine`.
    - `AI_RULES.md`: Appends local TypeScript 3D authoring rules without external URLs or quarantined crate names.
- **Host template routing & curated standards index:**
  - [`web/studio/foundry/studio/shell/host.js`](../../../../../web/studio/foundry/studio/shell/host.js): Registers `"verdant-3d"` in `get-templates` (preserving `"react"` first and default selection), updates `create-app` to apply `(templateId ?? settings?.selectedTemplateId) || "react"`, and exposes `window.__uorR4.verdantEngine` (`templateId`, `meta`, `buildProjectFiles`), leaving Home Chat tool-free and IPC `command-registry` (`368/179/310/306`) unchanged.
  - [`scripts/studio/build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs): Adds curated `verdant-engine-3d-template` section (`37` total sections, `< 110 KB`).
- **Tests:**
  - [`web/studio/foundry/studio/tools/studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs): Adds 2 Stage A4 tests checking `buildStandardsIndex`, `host.js` template wiring & clean Home Chat, `buildVerdant3dProjectFiles` immutability & offline/banned-token checks, and real `esbuild.transform` compilation of all `.ts`/`.tsx` template files with 0 warnings.

## Measurements

- **`verdant-3d-template.mjs`:** `43,558` B (SHA-256 `c7a622355d28d44389a33ad49e098a509c5d5c68d354c061ec5b612468a1557d`).
- **`standards-index.json`:** `57,693` B (`37` sections, SHA-256 `d8d5626f84604c4215e1ff05283e642fed08e363b11bb14507cb5e39deb38250`, `< 110 KB`).
- **`host.bundle.js`:** `1,105,131` B (SHA-256 `e24036d309741a689b4456bb107e4ad6c68d698d64a560889022004143c60654`).
- **`core.wasm`:** `1,033,712` B (SHA-256 `c3bbdbd959d3ea798cfafdd5b21cb523f63c2ebb93ae11c45bd696370978a761`).
- **`stamp-shell provenance` & closure:** `485` closure files verified, closure `70061bd10f0a8a424f1cec5497df4b92c957f04f2c844091937aeaed3fa516cc`.
- **WASM zero-matmul & zero-float audit (`uor_r4_integer_bg.wasm`, `1,289,272` B, SHA-256 `1f018be5955c034d04ecbcccc28388bca7f310b6add9a84293b448f40f3236ac`):** `ok 1..7`, `float=0`, `unclassified=0`.
- **Live `"verdant-3d"` template preview CDP (`task-17196`):** `problemsOutcome: "passed"`, `app_run.phase: "ready"`, `<canvas>` `722x518`, `"Verdant 3D — Orbital Beacon Arena"` rendering at `60 FPS` (`WebGL2 3D`, `Seed 7`, `exceptions: []`, `consoleErrors: []`).
- **Foundry Studio CDP E2E (`verify_foundry_studio.mjs web/studio/dist`):**
  - Root `/`: Home Chat Turn 1 (`34` chars) & Turn 2 (`126` chars), Build App `app=2` tool round-trip, Linux `6.18.7` VM, GitHub/Stitch, Custom AI API Endpoint, `Qwen3-1.7B` mount `42,466 ms` (`backend=opfs`), `TTFT=2,024 ms`, `7.77 tok/s`, `total=3,039 ms`.
  - Subpath `/uor-r4/studio/`: Home Chat Turn 1 (`34` chars) & Turn 2 (`126` chars), Build App `app=2` tool round-trip, Linux `6.18.7` VM, GitHub/Stitch, Custom AI API Endpoint, `Qwen3-1.7B` mount `40,534 ms` (`backend=opfs`), `TTFT=2,046 ms`, `7.77 tok/s`, `total=3,048 ms`.
- **GPU spend:** `$0.00` (`Running: 0 pod(s), $0/h of $8.00/h`).

## Checks run at the head

- `bash web/studio/build_foundry_studio.sh`: **PASS** (`core.wasm` `1,033,712` B, `standards-index.json` `57,693` B, `host.bundle.js` `1,105,131` B, `485` closure files).
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)`: **PASS**.
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` & `shell_parse.mjs`: **PASS**.
- `node --test web/studio/relay/relay.test.mjs web/studio/foundry/studio/tools/{prism_r4_route,command_ownership_registry,canonical_generation_ownership,preview_permissions,custom_api_endpoint,generation_patch,studio_phase1_3,studio_phase4a,stage3_os_github_stitch}.test.mjs`: **`69/69` PASS, `0` failed, `0` skipped**.
- `node web/studio/verify_studio_worker.mjs`: **PASS** (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`, `saveRestoreBitIdentical: true`).
- `bash scripts/studio/runtime-files.sh --verify web/studio/dist --typed`: **PASS** (`485` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap`: **PASS** (`ok 1..7`, `float=0`, `unclassified=0`).
- `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/foundry/studio/tools/command_ownership_registry_browser.mjs`: **PASS** (`commands: 368`, `handlers: 179`, `inputs: 310`, `outputs: 306`, `pageExceptions: 0`).
- Live `"verdant-3d"` template preview Chrome CDP smoke check: **PASS** (`appRunPhase: "ready"`, `<canvas>` `722x518`, `60 FPS`, `0` exceptions/errors).
- `CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/verify_foundry_studio.mjs web/studio/dist`: **PASS** (`ALL FOUNDRY STUDIO CDP CHECKS PASSED` on `/` and `/uor-r4/studio/`).

## Skills used

- None outside standard harness and Studio verification scripts.

## Still open

- Post-R1-polish `uor-harness` checks (`check --staged`, `ratchet`, `claim-wording`, `loop report --check`, `loop pr-check`), fresh-context `pro` R2 exact-head adversarial review (`APPROVE`), `[antigravity] uor-review v1`, CI + merge queue, and live `--typed` `gh-pages` publication (`$0.00` GPU spend).

## Plain summary

`antigravity-39` completes Stage A4 (`afflom/verdant-engine` sandboxed 3D template integration) and resolves all 5 R1 `MINOR` polish findings: adds `verdant-3d-template.mjs` (`MIT OR Apache-2.0` adapted offline WebGL2 + Canvas2D fallback Orbital Beacon Arena engine with outward CCW octahedron normals, `blur`/`pointercancel`/full state reset, `0.35s` drone cooldown, React HUD viewport, and procedural TypeScript AI rules with zero external dependencies or banned identifiers), wires `verdant-3d` into `host.js` (`get-templates`, `create-app` `(templateId ?? settings?.selectedTemplateId) || "react"`, `window.__uorR4.verdantEngine`) while keeping `"react"` default, Home Chat tool-free, and IPC command-registry (`368/179/310/306`) unchanged, and adds the `verdant-engine-3d-template` section to `standards-index.json` (`57,693` B, `37` sections). Build (`485` closure files), `hologram.typecheck`, `host_bundle --check`, `shell_parse`, Node (`69/69`), worker, WASM `float=0` audit, command-registry browser, live `"verdant-3d"` WebGL2 preview (`722x518`, `60 FPS`), and dual-origin Chrome CDP checks all pass at `$0.00` GPU spend.
