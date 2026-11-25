# REPORT: studio-lexlean-mvm-workbench, antigravity-37

Lab: `antigravity`. Order: [`ORDER.md`](ORDER.md). Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7). Base: `1317048cf` (`ORDER.md` committed alone first at `59066f66e`). GPU spend: `$0.00`.

## What did not work

- Adding `verdant_studio.rs` (`852,859` B) caused `stamp-shell provenance` in `web/studio/build_foundry_studio.sh` step 8/9 to hang indefinitely when archiving crate sources into a `.tar.gz`: [`stamp_shell.rs`](../../../../../web/studio/foundry/studio/core/src/bin/stamp_shell.rs#L677-L691) wrote all `tar_bytes` synchronously to `gzip`'s `stdin` before calling `gz.wait_with_output()`, overflowing the 64 KB OS pipe buffer and deadlocking `gzip`. Resolved using `std::thread::scope` to stream `tar_bytes` concurrently while draining `gz.wait_with_output()`.
- Deep recursive JSON/LexLean emission in `verdant_studio.rs` required bumping the `wasm32-unknown-unknown` link stack size in [`build_foundry_studio.sh`](../../../../../web/studio/build_foundry_studio.sh) (`RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-zstack-size=8388608"`).
- Existing integration tests `web/studio/foundry/studio/core/tests/{corpus,publish,inference}.rs` reference uncommitted `../model/*.json` fixtures not present in clean git worktrees (`model/corpus.json`, `model/publish.json`, `model/objects/publish/hello.holo`, `model/inference.json`), so Cargo tests ran with `--lib --test verdant_studio` rather than bare `--tests`.
- R1 fresh-context `pro` exact-head adversarial review on PR `#4259` (`5fc63a75f`) reported 3 `MAJOR` and 8 `MINOR` findings:
  1. `MAJOR-1`: `verdant_driver.rs` returned `Event::C2(Err(_))` for read/list/stat/canonical/spawn/wait errors and traversal. Fixed by returning operation-specific `Event::C1`/`C3`/`C4`/`C5`/`C6`/`C7(Err)` variants and aborting immediately via `HandleOutcome::DriverError` on any `normalize_path` violation.
  2. `MAJOR-2`: Unbounded CLI args and deeply nested JSON/XML caused stack overflows (`RangeError`) inside `core.wasm`. Fixed with arg (`<= 32`, `<= 512` B), path (`<= 512` B, `<= 32` segments), workspace (`<= 256` files, `<= 64 KiB`/file, `<= 512 KiB` total), and structural complexity bounds, plus `try/catch` and Verdant prefix filtering in `host.js` `runVerdantPlanInCore`.
  3. `MAJOR-3`: `stage-admissions`, `stage-decisions`, `stage-mappings`, and `stage-matrix` with `inline_input` failed without `.jsonl`-suffixed virtual files. Fixed via op-specific reserved virtual filenames (`virtual_input_path_for`: `.verdant-input.jsonl`/`.json`/`.csv`/`.xml`/`.xsd`) and caller collision rejection (`MINOR-3`).
  4. `MINOR-1..8`: Aligned `normalize_path` with `cleanWorkspacePath` (`<= 0x1f`, `0x7f`, `:`), rejected non-string `args`, enforced `VirtualFs` file-vs-dir exclusivity and nested dir relocation, returned explicit `(status, stdout, stderr)` instead of `\x00` splitting, checked `gz_out.status.success()` before `write_res` in `stamp_shell.rs`, omitted `written_files` content from `host.js` `verdant_plan` tool returns, and noted `tests/inference.rs` above.
- R2 fresh-context `pro` exact-head adversarial review on `38bd8ce07` verified all R1 `MAJOR-1`, `MAJOR-3`, `MINOR-1..8` fixes and reported 2 `MAJOR` and 4 `MINOR` remaining findings:
  1. `MAJOR-1`: Unbounded line counts (`\n` x `10,000` in JSONL or `model/authorities.toml`), unbounded XML attributes (`f330`), and UTF-8 BOM `\u{feff}` bypassing `is_xml_like` still caused `RangeError`/`RuntimeError` in `core.wasm`. Fixed by stripping leading UTF-8 BOM before format detection, checking `MAX_LINES = 512` across non-CSV text files (including `.toml` and `.lex.tex`) and `MAX_CSV_LINES = 1024`, and bounding XML attributes (`MAX_XML_ATTRS = 256`).
  2. `MAJOR-2`: `validate_content_complexity` rejected valid `ingest-index` CSV files >= 51 rows. Fixed by scoping JSON `,`/`{`/`[` checks to `is_json_like`, verified with a 65-row CSV.
  3. `MINOR-1..4`: Added `stage-mappings`, `stage-matrix`, and `merge-gate` to `VERDANT_PLAN_OPS`, added pre-mutation dir-merge child conflict checks in `files_rename`, normalized `/workspace/` and `./` in `referencedPaths`, and removed duplicate `deleted_files` from `host.js` returns.
- R3 fresh-context `pro` exact-head adversarial review on `c420c5be3` verified all R2 fixes and reported 2 `MAJOR` and 2 `MINOR` findings:
  1. `MAJOR-1`: `is_json_like` omitted `!is_toml`, misclassifying `model/authorities.toml` (`[[authority]]`) as JSON. Fixed by adding `!is_toml` to the fallback branch of `is_json_like`, verified against 60 `[[authority]]` TOML blocks on live `core.wasm`.
  2. `MAJOR-2`: `MAX_STRUCTURAL_ITEMS = 400` allowed `admit` with >= 67 requirements to emit a single-line `model/requirements.json` that then failed `check` on reload. Fixed by separating `MAX_JSON_STRUCT_ITEMS = 1_000` (from `MAX_XML_TAGS = 400`) and applying the exact `should_validate_complexity` check inside `VirtualFs::files_write` (`HandleOutcome::DriverError` if exceeded), guaranteeing `admit` can never emit an unreloadable register (`70`-req roundtrip verified in Cargo + live `core.wasm`).
  3. `MINOR-1..2`: `host.js` `verdant_plan` now passes `{ path }[]` objects for `deletes` to `applyPrismTags` matching `canonical-generation.mjs`, and `normalizeVerdantArgPath()` collapses interior `./` and `//` segments cleanly.

## What changed

- **Lean-certified `afflom/verdant-studio` backend vendored into `core.wasm`:**
  - [`web/studio/foundry/studio/core/src/verdant_studio.rs`](../../../../../web/studio/foundry/studio/core/src/verdant_studio.rs) (`852,859` B, SHA-256 `1642b5877447becf3d28ab4f5af119e7100f9690b85dffc6895fda80a76ce55f`) and [`verdant_provenance.json`](../../../../../web/studio/foundry/studio/core/src/verdant_provenance.json) (`792` B, SHA-256 `c13d8b78c7d3a6d9aa617b1bc93e665fe93f86221cdd81e77e87f9e92157aea0`): Imported verbatim from `afflom/verdant-studio` (`MIT OR Apache-2.0`, `#![forbid(unsafe_code)]`, `core`/`std` only, zero floats, zero external dependencies, zero edits to `Cargo.toml`/`Cargo.lock`).
  - [`web/studio/foundry/studio/core/src/verdant_driver.rs`](../../../../../web/studio/foundry/studio/core/src/verdant_driver.rs) (`42,137` B, SHA-256 `b69bea7720456ed57a0488ef3c8474af721193172627f542c0eb5f3c8b553d4e`), [`lib.rs`](../../../../../web/studio/foundry/studio/core/src/lib.rs), [`abi.rs`](../../../../../web/studio/foundry/studio/core/src/abi.rs): Implemented `#![forbid(unsafe_code)]` isolated virtual-FS driver (`run_from_json`) over `BTreeMap` + `BTreeSet` with traversal/NUL/`:`/control refusal, typed `Event` error variants, symmetric read/write BOM-aware JSON/XML/TOML/LexTex/CSV bounds (`MAX_JSON_STRUCT_ITEMS = 1_000`, `MAX_XML_TAGS = 400`, `MAX_XML_ATTRS = 256`, `MAX_LINES = 512`), minimal register seeding (`app/lexlean.toml`, `model/{bindings,encodings,mappings,matrix,queue}`, `app/src/Catalog`), op-typed virtual `inline_input`, and `abi.rs` `op: "verdant-studio"` routing.
  - [`web/studio/foundry/studio/core/src/bin/stamp_shell.rs`](../../../../../web/studio/foundry/studio/core/src/bin/stamp_shell.rs) & [`web/studio/build_foundry_studio.sh`](../../../../../web/studio/build_foundry_studio.sh): Fixed `cmd_provenance` `gzip` pipe deadlock with `std::thread::scope` and configured 8 MiB WASM stack size (`-zstack-size=8388608`).
- **Studio `verdant_plan` tool wiring & curated standards index:**
  - [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs) & [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js): Added `verdant_plan(op, input?, args?)` to `FOUNDRY_CHAT_TOOL_SYSTEM_PROMPT`, `FOUNDRY_SYSTEM_PROMPT`, `TOOL_INTENT_RE`, `FOUNDRY_TOOL_NAMES`, and `BARE_TOOL_PROMPT_RE`; wired `VERDANT_PLAN_OPS`, `normalizeVerdantArgPath()`, Verdant workspace filtering + `try/catch` error guard in `runVerdantPlanInCore()`, `{ path }[]` delete patches and concise `written_paths`/`deleted_paths` tool responses, and `window.__uorR4.verdantPlan`; kept `FOUNDRY_CHAT_SYSTEM_PROMPT` completely tool-free and left IPC `command-registry` (`368/179/310/306`) unchanged.
  - [`scripts/studio/build-standards-index.mjs`](../../../../../scripts/studio/build-standards-index.mjs): Updated `foundry-tools-and-holo` (`web_search`, `verdant_plan`) and added 3 curated sections (`verdant-studio-certified-api`, `verdant-studio-admission-and-strata`, `verdant-studio-game-and-web-standards-catalog`).
- **Tests:**
  - [`web/studio/foundry/studio/core/tests/verdant_studio.rs`](../../../../../web/studio/foundry/studio/core/tests/verdant_studio.rs): Added 6 Cargo tests covering provenance SHA-256 verification, virtual-FS workflows, traversal refusal, and R1/R2/R3 regressions (`60` `[[authority]]` TOML check, `70`-requirement `admit` -> `check` roundtrip, newline/XML attr/BOM bounds, 65-row CSV `ingest-index`, SKOS `stage-mappings`).
  - [`web/studio/foundry/studio/tools/studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs): Added 3 Stage A2 tests covering `buildStandardsIndex` sections & `< 110 KB` budget, `verdant_plan` routing + slim tool response + `{ path }` delete patch contract + tool-free Home Chat preservation, and live `WebAssembly.instantiate` execution of `core.wasm` `holo_run` across happy paths and R1/R2/R3 regressions.

## Measurements

- **`core.wasm`:** `967,730` B (SHA-256 `8a22da7e70d8a4d0560ead0311e08b2c178252fc2705748b44a445b63b552b16`, well below `2.5 MB`).
- **`standards-index.json`:** `55,271` B (`35` sections, SHA-256 `64d32149191677e9217381b1c9dca08ccefe74dc473662e4ffd5b79c8a50ff56`).
- **`host.bundle.js`:** `1,057,484` B (`103` inputs, SHA-256 `2e42f5f82129f13dc414aa49a15513e95a77217c853ffdf92d708e0285269e4a`).
- **`stamp-shell provenance` & bundle:** `483` closure files (`486` total files staged into `web/studio/dist/`), closure `edcb2e61ab1ba73984d68c9d0b4fb7210a648b2b446d3376d19c762b1efd7095`.
- **WASM zero-matmul & zero-float audit (`uor_r4_integer_bg.wasm`, `1,289,272` B, SHA-256 `1f018be5955c034d04ecbcccc28388bca7f310b6add9a84293b448f40f3236ac`):** `ok 1..7`, **`float=0`, `unclassified=0`** on all 7 entrypoints (`loadStack`, `step`, `saveSession`, `restoreSession`, `weightsPerToken`, `stack::IntegerStackSession::step`, `stack::stack_argmax`).
- **Foundry Studio CDP E2E (`verify_foundry_studio.mjs web/studio/dist`):**
  - Root `/`: Home Chat Turn 1 (`34` chars) & Turn 2 (`126` chars), Build App tool round-trip (`app=2`), Linux `6.18.7` VM, GitHub/Stitch, Custom AI API Endpoint, `Qwen3-1.7B` mount `40,221 ms` (`opfs`), `TTFT=2,017 ms`, `7.87 tok/s`, `total=3,027 ms`.
  - Subpath `/uor-r4/studio/`: Home Chat Turn 1 (`34` chars) & Turn 2 (`126` chars), Build App tool round-trip (`app=2`), Linux `6.18.7` VM, GitHub/Stitch, Custom AI API Endpoint, `Qwen3-1.7B` mount `39,019 ms` (`opfs`), `TTFT=2,041 ms`, `7.77 tok/s`, `total=3,050 ms`.
- **GPU spend:** `$0.00` (`Running: 0 pod(s), $0/h of $8.00/h`).

## Checks run at the head

- `cargo test --locked --manifest-path web/studio/foundry/studio/core/Cargo.toml --lib --test verdant_studio`: **`6 passed; 0 failed`**.
- `bash web/studio/build_foundry_studio.sh`: **PASS** (`486` staged files, `core.wasm` `967,730` B).
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)`: **PASS** (`adds no type error to upstream's own (28)`).
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` & `shell_parse.mjs`: **PASS**.
- `node --test web/studio/relay/relay.test.mjs web/studio/foundry/studio/tools/{prism_r4_route,command_ownership_registry,canonical_generation_ownership,preview_permissions,custom_api_endpoint,generation_patch,studio_phase1_3,studio_phase4a,stage3_os_github_stitch}.test.mjs`: **`65/65` PASS, `0` failed, `0` skipped** (2 suites).
- `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/foundry/studio/tools/command_ownership_registry_browser.mjs`: **PASS** (`commands: 368`, `handlers: 179`, `inputs: 310`, `outputs: 306`, `pageExceptions: 0`).
- `node web/studio/verify_studio_worker.mjs`: **PASS** (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`, `saveRestoreBitIdentical: true`).
- `bash scripts/studio/runtime-files.sh --verify web/studio/dist --typed`: **PASS** (`483` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap`: **PASS** (`ok 1..7`, `float=0`).
- `CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/verify_foundry_studio.mjs web/studio/dist`: **PASS** (`ALL FOUNDRY STUDIO CDP CHECKS PASSED`).

## Skills used

- None outside standard harness and Studio verification scripts.

## Still open

- Post-fix `uor-harness` checks (`check --staged`, `ratchet`, `claim-wording`, `loop report --check`, `loop pr-check`), fresh-context `pro` R4 exact-head adversarial review (`APPROVE`), `[antigravity] uor-review v1`, CI + merge queue, and live `--typed` `gh-pages` publication (`$0.00` GPU spend).

## Plain summary

`antigravity-37` completes Stage A2 (`afflom/verdant-studio` integration): vendors the Lean-certified `verdant_studio.rs` planning backend (`852,859` B, `#![forbid(unsafe_code)]`, zero floats/external dependencies) into `core.wasm` behind a bounded, isolated virtual-FS driver (`op: "verdant-studio"`), fixes a pre-existing 64 KB `gzip` pipe deadlock in `stamp_shell.rs`, exposes `verdant_plan` in Studio chat/build modes without altering tool-free Home Chat or the IPC command registry, and adds 3 curated Verdant Studio sections to `standards-index.json` (`55,271` B, `35` sections). All R1, R2, and R3 adversarial review findings are resolved and verified across Cargo (`6/6`), Node (`65/65`), worker, WASM `float=0` audit, command-registry browser, and dual-origin Chrome CDP checks at `$0.00` GPU spend.
