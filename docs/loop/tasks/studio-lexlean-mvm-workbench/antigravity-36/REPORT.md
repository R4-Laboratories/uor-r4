# REPORT: studio-lexlean-mvm-workbench, antigravity-36

Lab: `antigravity`. Order: [`ORDER.md`](ORDER.md). Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7). Base: `e01297aa6` (`ORDER.md` committed alone first at `0b472e376`). GPU spend: `$0.00`.

## What did not work

- Initial Gap 4 draft added `canonical_anchors` / `anchor_len` directly onto `PointKeyImage` (which has 34 existing struct literals in `crates/uor-r4-cpu-train`, outside `ORDER.md` scope). Reverted `PointKeyImage` to its original 4-field layout (`width`, `points`, `targets`, `vote_k`) and introduced `PointKeyAnchors` + `add_point_key_with_anchors_uorlut01` + `ServedTurnMemory::new_with_point_anchors`, keeping existing callers and `/8` artifacts without anchors byte-identical.
- Initial `studio_phase4a.test.mjs` Stage A1 additions failed 3 assertions on the first run (`normalizeRelayEndpoint('')` returns `null` rather than throwing, `FOUNDRY_TOOL_NAMES` was missing from imports, `runUorR4AgentTurn` uses `userPrompt`/`generateFn`, and `prism-code/src/vendor/verify.mjs` is async and expects an 8-row Holo/1 section table while synchronous `parseArchive` checks footer digests directly); corrected the test harness calls.
- R1 fresh `pro` adversarial exact-head review (`0b2e7d6bf`) raised 2 `MAJOR` findings:
  1. Untrusted `web_search` / tool outputs inside `<tool_response>` could smuggle `<prism-write>` / `<prism-spec>` tags into Build-mode `parsePrismTags` / `extractFallbackCodeWrites` / `afterTurn` (`host.js`), and `htmlToReadableExcerpt` decoded entities after `<[^>]+>` stripping (`worker.mjs`). Fixed with defense-in-depth across `host.js` (strip `<tool_response>`/`<tool_call>` blocks inside `parsePrismTags`, `extractFallbackCodeWrites`, and `specRawTurn` before `afterTurn`), `r4-agent-loop.mjs` (`\u003c`/`\u003e` escaping in `<tool_response>` JSON envelopes), and `worker.mjs` (strip entity-decoded `<[^>]*>` & residual `[<>]` in relay HTML excerpts and search results), plus added `web_search` to `BARE_TOOL_PROMPT_RE`.
  2. `R4-60` off-origin hydration consent allowed partial-cache bypass (`hasOfflineV13Cache` lacked `pack_seal.json`), origin-only consent persisted pre-load in `localStorage` (`qwen3-0.6b` authorized `1.7b`), and `stitch_manifest_url` / `testStitchRecover` were unchecked. Fixed in `uor-r4-runtime.mjs` by requiring `pack_seal.json` in `hasOfflineV13Cache`, scoping consent per `${origin}::${modelId}` and writing only after `load_v13` succeeds, checking both `manifest_url` & `stitch_manifest_url`, gating `testStitchRecover`, and clearing stale progress fields on refusal. Also validated `point_anchors()` inside `memory_image()` and cleared the `ServedTurnMemory::new` dead-code warning in `memory.rs`.
- Post-fix `task-16032` CDP hit a transient cold `/uor-r4/studio/` `uor-r4:select-model` error; clean rerun `task-16048` passed both `/` and `/uor-r4/studio/` end-to-end (`ALL FOUNDRY STUDIO CDP CHECKS PASSED`).
- Cross-repo audit of `Casey-allard/lexlean-r4`:
  - `wasm-v13-foundry` branch HEAD (`0b5b6d30a`) already matches `foundry-assets.lock.json` (`uor_foundry_v13_wasm`); no binary crate bump needed.
  - `native_q31_lane.rs` (`UORQ31V1`) and `sparse_basin_v15.rs` (`UORR4V15` 64-basin router) remain non-default experimental formats in `lexlean-r4/main` rather than replacing UOR-R4 M1/D85 geometry in `crates/uor-r4-integer`; left unimported per scope.
  - `holo.js` (`R4-54`) already delegates `.holo` verification to `core.run({ op: "validate-application" })` (`HoloLoader::from_bytes` + `validate_application`), backed by `prism-code/src/archive.mjs` trailing 32-byte BLAKE3 footer verification; added explicit regression coverage rather than modifying `holo.js`.

## What changed

- **Core Gap 4 Paraphrase Continuation Handshake (`/8 -> /7` canonical reply anchors):**
  - [`crates/uor-r4-integer/src/stack/format.rs`](../../../../../crates/uor-r4-integer/src/stack/format.rs): Added optional `#[serde(default, skip_serializing_if = "Option::is_none")] pub anchor_len: Option<usize>` on `StackPointKey` (refusing `Some(0)` in `check()`) and added `add_point_key_with_anchors_uorlut01()` writing `memory.point.anchors` when `PointKeyAnchors` is provided (`add_point_key_uorlut01` delegates with `None`).
  - [`crates/uor-r4-integer/src/stack/point_key.rs`](../../../../../crates/uor-r4-integer/src/stack/point_key.rs): Added `PointKeyAnchors`, `point_anchors()`, `ServedPointKey::with_anchors()`, `name_with_row()`, and `canonical_reply_anchor()`.
  - [`crates/uor-r4-integer/src/stack/memory.rs`](../../../../../crates/uor-r4-integer/src/stack/memory.rs), [`session.rs`](../../../../../crates/uor-r4-integer/src/stack/session.rs), [`mod.rs`](../../../../../crates/uor-r4-integer/src/stack/mod.rs): Exported `PointKeyAnchors` & `add_point_key_with_anchors_uorlut01`, validated `point_anchors()` in `memory_image()`, wired `point_anchors()` in `IntegerStackModel::parse`, and updated both `point_miss` paths in `stack_step_fbs` / `stack_step_fbs_timed` to copy winning canonical reply anchors (trimmed before `MEMORY_PAD`) into `session.memory.anchors.reply`.
  - [`crates/uor-r4-integer/src/stack/tests.rs`](../../../../../crates/uor-r4-integer/src/stack/tests.rs): Added `canonical_reply_anchors_bridge_point_miss_to_reply_table_and_reject_bad_table`.
- **SpiralCore Stateless Edge Relay (`R4-39`) & Studio Tooling Isolation:**
  - [`web/studio/relay/`](../../../../../web/studio/relay/): Imported MIT/Apache-2.0 zero-dependency relay files (`worker.mjs`, `local-relay.mjs`, `relay.test.mjs`, `README.md`) from `Casey-allard/lexlean-r4/relay/` and sanitized HTML/entity-decoded tags + residual `[<>]` in search/excerpt outputs.
  - [`web/studio/foundry/studio/shell/spiralcore-relay.mjs`](../../../../../web/studio/foundry/studio/shell/spiralcore-relay.mjs): Added opt-in SPVS browser client (`normalizeRelayEndpoint`, `createSpiralCoreRelayClient`), making zero network calls on boot and enforcing `wss://`/`https://` (plain `ws://`/`http://` restricted to loopback). Registered in [`shell_parse.mjs`](../../../../../web/studio/foundry/studio/tools/shell_parse.mjs).
  - [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs) & [`host.js`](../../../../../web/studio/foundry/studio/shell/host.js): Added `web_search` tool schema/routing & `spiralCoreRelay` wiring without altering IPC `command-registry` (`368/179/310/306`), escaped `<`/`>` inside `<tool_response>` JSON envelopes, and stripped `<tool_response>`/`<tool_call>` envelopes before `parsePrismTags`, `extractFallbackCodeWrites`, and `afterTurn`.
- **`R4-54` `.holo` & `R4-60` Off-Origin Hydration Consent Gate:**
  - [`web/studio/foundry/studio/shell/uor-r4-runtime.mjs`](../../../../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs): Enforced per-model `${origin}::${modelId}` off-origin download consent persisted only after `load_v13` succeeds, `pack_seal.json` completeness check in `hasOfflineV13Cache`, `stitch_manifest_url` + `testStitchRecover` gating, and `hydrationProvenance` reporting.
  - [`web/studio/foundry/studio/tools/studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs): Added unit coverage for SpiralCore relay + `web_search`, `<tool_response>` workspace-write injection defense, `R4-60` per-model consent gating/provenance, and `R4-54` `.holo` tampered BLAKE3 footer rejection.

## Measurements

- **WASM zero-matmul & zero-float audit (`audit_zero_matmul_wasm.py --serving --call-graph --tap`):** `ok 1..7` PASS, **`float=0`, `unclassified=0`** on all 7 entrypoints (`loadStack`, `step`, `saveSession`, `restoreSession`, `weightsPerToken`, `stack::IntegerStackSession::step`, `stack::stack_argmax`, `uor_r4_integer_bg.wasm`, `1,289,272` bytes, sha256 `1f018be5955c034d04ecbcccc28388bca7f310b6add9a84293b448f40f3236ac`).
- **Foundry Studio CDP E2E (`verify_foundry_studio.mjs web/studio/dist`, `task-16048`):**
  - Root `/`: `Qwen3-0.6B` Home Chat Turn 1 (`"Hello! How can I assist you today?"`, `34` chars) & Turn 2 (`126` chars), Build App tool round-trip (`app=2`), Linux `6.18.7` VM, GitHub/Stitch, Custom API Endpoint, `Qwen3-1.7B` mount `37,269 ms` (`opfs`), `TTFT=2,051 ms`, `7.81 tok/s`, `total=3,064 ms`.
  - Subpath `/uor-r4/studio/`: `Qwen3-0.6B` Home Chat Turn 1 (`"Hello! How can I assist you today?"`, `34` chars) & Turn 2 (`126` chars), Build App tool round-trip (`app=2`), `Qwen3-1.7B` mount `37,796 ms` (`opfs`), `TTFT=2,038 ms`, `7.8 tok/s`, `total=3,054 ms`.
- **GPU spend:** `$0.00` (`Running: 0 pod(s), $0/h`).

## Checks run at the head

- `cargo test --locked -p uor-r4-integer --lib`: **`222 passed; 0 failed`**.
- `cargo test --locked -p uor-r4-integer --features serving-wasm --lib`: **`734 passed; 0 failed; 1 ignored`** (includes `canonical_reply_anchors_bridge_point_miss_to_reply_table_and_reject_bad_table` PASS).
- `cargo check --locked -p uor-r4-cpu-train --tests`: **PASS** (zero edits to `uor-r4-cpu-train`).
- `node --test web/studio/relay/relay.test.mjs web/studio/foundry/studio/tools/{prism_r4_route,command_ownership_registry,canonical_generation_ownership,preview_permissions,custom_api_endpoint,generation_patch,studio_phase1_3,studio_phase4a,stage3_os_github_stitch}.test.mjs`: **`62/62` PASS, `0` failed**.
- `bash web/studio/build_foundry_studio.sh` & `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)` (`adds no type error to upstream's own (28)`): **PASS**.
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` & `shell_parse.mjs`: **PASS**.
- `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/foundry/studio/tools/command_ownership_registry_browser.mjs`: **PASS** (`commands: 368`, `handlers: 179`, `inputs: 310`, `outputs: 306`, `pageExceptions: 0`).
- `node web/studio/verify_studio_worker.mjs`: **PASS** (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`, `saveRestoreBitIdentical: true`).
- `bash scripts/studio/runtime-files.sh --verify web/studio/dist --typed`: **PASS** (`483` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap`: **PASS** (`ok 1..7`, `float=0`).
- `CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/verify_foundry_studio.mjs web/studio/dist`: **PASS** (`ALL FOUNDRY STUDIO CDP CHECKS PASSED`).

## Skills used

- None outside standard harness and Studio verification scripts.

## Still open

- Post-fix `uor-harness` checks (`check --staged`, `ratchet`, `claim-wording`, `loop report --check`, `loop pr-check`), fresh-context `pro` R2 exact-head adversarial review (`APPROVE`), `[antigravity] uor-review v1`, CI + merge queue, and live `--typed` `gh-pages` publication (`$0.00` GPU spend).

## Plain summary

`antigravity-36` completes Stage A1 (`Casey-allard/lexlean-r4` sync & audit): ports the Gap 4 `/8 -> /7` canonical reply anchor handshake (`PointKeyAnchors`, `memory.point.anchors`, `add_point_key_with_anchors_uorlut01`) without breaking `uor-r4-cpu-train`, imports and integrates the opt-in SpiralCore stateless edge relay + `web_search` tool (`R4-39`) with `<tool_response>` workspace-write isolation across `host.js`, `r4-agent-loop.mjs`, and `worker.mjs`, adds `.holo` tampered BLAKE3 footer regression coverage (`R4-54`), and enforces per-model post-load off-origin weight download consent + hydration provenance (`R4-60`). All 734+222 Rust tests, 62 Node unit/relay tests, WASM `float=0` audit, worker/registry checks, and dual-origin Chrome CDP Studio E2E checks pass at `$0.00` GPU spend.
