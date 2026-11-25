# REPORT: studio-lexlean-mvm-workbench, antigravity-35

Lab: `antigravity`. Order: [`ORDER.md`](ORDER.md). Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7). Base: `a4f55eb07` (`ORDER.md` committed alone first at `0c7512ef3`). GPU spend: `$0.00`.

## What did not work

- During initial development, adding `FOUNDRY_CHAT_TOOL_SYSTEM_PROMPT` and gating `stopSequences` temporarily failed `custom_api_endpoint.test.mjs:192` (`'TRIGGER_TOOL list root files'`) because `TOOL_INTENT_RE` required `list` immediately adjacent to `files` (`list root files` missed) and Build mode dropped `['</tool_call>']`; resolved by accepting `(?:root|workspace|project)\s+` in `TOOL_INTENT_RE` and always keeping `stopSequences: ['</tool_call>']` in `build`/`local-agent` modes (`48/48` PASS afterward).
- The first chained Chrome CDP run (`task-15512`) hit a one-off `uor-r4:select-model` timeout/failure at step 2; re-running unmodified `verify_foundry_studio.mjs` (`task-15522`) passed both `/` and `/uor-r4/studio/` completely (`ALL FOUNDRY STUDIO CDP CHECKS PASSED`).

## What changed

- [`r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs):
  - Replaced `FOUNDRY_CHAT_SYSTEM_PROMPT` with a clean natural-language chat prompt (zero `<tool_call>` syntax and zero tool schema list) so `qwen3-0.6b` answers conversational and explanatory questions directly in natural English rather than emitting `{"name": "hello", ...}` / `{"name": "predict_next_token", ...}`.
  - Added `FOUNDRY_CHAT_TOOL_SYSTEM_PROMPT` (selected only when `TOOL_INTENT_RE` matches an explicit tool request in `chat`/`ask` mode).
  - Updated `runUorR4AgentTurn` to pass `stopSequences: []` for casual chat (preserving `['</tool_call>']` for Build/Agent and tool-intent chat turns).
  - Exported `sanitizeCasualChatHistory(messages)` to unwrap legacy bare-JSON assistant turns (`arguments.message|text|reply|response`) or drop poisoned scaffolding turns during casual chat so stale IndexedDB histories do not perpetuate JSON imitation.
- [`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs):
  - Honored explicit empty `stopSequences` (`Array.isArray(payload.stopSequences) ? payload.stopSequences : ['</tool_call>']`).
- [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs) & [`verify_foundry_studio.mjs`](../../../../../web/studio/verify_foundry_studio.mjs):
  - Added unit coverage for clean chat prompt isolation, empty `stopSequences`, explicit chat tool routing, and `sanitizeCasualChatHistory()`, plus dual-origin Chrome CDP multi-turn verification (`'hello'` -> `'how do you predict the next token'`) and updated prewarm token threshold (`>= 40`, clean chat system prompt prewarms `92` tokens).

## Measurements

- **Prewarm & 0-LSB parity (`verify_foundry_studio.mjs`):** `prewarmedTokens: 92`, `exactGen === specGen` (`float_ops=0`, `dense_matmul_ops=0`) on both `/` and `/uor-r4/studio/`.
- **Real Home Chat `qwen3-0.6b` WASM replies on `/` & `/uor-r4/studio/`:**
  - Turn 1 (`'hello'`): `"Hello! How can I assist you today?"` (`34` chars).
  - Turn 2 (`'how do you predict the next token'`): `"I predict the next token using the UOR-R4 architecture, which operates on fixed-point arithmetic and exact integer operations."` (`126` chars).
- **`Qwen3-1.7B` CDP on `/`:** mount `37,894 ms` (`opfs`), `TTFT=2,092 ms`, `7.72 tok/s`, `total=3,108 ms`.
- **`Qwen3-1.7B` CDP on `/uor-r4/studio/`:** mount `36,365 ms` (`opfs`), `TTFT=2,473 ms`, `6.1 tok/s`, `total=3,775 ms`.
- **GPU spend:** `$0.00` (`Running: 0 pod(s), $0/h`).

## Checks run at the head

- `node --test web/studio/foundry/studio/tools/{prism_r4_route,command_ownership_registry,canonical_generation_ownership,preview_permissions,custom_api_endpoint,generation_patch,studio_phase1_3,studio_phase4a,stage3_os_github_stitch}.test.mjs`: **`48/48` PASS, `0` failed**.
- `bash web/studio/build_foundry_studio.sh`: **PASS**.
- `(cd web/studio/foundry/studio/vendor/prism && node hologram.typecheck.mjs)`: **PASS** (`adds no type error to upstream's own (28)`).
- `node web/studio/foundry/studio/tools/host_bundle.mjs --check` & `node web/studio/foundry/studio/tools/shell_parse.mjs`: **PASS**.
- `CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/foundry/studio/tools/command_ownership_registry_browser.mjs`: **PASS** (`commands: 368`, `handlers: 179`, `inputs: 310`, `outputs: 306`, `pageExceptions: 0`).
- `node web/studio/verify_studio_worker.mjs`: **PASS** (`ALL STUDIO WORKER CHECKS PASSED`, `remoteRequests: 0`, `saveRestoreBitIdentical: true`).
- `bash scripts/studio/runtime-files.sh --verify web/studio/dist --typed`: **PASS** (`482` closure files verified).
- `python3 scripts/audit_zero_matmul_wasm.py web/studio/pkg/uor_r4_integer_bg.wasm --serving --call-graph --tap`: **PASS** (`ok 1..7`, `float=0`, `sha256: 414179c08df88d38fa12f26c40afbb8c86e0c18d1bbc52d64f3471a7ba78161c`).
- `CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node web/studio/verify_foundry_studio.mjs web/studio/dist`: **PASS** (`ALL FOUNDRY STUDIO CDP CHECKS PASSED` across `/` and `/uor-r4/studio/`).

## Skills used

- None outside standard harness and Studio verification scripts.

## Still open

- Fresh-context `pro` exact-head adversarial review, `[antigravity] uor-review v1`, CI + merge queue, and live `gh-pages` publication (`$0.00` GPU spend).

## Plain summary

`antigravity-35` removes tool-calling XML and all 18 tool schemas from the default casual Home Chat prompt (`FOUNDRY_CHAT_SYSTEM_PROMPT`), routing explicit tool requests in chat to `FOUNDRY_CHAT_TOOL_SYSTEM_PROMPT`, honoring empty `stopSequences` in `uor-v13-worker.mjs`, and sanitizing stale bare-JSON assistant turns via `sanitizeCasualChatHistory()`. Real `qwen3-0.6b` Chrome CDP multi-turn chat on both `/` and `/uor-r4/studio/` now replies in natural English (`"Hello! How can I assist you today?"` -> `"I predict the next token using the UOR-R4 architecture, which operates on fixed-point arithmetic and exact integer operations."`) while preserving all Build tool round-trips (`48/48` unit tests and full dual-origin CDP PASS, `$0.00` GPU spend).
