# ORDER: studio-lexlean-mvm-workbench, antigravity-35

Lab: antigravity. Item: [`studio-lexlean-mvm-workbench`](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (Issue `#3939`, M7).

## Task (Natural Language Home Chat Prompt & JSON Scaffolding Guard)

Eliminate bare JSON tool-call scaffolding responses (`{"name": "hello", ...}`, `{"name": "predict_next_token", ...}`) from UOR-R4 Qwen3-0.6B/1.7B casual Home Chat turns while preserving explicit tool calling:

1. Replace [`FOUNDRY_CHAT_SYSTEM_PROMPT`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs) with a clean natural-language chat prompt (zero `<tool_call>` syntax and zero tool schema list), moving tool schemas to `FOUNDRY_CHAT_TOOL_SYSTEM_PROMPT` selected only when `TOOL_INTENT_RE` matches an explicit tool request in `chat`/`ask` mode.
2. Honor explicit `stopSequences: []` in [`uor-v13-worker.mjs`](../../../../../web/studio/foundry/studio/shell/uor-v13-worker.mjs) (pass `Array.isArray(payload.stopSequences) ? payload.stopSequences : ['</tool_call>']`) and pass `stopSequences: wantsToolPrompt ? ['</tool_call>'] : []` from `runUorR4AgentTurn`.
3. Add `sanitizeCasualChatHistory(messages)` in [`r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs) so stale pre-fix hallucinated bare-JSON assistant turns in IndexedDB history unwrap or drop cleanly during casual chat rather than chaining subsequent replies into JSON mode.
4. Add unit and multi-turn real-WASM Chrome CDP assertions in [`studio_phase4a.test.mjs`](../../../../../web/studio/foundry/studio/tools/studio_phase4a.test.mjs) and [`verify_foundry_studio.mjs`](../../../../../web/studio/verify_foundry_studio.mjs), verify build/tests/CDP/`gh-pages` live publication (`$0.00` GPU spend).

## Cancel probe

- Any edit touches `crates/**`, `src-tauri/**`, `Cargo.lock`, `CHARTER.md`, `DECISIONS.md`, or `docs/harness/native-space.json`.
- Any existing Studio tool round-trip (`workspace_list`, `github`, `mvm`, `linux`, `holo`, `lexlean`) regresses.

## Scope

Writes strictly inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-35/**`, `web/studio/foundry/studio/shell/r4-agent-loop.mjs`, `web/studio/foundry/studio/shell/uor-v13-worker.mjs`, `web/studio/foundry/studio/tools/studio_phase4a.test.mjs`, and `web/studio/verify_foundry_studio.mjs` (plus rebuilt gitignored `web/studio/dist/**`).

## Done check

`bash web/studio/build_foundry_studio.sh`, `hologram.typecheck.mjs`, `host_bundle.mjs --check`, `shell_parse.mjs`, 9 Foundry suites (`node --test`), `command_ownership_registry_browser.mjs`, `verify_studio_worker.mjs`, `runtime-files.sh --verify web/studio/dist --typed`, `audit_zero_matmul_wasm.py`, dual-origin Chrome CDP `verify_foundry_studio.mjs` (verifying `'hello'` -> `'how do you predict the next token'` natural English replies in real Home Chat), `uor-harness` checks/ratchets/`loop pr-check`, fresh `pro` exact-head review, merge, and live `gh-pages` Studio deploy PASS at `$0.00` GPU spend.
