# REPORT for Qwen3-0.6B Foundry Webharness LoRA v2 Pre-Registration (`antigravity-16`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `8849fd267` (PR #4213, Phase 4A; `ORDER.md` committed alone at `babe53cc3`).

## What did not work

- **Initial generator literal-tag and short-Chat catches caught at `$0.00` before any GPU spend:**
  - `eval_foundry_v13_v2.py --self-test` initially failed on `Train chat pool #4` (`forbidden_tag:<prism-write`) because several conceptual Foundry Chat descriptions embedded literal `<prism-write>`, `<tool_call>`, `<tool_response>`, and `<|im_start|>`/`<|im_end|>` substrings verbatim (the same pattern that caused Run 2 Bar #3 Chat tag leakage), and failed on `Train chat pool #88`/`#125` (`too_few_words:73`/`81`). Replaced all literal XML/ChatML tags in Chat text and keywords with prose names (`prism-write file blocks`, `tool-call JSON blocks`, `ChatML system prompt header`), enriched concise `[Standards]` / SQL / CSS / multi-turn Chat entries so every entry meets `>= 85` words, and enforced `unique_ratio >= 0.50` (`DATA_SEED = 20261123`).
  - Pre-review of `_gen_multi_turn_tool_chains` and `train_foundry_lora_v2.py` caught odd-length 3-message history tuples, a synthetic `[N]` collision fallback on `holo_publish` val prompts, non-uniform integer spacing on `export_fp32_parity_probe`, and unbuffered `f64.convert_i32_u` bridge line attribution. Rewrote multi-turn episodes into strict even `(user, assistant)` pairs (`2`-entry round-1 `<tool_response>` turns and `4`-entry follow-ups matching `r4-agent-loop.mjs` / `HoloV13Model::generate_chat_turn_with_history`), gave `holo_publish` distinct target phrases (`0` synthetic `[N]` suffixes), added explicit `model.generation_config` greedy reset + exact arithmetic 36-position interpolation, and buffered WAT `(func ...)` blocks in `run_wasm_float_audit`.
- **LoRA hyperparameter alignment before fitness review (`r = 32 -> 16`, `lora_alpha = 64 -> 32`, unsmoothed `val_loss`):** Initial draft `train_foundry_lora_v2.py` (`dc4e5c2b...`) used `lora_r = 32, lora_alpha = 64` and used label-smoothed loss during validation selection. Restored Run 5's proven GPTQ-safe defaults (`lora_r = 16`, `lora_alpha = 32`, train-only `label_smoothing = 0.05` vs unsmoothed `val_loss` selection) in `80c3db4f...` before locking `PREP.md` (`a57f0a8a64f2c07330f0c7d0bb1fcdc5d439099f`).
- **Non-blocking precision notes recorded from fitness reviewer (`532b3052-77b9-49f1-a8a8-4d4b304627d6`):**
  1. `ORDER.md` and `PREP.md` summarize Chat target length as `100–235` words, whereas `generate_foundry_sft_v2.py` checks `min_words = 85, max_words = 235` (generated word counts observed at `85–133` train / `85–120` val; Bar #3 evaluates at `>= 60` words and `>= 240` non-ws chars).
  2. `CHAT_TOPICS_VAL` evaluates `100` unique unseen prompts grouped across `20` disjoint domain topic answers × `5` unseen phrasings; the `<= 4.3 pp` CI half-width reflects per-prompt `N = 100` rather than topic-clustered `N = 20` variance.
- **No GPU pod launched in this PR:** `uor-pod status` verified `Running: 0 pod(s) ($0/h of $8.00/h)`; actual `d90-foundry-lora-v7` GPU execution waits for pre-run review after this pre-registration PR merges.

## What changed

- **Charter §4.5 Pre-Registration (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-16/`):**
  - [`ORDER.md`](ORDER.md): Task scope & cancel criteria strictly bounded to `antigravity-16/**`.
  - [`PREP.md`](PREP.md): Pinned `d90-foundry-lora-v7` pre-registration for `Qwen3-0.6B v2` (`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`), D68 Bars #1–#5 (including `Bar #4a` and `Bar #4b`), `<= $0.80 / 1.5h` cap (`<= $2.65` combined D90 spend within `$3.00`), and non-author `TEST FITNESS: FIT` signoff (`a57f0a8a64f2c07330f0c7d0bb1fcdc5d439099f` by `532b3052-77b9-49f1-a8a8-4d4b304627d6`).
  - Untracked local toolchain prepared & verified at `$0.00` under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/v2/` (0 repo binaries/scripts added).

## Measurements

- **Pinned local script SHA-256 digests (`local/antigravity/d90-foundry-lora/`):**
  - `v2/foundry_schemas_v2.json`: `56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, `18` tools byte-identical to [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs))
  - `v2/generate_foundry_sft_v2.py`: `011559b23ab9f4ac1f3498eee764924c76b9531bc95af16a10e86ad215902b68`
  - `v2/train_foundry_lora_v2.py`: `80c3db4f0cfc3b1fc54303dcd988067ec06f039930c76b2182b9f7d28f339827`
  - `v2/eval_foundry_v13_v2.py`: `de98ed2c19471c539921dc82f8e22ec7e89ffbb839606392753d461a3a3832ac`
  - `v2/foundry_v13_eval_pack_v2.rs`: `0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`
- **Deterministic dataset verification (`DATA_SEED = 20261123`):**
  - `train.jsonl` (`7,200` rows: `2,700` `tool_call`, `1,400` `prism_write`, `1,260` `multi_turn`, `1,840` `chat`): `ecc7d94955baf63c93e742364048e3f106cacf2da789b5e2c0977e5417bc67d8`
  - `val.jsonl` (`440` rows: `288` `tool_call`, `52` `prism_write`, `100` `chat`): `3119a5ebf8289c0dea5bd082e2c866c9ceeec5e8317ed6e9e40a977324ea7c9b` (`train_val_overlap_hashes: 0`)
  - `calib_foundry.txt` (`96` user-only rows from `train.jsonl`): `64bd52451695e0031dc04ced70261fe91e8ab421562c8349dfc059689ed24ff9`
  - `parity_probe_prompts.json` (`16` stratified `val.jsonl` episodes -> `576` positions): `ddff8616e20eb53f5f507a0c420045875387ae63c8ed74bd68cab96ac0c51cc8`
- **Pod ledger & balance:** `uor-pod status` shows `Running: 0 pod(s) ($0/h of $8.00/h)`; cumulative Runs 1–6 spend `~$1.85` (`$1.8481` verified in `uor-pod log`), leaving `$1.15` within `$3.00` D90 cap.

## Checks run at the head

- `python3 -m py_compile local/antigravity/d90-foundry-lora/v2/*.py` — PASS (`0` errors, re-verified on updated `train_foundry_lora_v2.py` `80c3db4f...`).
- `python3 local/antigravity/d90-foundry-lora/v2/eval_foundry_v13_v2.py --self-test` — PASS (`self_test: "PASS"`, `agent_loop_verified: true`, `tool_count: 18`, `train_val_overlap_hashes: 0`, `gold_eval: 340/340` tool/build + `100/100` Chat = `100.0%`, `bars_passed: true`, plus non-author re-verification inside `532b3052-77b9-49f1-a8a8-4d4b304627d6`).
- `eval_foundry_v13_v2.py --wasm local/antigravity/d90-foundry-lora/foundry-v13-wasm/pkg/uor_foundry_v13_wasm_bg.wasm --holo-src /Users/casey.allard/lexlean-r4/crates/wasm/src/holo_geometric_runtime.rs` — PASS (`holo_inference_functions_scanned: 307`, `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
- `cargo check --manifest-path /Users/casey.allard/lexlean-r4/Cargo.toml -p lexlean-r4-wasm --bin foundry_v13_eval_pack_v2` (via clean temporary copy removed afterward) — PASS (`Finished dev profile` with `0` warnings; `git -C /Users/casey.allard/lexlean-r4 status --short` clean).
- `532b3052-77b9-49f1-a8a8-4d4b304627d6` (`pro` non-author fitness review on `a57f0a8a64f2c07330f0c7d0bb1fcdc5d439099f`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` — executed cleanly on committed `antigravity-16` branch prior to push.

## Skills used

- Standard git, `python3 -m py_compile`, Cargo, `wasm-tools print`, `uor-pod`, and `uor-harness` loop/ratchet checks.

## Still open

- Push `antigravity/studio-phase4d-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + `[antigravity] uor-review v1`, confirm `uor-harness loop pr-check --pr <N>` & CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for final GPU pre-run approval before running `d90-foundry-lora-v7` on Vast.ai.

## Plain summary

Completed all `$0.00` offline self-tests, WASM WAT float checks, Rust compilation check (`0` warnings), and non-author Charter §4.5 fitness review (`TEST FITNESS: FIT` by `532b3052-77b9-49f1-a8a8-4d4b304627d6` at `a57f0a8a6`) for the Phase 4D `Qwen3-0.6B v2` Foundry SFT/GPTQ run (`d90-foundry-lora-v7`), ready to merge pre-registration `antigravity-16` before provisioning a Vast.ai GPU pod.
