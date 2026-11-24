# REPORT for Qwen3-0.6B Foundry Webharness LoRA v2.1 Pre-Registration (`antigravity-17`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `596fd86d3` (PR #4214, `antigravity-16`; `ORDER.md` committed alone at `f7f82344d`).

## What did not work

- **Run 7 (`d90-foundry-lora-v7`, `vast-62261356`, `$0.2313`) stopped under Charter Stop Rule before touching live artifacts:**
  - Despite passing Bar #1 WAT (`compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_inference_functions_scanned: 249`) and moving baseline (`29/340` tool = `8.53%`, `13/100` Chat = `13.0%`) up to `314/340` (`92.35%`) tool on merged BF16, heavy repetition in the `7,200`-row `v2` dataset (`8` build specs repeated across `1,400` rows, `~46` Chat answers repeated across `1,840` rows) caused monotonic validation loss overfitting after `0.5` epoch (`opt_step=225: val_loss=2.59591` -> `450: 2.73716` -> `675: 2.86139` -> `900: 2.87949`).
  - Restoring the under-trained `0.5`-epoch step (`225`) yielded only `69/100` (`69.0%`) Chat on `lora_eval.json` (failing Bar #3 `>= 95.0%`; `bad_terminator:'1'`, `missing_expected_keyword`) and misclassified `8` `github_pull` prompts as `github_push` plus `7` unclosed `<prism-write>` blocks.
  - Quantized `UORR4V13` (`473,524,382` B, `--promote-down-int8`) scored `top1=83.33%`, `top5=100.00%`, `mean_KL=0.128702` (> `0.08`, failing Bar #4a) on `recompile.log`, switching into Chinese on `[ood_01_wasm]` (`LexLean-R4 是一个专为 web 接口开发的静态数学推导引擎...`) with `0.29–0.35` spikes on `ood_04/08/10`.
  - **Run 7 Archive Caveat:** Downloading Run 7's local tarball yielded `0` bytes because remote cleanup ran `pkill -TERM -f foundry_v13_eval_pack_v2`, terminating its own SSH shell command line; Run 7 primary metrics and pod billing (`uptime_s: 1598`, `$0.2313`, cumulative `$2.0794`) were preserved in local host session logs and `uor-pod log`, and Run 8 pre-registers an explicit safe-archive rule forbidding `pkill -f` self-matches and checking tarball contents prior to `--confirm-archived`.
- **Initial `v2.1` Chat template catches fixed offline at `$0.00` before review:**
  - Expanding `EXTRA_FOUNDRY_TOPICS_TRAIN`, `GENERAL_TECH_TOPICS_TRAIN`, `STANDARDS_RAG_TRAIN`, and `mt_chat` triggered `check_chat_quality` rejections on a literal `<tool_call>` substring inside topic `#10` (`forbidden_tag:<tool_call>`) and `too_few_words` (`59–84 < 85`) across short newly added entries; replaced the literal XML tag with prose and enriched each explanation to `85–133` words (`0` failures across all `3,200` train / `440` val rows).
- **Precision notes recorded from non-author fitness review (`943c6b61-0483-4782-8dbc-fd49b7ed094c`):**
  1. `ORDER.md` (committed alone first at `f7f82344d`) cites `32` `<prism-write>` specs, whereas `generate_foundry_sft_v2.py` (`c6c25e1d...`) holds `31` entries in `TRAIN_BUILD_SPECS` (`186` spec×template combinations generating `640` train rows; `13` disjoint specs in `VAL_BUILD_SPECS`). Corrected `PREP.md` to `31` (`186` spec×template pairs).
  2. `foundry_v13_eval_pack_v2.rs` (`0feb9b07...`) defaults internally to `DEFAULT_MAX_NEW_TOKENS = 384`, so Run 8 explicitly passes `--max-new-tokens 448` on CLI as specified in `PREP.md`.
- **No GPU pod launched in this PR:** `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`; actual `d90-foundry-lora-v8` execution awaits pre-run approval after `antigravity-17` merges into `main`.

## What changed

- **Charter §4.5 Pre-Registration (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-17/`):**
  - [`ORDER.md`](ORDER.md): Task order & cancel criteria strictly bounded to `antigravity-17/**` (committed alone first at `f7f82344d`).
  - [`PREP.md`](PREP.md): Pinned `d90-foundry-lora-v8` (`v2.1`) pre-registration documenting Run 7 diagnosis, cumulative D90 ledger (`$2.0794` spent across Runs 1–7, `$0.9206` remaining of `$3.00` cap, Run 8 hard cap `1.25h / <= $0.68`), `3,200`-row diversified `v2.1` split (`DATA_SEED = 20261124`), `24` OOD-disjoint general English anchors (`120` `"You are a helpful assistant."` + `120` `FOUNDRY_SYSTEM_PROMPT`), `lr = 6.0e-5`, `weight_decay = 0.05`, `--max-new-tokens 448`, epochs `1.0 / 1.5 / 2.0` selection, unchanged Bars #1–#5, safe archive protocol, and non-author `TEST FITNESS: FIT` signoff (`1d598735f` by `943c6b61-0483-4782-8dbc-fd49b7ed094c`).
  - Untracked local toolchain updated & verified offline at `$0.00` under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/v2/` (0 repo scripts or binaries committed).

## Measurements

- **Pinned local script SHA-256 digests (`local/antigravity/d90-foundry-lora/`):**
  - `v2/foundry_schemas_v2.json`: `56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, `18` tools byte-identical to [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs))
  - `v2/generate_foundry_sft_v2.py`: `c6c25e1d9f7417c9e39ed7f9b72cc2b563e933c83c20aa0ea00c199d1f5394ec`
  - `v2/train_foundry_lora_v2.py`: `fdcbf6fbbe080165ae9f562bcc83f703e2ec3ecaae1cda8d94718eecefef05e6`
  - `v2/eval_foundry_v13_v2.py`: `93fdd159f65aa0471a5eb3ce62a7e8fc7030f4b1484956680543db87422997b9`
  - `v2/foundry_v13_eval_pack_v2.rs`: `0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5` (unchanged from PR #4214)
  - `patch_recompile.py`: `bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de` (unchanged from Run 5 / PR #4214)
- **Deterministic dataset verification (`DATA_SEED = 20261124`):**
  - `train.jsonl` (`3,200` rows: `1,080` `tool_call`, `640` `prism_write`, `540` `multi_turn`, `940` `chat`; `120` `"You are a helpful assistant."` + `3,080` `FOUNDRY_SYSTEM_PROMPT`; `0` OOD `ood_01..12` leakage markers): `7cfdccd128d50fb862b082eb6bfe9eb81738206c9ae84316efc2ccf0e96bec87`
  - `val.jsonl` (`440` rows: `288` `tool_call`, `52` `prism_write`, `100` `chat`; `440/440` `FOUNDRY_SYSTEM_PROMPT`; `train_val_overlap_hashes: 0`): `3bacd41bd1889ab5d3f6754ddda446022d00ed3a0d3e0c10e9dad220ecc2be9b`
  - `calib_foundry.txt` (`96` user-only rows from `train.jsonl`): `934cb61d6ce6000b2220ba3d2d89614376a5367be8792c7ef69634c84106c67e`
  - `parity_probe_prompts.json` (`16` stratified `val.jsonl` episodes -> `576` positions): `393c4664da7d2369fd046443066ec41f2ac0e96c486a31e865e3c56262791de0`
- **Pod ledger & balance:** `uor-pod status` shows `Running: 0 pod(s) ($0/h of $8.00/h)`; `uor-pod log` confirms Runs 1–6 `$1.8481` + Run 7 `vast-62261356` (`uptime_s: 1598` @ `$0.521/h` = `$0.2313`) = `$2.0794` cumulative spend, leaving `$0.9206` inside D90's `$3.00` cap (`<= $0.68` allocated for Run 8).

## Checks run at the head

- `python3 -m py_compile local/antigravity/d90-foundry-lora/v2/*.py` — PASS (`0` errors on final `v2.1` scripts).
- `python3 local/antigravity/d90-foundry-lora/v2/eval_foundry_v13_v2.py --self-test` — PASS (`self_test: "PASS"`, `agent_loop_verified: true`, `tool_count: 18`, `train_count: 3200`, `val_count: 440`, `calib_foundry_count: 96`, `train_val_overlap_hashes: 0`, `gold_eval: 340/340` tool + `100/100` Chat = `100.0%`, `bars_passed: true`, plus independent non-author run inside `943c6b61-0483-4782-8dbc-fd49b7ed094c`).
- `python3 local/antigravity/d90-foundry-lora/v2/eval_foundry_v13_v2.py --wasm local/antigravity/d90-foundry-lora/foundry-v13-wasm/pkg/uor_foundry_v13_wasm_bg.wasm --holo-src /Users/casey.allard/lexlean-r4/crates/wasm/src/holo_geometric_runtime.rs` — PASS (`holo_inference_functions_scanned: 307`, `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
- `943c6b61-0483-4782-8dbc-fd49b7ed094c` (`pro` non-author fitness review on `1d598735f280c8fe9ec84526532b15d14fd81701`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` — verified cleanly in `.worktrees/antigravity-studio-phase4d-prep-v8` prior to push.

## Skills used

- Standard git, `python3 -m py_compile`, `wasm-tools print`, `uor-pod`, and `uor-harness` loop/ratchet checks.

## Still open

- Push `antigravity/studio-phase4d-prep-v8`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub Actions CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for final Run 8 pre-run approval (`d90-foundry-lora-v8`).

## Plain summary

Diagnosed Run 7's `0.5`-epoch overfitting (`69/100` Chat Bar #3 FAIL, `mean_KL=0.128702` Bar #4a FAIL with Chinese OOD drift), verified the balanced `3,200`-row `v2.1` dataset and `train/eval` pipeline offline at `$0.00` with zero OOD leakage, obtained independent non-author fitness signoff (`TEST FITNESS: FIT` by `943c6b61-0483-4782-8dbc-fd49b7ed094c` at `1d598735f`), and prepared pre-registration `antigravity-17` (`d90-foundry-lora-v8`, `<= $0.68` inside `$0.9206` remaining D90 headroom) for merge ahead of GPU launch review.
