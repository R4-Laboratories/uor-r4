# REPORT for Qwen3-0.6B Foundry Webharness LoRA v2.3 Pre-Registration (`antigravity-19`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `2fab62985` (PR #4218, `antigravity-18`; `ORDER.md` committed alone first at `97141742a`).

## What did not work

- **Run 9 (`d90-foundry-lora-v9`, `vast-62279565`, `1× RTX 5090`, `uptime_s: 1881`, `$0.2722`) stopped under Charter Stop Rule before `recompile_holo_v13`:**
  - Bar #1 WAT (`wasm_float_audit.json`: `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_inference_functions_scanned: 248`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`) and baseline (`28/340` tool = `8.24%`, `18/100` Chat = `18.0%`, `394` failures) ran cleanly, and merged BF16 (`lora_eval.json`, `--max-new-tokens 448`) reached **`333/340` (`97.94%`)** on held-out tool/build (`283/288` `tool_call`, `50/52` `prism_write`; `7` failures: `val_tc_workspace_read_013/016` on unseen `"Fetch ... from project storage"`, `val_tc_github_pull_002/003/014` misclassified as `github_push`, `val_pw_008` `invalid_path:linux_exec`, and `val_pw_052` `unclosed_prism_write:1!=0` digit-loop).
  - However, **`77/100` (`77.0%`) Chat failed Bar #3 (`>= 95.0%`)**: across `100` unseen prompts (`0` `\ufffd`, `0` Chinese tokens, `9/20` topics `5/5`), `23` Chat episodes failed (`8` `low_unique_ratio` decoding loops mostly hitting `448` tokens, `7` `missing_expected_keyword`, `3` `repeated_4gram:3`, `3` `too_few_words` [`58`, `58`, `30`], and `2` `forbidden_tag` [`<prism-write`, `<tool_call>`]).
  - Offline inspection of `/Users/casey.allard/lexlean-r4/crates/wasm/src/holo_geometric_runtime.rs` (`6762-6827`) showed an `L-007` instrument mismatch: compiled `HoloV13Model::pick_next_holo_token` already blocks `>= 3` consecutive identical alphanumeric tokens, repeated `,`/`;`, and `p in 2..=8` token cycles, whereas BF16 `model.generate(do_sample=False)` had zero cycle guard. Meanwhile `train.log` showed `val_loss_tool` (`0.53079 -> 0.55014 -> 0.56557`) driving selection back to uncooled `0.75`-epoch step `168/225` (`lr=1.93e-5`, `val_loss=1.72266`) while `val_loss_chat` stayed at `2.92652 -> 2.89838 -> 2.90543` (`~5.5` exposures per unique Chat target).
- **Initial `v2.3` Chat template catches fixed offline at `$0.00`:** Adding `+4` `EXTRA_FOUNDRY_TOPICS_TRAIN` (`84 -> 88`) and `+2` `mt_chat` (`12 -> 14`) entries initially failed `check_chat_quality(..., min_words=85)` on `EXTRA[87]` (`83`w) and `mt_chat[12..13]` (`81`w, `84`w); expanded all three cleanly (`85..133` words across all `158` unique targets, `0` repeated 4-grams).
- **Host unit-test environment caveat:** Laptop system `python3` has no `torch` installed, so `HoloCycleGuardLogitsProcessor` parity (`6/6` deterministic cases + 5,000-trial differential fuzzing inside `64af0cf6-0d5e-4d15-bc1b-483eafec74d5`) was verified offline using lightweight in-memory `torch`/`transformers`/`peft` stubs against `holo_geometric_runtime.rs:6762-6827`; full live `torch 2.8.0+cu128` execution occurs on the GPU pod.
- **No GPU pod launched in this PR:** `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`; Run 10 (`d90-foundry-lora-v10`, final `$3.00`-cap run) awaits pre-run approval after `antigravity-19` merges into `main`.

## What changed

- **Charter §4.5 Pre-Registration (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-19/`):**
  - [`ORDER.md`](ORDER.md): Task order & cancel criteria (`3,582 B < 4,096 B`, strictly scoped to `antigravity-19/**`, committed alone first at `97141742a`).
  - [`PREP.md`](PREP.md): Pinned `d90-foundry-lora-v10` (`v2.3`, final D90 `$3.00`-cap run) pre-registration documenting Run 9 observed results, cumulative D90 ledger (`$2.5306` spent across Runs 1–9, `$0.4694` remaining, Run 10 hard cap `0.75h / <= $0.40`), `4,000`-row `60%`-Chat `v2.3` split (`DATA_SEED = 20261126`, `720` `tool_call`, `448` `prism_write`, `432` `multi_turn`, `2,400` `chat` across `158` unique answers repeated `15..16`× with `2,400/2,400` unique prompts, `192` default + `3,808` `FOUNDRY_SYSTEM_PROMPT`, `0` `.lex.tex` leakage, `0` repeated 4-grams), `--lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --weight-decay 0.03`, `--epochs 1 --lr 1.3e-4` (`∫lr = 0.016250`, cooled `1.00`-epoch default with `val_loss_chat > 0.10` guard against `0.75`), `HoloCycleGuardLogitsProcessor`, `--max-new-tokens 448` passed explicitly, unchanged Bars #1–#5, safe archive protocol, and non-author `TEST FITNESS: FIT` signoff (`195a0b527` by `64af0cf6-0d5e-4d15-bc1b-483eafec74d5`).
  - Untracked local toolchain updated & verified offline at `$0.00` under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/v2/` (0 repo scripts or binaries committed).

## Measurements

- **Pinned local script SHA-256 digests (`local/antigravity/d90-foundry-lora/`):**
  - `v2/foundry_schemas_v2.json`: `56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, `18` tools byte-identical to [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs))
  - `v2/generate_foundry_sft_v2.py`: `6048a5ac6b0fae1863626a03c369b61e3a95b630f15e66dee45e3f729623b01b`
  - `v2/train_foundry_lora_v2.py`: `1baff605df1c055f8be34f2ae07437ec208fb008e6217e5b7333a8bdfdf5e91c`
  - `v2/eval_foundry_v13_v2.py`: `bf82c7feb6733ad976c86da3eb8f214c8a50831b67af7c067148526ff25e81de`
  - `v2/foundry_v13_eval_pack_v2.rs`: `0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5` (unchanged from PR #4214)
  - `patch_recompile.py`: `bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de` (unchanged from Run 5 / PR #4214)
- **Deterministic dataset verification (`DATA_SEED = 20261126`):**
  - `train.jsonl` (`4,000` rows: `720` `tool_call`, `448` `prism_write`, `432` `multi_turn`, `2,400` `chat`; `158` unique Chat targets repeated `15..16`×; `2,400/2,400` unique train Chat prompts; `0` repeated 4-grams; `192` `"You are a helpful assistant."` + `3,808` `FOUNDRY_SYSTEM_PROMPT`; `0` `ood_01..12` leakage; `0` `.lex.tex` path overlap): `4d184fcf5e9d2bbb5cc62a15b331fb16ae610664ea4790fac88b62259d0a1832`
  - `val.jsonl` (`440` rows: `288` `tool_call`, `52` `prism_write`, `100` `chat`; `440/440` `FOUNDRY_SYSTEM_PROMPT`; `train_val_overlap_hashes: 0`): `c8b14760fdae0382fb76093f56a2708062803374c5fcd7f91170c3cd40b01987`
  - `calib_foundry.txt` (`96` user-only rows from `train.jsonl`): `5a1abd396980246b7a29eb828831d554dde49264fce3ae4134612e4bcdb6d540`
  - `parity_probe_prompts.json` (`16` stratified `val.jsonl` episodes -> `576` positions): `65784c013739c90158d0982b473eb7326267804f07f192c1461de8de1e4b0f7f`
- **Pod ledger & balance:** `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`; `uor-pod log` confirms Runs 1–8 `$2.2584` + Run 9 `vast-62279565` (`uptime_s: 1881` @ `$0.521/h` = `$0.2722`) = `$2.5306` cumulative spend, leaving `$0.4694` inside D90's `$3.00` cap (`<= $0.40` allocated for final Run 10).

## Checks run at the head

- `python3 -m py_compile local/antigravity/d90-foundry-lora/v2/*.py` — PASS (`0` errors on final `v2.3` scripts).
- `HoloCycleGuardLogitsProcessor` host unit test & 5,000-trial differential fuzzing (`64af0cf6-0d5e-4d15-bc1b-483eafec74d5`) against `holo_geometric_runtime.rs:6762-6827` — PASS (`6/6` directed assertions + `154` cycle triggers, `0` mismatches using in-memory `torch` stub).
- `python3 local/antigravity/d90-foundry-lora/v2/eval_foundry_v13_v2.py --self-test` — PASS (`self_test: "PASS"`, `agent_loop_verified: true`, `tool_count: 18`, `train_count: 4000`, `val_count: 440`, `calib_foundry_count: 96`, `train_val_overlap_hashes: 0`, `gold_eval: 340/340` tool + `100/100` Chat = `100.0%`, `bars_passed: true`).
- `python3 local/antigravity/d90-foundry-lora/v2/eval_foundry_v13_v2.py --wasm local/antigravity/d90-foundry-lora/foundry-v13-wasm/pkg/uor_foundry_v13_wasm_bg.wasm --holo-src /Users/casey.allard/lexlean-r4/crates/wasm/src/holo_geometric_runtime.rs` — PASS (`holo_inference_functions_scanned: 307`, `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
- `64af0cf6-0d5e-4d15-bc1b-483eafec74d5` (`pro` non-author fitness review on `195a0b527a847c36fec1f782db502a43de8f1155`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` — verified cleanly in `.worktrees/antigravity-studio-phase4d-prep-v10` prior to push.

## Skills used

- Standard git, `python3 -m py_compile`, `wasm-tools print`, `uor-pod`, and `uor-harness` loop/ratchet checks.

## Still open

- Push `antigravity/studio-phase4d-prep-v10`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub Actions CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 10 pre-run approval (`d90-foundry-lora-v10`).

## Plain summary

Diagnosed Run 9's Bar #3 failure (`333/340` tool = `97.94%`, `77/100` Chat = `77.0%`, driven by `8` BF16 decoding loops lacking `HoloV13Model`'s cycle guard and `val_loss_tool` pulling selection back onto uncooled `0.75`-epoch `val_loss_chat = 2.898`), upgraded local scripts to `v2.3` (`4,000` rows with `60%` Chat / `2,400` unique prompts across `158` targets repeated `15..16`×, `r=32/alpha=64`, `--epochs 1 --lr 1.3e-4` with exact `∫lr = 0.016250`, cooled `1.00`-epoch default selection, and `HoloCycleGuardLogitsProcessor`), verified all 10 SHA-256s at `$0.00`, obtained `TEST FITNESS: FIT` (`64af0cf6-0d5e-4d15-bc1b-483eafec74d5` at `195a0b527`), and prepared final D90 pre-registration `antigravity-19` (`<= $0.40` inside `$0.4694` remaining headroom) ahead of Run 10 approval.
