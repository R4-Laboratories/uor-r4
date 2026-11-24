# REPORT for D91 Run 4 Stop-Rule Negative & Qwen3-1.7B `v3.4` Pre-Registration (`antigravity-25`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `9af20178c` (PR #4230, [`antigravity-24/REPORT.md`](../antigravity-24/REPORT.md); `ORDER.md` committed alone first at `362247a02`).

## What did not work

- **D91 Run 4 (`d91-foundry-17b-v4`, `v3.3`, `vast-62316259`, `$0.1799` spent) passed Bar #1 WAT & BF16 Bar #2 (`96.47%`) and improved Chat to `93.0%`, stopping cleanly 2 prompts short at Stop Rule Gate #1 (`93/100 < 95.0%`):**
  - Restoring `r=32, alpha=64, dropout=0.05` (`1.9861%` trainable) lifted BF16 Chat from Run 3's `86.0%` to **`93.0%` (`93/100`)** alongside **`96.47%` (`328/340`: `279/288` `<tool_call>`, `49/52` `<prism-write>`)** Tool/Build while keeping `val_loss_chat = 2.78850` (lower than Run 2's `3.01057`). Only `7` Chat prompts missed: `2` `missing_expected_keyword` (`val_ch_025, 045`), `2` `too_few_words` (`val_ch_046:46`, `val_ch_073:37`), `1` `repeated_4gram:4` (`val_ch_008`: `uor-r4-studio-mvm`), `1` `bad_terminator:'*'` (`val_ch_048`), and `1` `forbidden_tag:<tool_call>` (`val_ch_071`). Halted cleanly at Stop Gate #1 (`recompile_holo_v13` & `--mode pack` skipped).
  - **Root cause (`results_v3_run4/d91-out/train.log` vs Run 2):** Putting `∫lr = 0.015000` into `1` epoch (`T=150, W=12, lr=2.0e-4`) front-loaded `79%` of the LR budget into the first half-epoch (`cum_∫lr = 0.011795` vs Run 2's gentle `0.007336`), lifting early `val_loss_tool` (`0.43108` vs Run 2's `0.41397` at `0.50 ep`) and leaving `val_ch_048` trailing `*` and `046`/`073` concise.
- **Host test environment caveat (`$0.00` spent in `antigravity-25`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `train_foundry_lora_v3_4.py` host unit checks ran against in-memory stubs while `Running: 0 pod(s)` is verified here (`$0.00` spent in this PR).

## What changed

- **Built & offline-verified `v3/train_foundry_lora_v3_4.py` (`$0.00`, untracked local toolchain only — zero repo code or scripts committed):**
  - Left `train_foundry_lora_v3.py`, `train_foundry_lora_v3_1.py`, `train_foundry_lora_v3_2.py`, `train_foundry_lora_v3_3.py`, `generate_foundry_sft_v3.py`, `eval_foundry_v13_v3.py`, `foundry_schemas_v2.json`, `foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) **100% byte-identical**.
  - Kept `v3_1.py`/`v3_2.py`/`v3_3.py`'s `chat`-history masking in `encode_sft_episode()`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, and `select_best_checkpoint()` while applying 2 targeted fixes in `v3/train_foundry_lora_v3_4.py` (`sha256: eed988d5eafd9c7b59be174085d06c115ae14e2af4b49eb9ce091e2ca4457d35`):
    1. Matched Run 2's gentle early LR (`--epochs 1.5 --lr 1.3e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --weight-decay 0.015 --label-smoothing 0.02`, `T=225, W=18`, `∫lr = 0.014625000`, `cum_∫lr` at `0.50/0.75/1.00 ep` = `0.008081 / 0.011481 / 0.013619`) cooling smoothly to `lr = 0.0` at `1.50 ep`.
    2. Added the fractional-epoch `global_batch` / `accum_count` training loop (`total_batches=900`, `remaining_in_group` scaling, `{0.50, 0.75, 1.00, 1.25}` `telemetry_only` -> cooled `1.50 ep` restored).
- **Pre-registered Run 5 (`d91-foundry-17b-v5`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-25/`):**
  - [`ORDER.md`](ORDER.md) (`3,494 B < 4,096 B`, committed alone first at `362247a02`).
  - [`PREP.md`](PREP.md): Pinned `d91-foundry-17b-v5`, unchanged Bars #1–#5 (`--max-new-tokens 448` explicit on `--mode eval`, `--mirror-url ...weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`), Charter §4.1/§4.2 Bar #4a no-`--calib-file` fallback (`v13_pieces_nocalib`, passing ALL Bars #1–#5 on `$ACTIVE_PIECES`), `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `1.5h / <= $1.00`, and non-author `TEST FITNESS: FIT` signoff (`0115eb7d1` by `b70740ed-05e0-4fad-b16c-d48ef3c810c5`).

## Measurements

- **Archived Run 4 receipts (`local/antigravity/d90-foundry-lora/results_v3_run4/d91-out/`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`.
  - `baseline_eval.json`: `tool_pass: 168/340` (`49.41%`: `167/288` `tool_call`, `1/52` `prism_write`), `chat_pass: 43/100` (`43.0%`).
  - `lora_eval.json` (`pred_sha256: 843743e68ee7cd67bc2a2095596e973c020d40f16a483900b13e00cf937562b9`): `tool_pass: 328/340` (**`96.47%`**: `279/288` `tool_call`, `49/52` `prism_write`), `chat_pass: 93/100` (**`93.0%`**), `total_failures: 19`, `bars_passed: false`.
  - `train.log` (`34,865,152 / 1,755,440,128 = 1.9861%`): `0.50ep` (`1.46893, tool=0.43108, chat=2.52861`) -> `0.75ep` (`1.59021, 0.45533, 2.74896`) -> `0.875ep` (`1.59358, 0.46009, 2.75091`) -> `1.00ep` (`1.61384, 0.46337, 2.78850, train_loss=1.10557, restored`).
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Run 1 `vast-62298484` (`$0.1672`) + Run 2 `vast-62303346` (`$0.2440`) + Run 3 `vast-62309873` (`$0.2048`) + Run 4 `vast-62316259` (`$0.508/h`, `1275s`): **`$0.1799`** = **`$0.7959` cumulative (`$2.2041` remaining of D91's `+$3.00` cap)**; `uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.
- **Pinned `v3.4` SHA-256:** `v3/train_foundry_lora_v3_4.py`: `eed988d5eafd9c7b59be174085d06c115ae14e2af4b49eb9ce091e2ca4457d35` (all 9 sibling scripts & `val.jsonl` `06814bd9...` unchanged).

## Checks run at the head

- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_4.py` — PASS (`0` errors).
- `train_foundry_lora_v3_4.py` offline host suite (`encode_sft_episode` `chat` masking vs `multi_turn`, fractional-epoch loop simulation across 5 `(batches_per_epoch, epochs, grad_accum)` edge cases including `(600, 1.5, 4)` plus 10 boundary cases verified inside `b70740ed`, `select_best_checkpoint` 4/4, exact cosine `∫lr = 0.014625000` + checkpoint LRs, `HoloCycleGuardLogitsProcessor` 7/7) — PASS.
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `0` overlap, `100.0%` gold).
- `b70740ed-05e0-4fad-b16c-d48ef3c810c5` (`pro` non-author FIT review at `0115eb7d167bf3682c8d40e04eef2fba49c7cd58`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` verified inside `.worktrees/antigravity-studio-phase5-d91-r5-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Push `antigravity/studio-phase5-d91-r5-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, enqueue via `gh pr merge --squash`, pull fresh `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 5 (`d91-foundry-17b-v5`) GPU execution.

## Plain summary

Closed out D91 Run 4 (`d91-foundry-17b-v4`, `vast-62316259`, `$0.1799` spent; `$0.7959` cumulative of `+$3.00`, `$2.2041` remaining, `0` pods active), which passed Bar #1 (`compute_float_ops: 0`), BF16 Bar #2 (**`96.47%`** = `328/340` tools/builds), and lifted BF16 Chat to **`93.0%`** (`93/100`), halting 2 prompts short at Stop Rule Gate #1 because compressing `∫lr=0.015` into `1` epoch front-loaded `79%` of updates (`cum_∫lr=0.011795`) by `0.50 ep`. Built & offline-verified gentle cooled `1.5`-epoch `v3/train_foundry_lora_v3_4.py` (`r=32, lr=1.3e-4, wd=0.015, W=18, ∫lr=0.014625`, fractional-epoch loop), earned non-author `TEST FITNESS: FIT` (`b70740ed-05e0-4fad-b16c-d48ef3c810c5`), and pre-registered `d91-foundry-17b-v5`.
