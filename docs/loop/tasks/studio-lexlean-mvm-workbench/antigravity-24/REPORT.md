# REPORT for D91 Run 3 Stop-Rule Negative & Qwen3-1.7B `v3.3` Pre-Registration (`antigravity-24`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `e575959fb` (PR #4229, [`antigravity-23/REPORT.md`](../antigravity-23/REPORT.md); `ORDER.md` committed alone first at `b2a6fe285`).

## What did not work

- **D91 Run 3 (`d91-foundry-17b-v3`, `v3.2`, `vast-62309873`, `$0.2048` spent) passed Bar #1 WAT & BF16 Bar #2 (`96.76%`), then triggered Stop Rule Gate #1 on BF16 Bar #3 Chat (`86.0% < 95.0%`):**
  - After passing Bar #1 WAT (`compute_float_ops: 0` across `248` funcs) and improving Tool/Build from `49.41%` (`168/340`) baseline to **`96.76%` (`329/340`: `281/288` `<tool_call>`, `48/52` `<prism-write>`)** and Chat from `43.0%` (`43/100`) baseline to **`86.0%` (`86/100`)** with zero `repeated_4gram` or `too_few_words` failures, BF16 Chat missed the `95.0%` bar by `9` prompts (`14` Chat misses: `7` `forbidden_tag` [`val_ch_005, 061, 064, 070, 078, 079, 080`], `5` `missing_expected_keyword` [`val_ch_022, 025, 027, 035, 059`], `2` `low_unique_ratio` [`val_ch_008:0.472`, `val_ch_083:0.403`]), halting cleanly at Stop Rule Gate #1 before `recompile_holo_v13` and `--mode pack`.
  - **Root cause (`results_v3_run3/d91-out/train.log` vs Runs 1–2):** Cutting rank/trainable parameters back to `r=16` (`1.0030%`) with heavier regularization (`dropout=0.08, wd=0.03, epochs=1, ∫lr=0.012000`) left `train_loss=1.63281` at `1.00ep` (`0.50ep val=1.46317/2.53240` -> `0.875ep 1.45563/2.51469` -> `1.00ep 1.45866/2.52009`). Comparing Runs 1 (`r=16, ∫lr=0.006000, train_loss=2.586 -> 76%`), 3 (`r=16, ∫lr=0.012000, train_loss=1.633 -> 86%`), and 2 (`r=32, drop=0.05, cut@1.00ep train_loss=1.169 -> 97.35% tool & 96.0% Chat`) shows `1.7B` needs `r=32` (`1.9861%`) capacity + `∫lr≈0.015` to learn all unseen Foundry self-knowledge essays (`022, 025, 027, 035, 059`) and prose/tool boundary separation (`forbidden_tag`), while `val_loss_chat` rises monotonically before full convergence.
- **Host test environment caveat (`$0.00` spent in `antigravity-24`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `train_foundry_lora_v3_3.py` host unit checks ran against in-memory stubs while `Running: 0 pod(s)` is verified here (`$0.00` spent in this PR).

## What changed

- **Built & offline-verified `v3/train_foundry_lora_v3_3.py` (`$0.00`, untracked local toolchain only — zero repo code or scripts committed):**
  - Left `train_foundry_lora_v3.py`, `train_foundry_lora_v3_1.py`, `train_foundry_lora_v3_2.py`, `generate_foundry_sft_v3.py`, `eval_foundry_v13_v3.py`, `foundry_schemas_v2.json`, `foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) **100% byte-identical**.
  - Kept `v3_1.py`/`v3_2.py`'s `chat`-history masking in `encode_sft_episode()`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, and `select_best_checkpoint()` while applying ONLY 2 targeted fixes in `v3/train_foundry_lora_v3_3.py` (`sha256: f88736f925cf6ceab37a723ae1ebcfbbd39f52d1e732a66ba86ae6b281bb499c`):
    1. Restored Run 2's proven `r=32, alpha=64, dropout=0.05` capacity (`1.9861%` trainable) with `2×` weight decay (`wd=0.02`, `ls=0.02`) and full `1`-epoch cosine cooling (`--epochs 1 --lr 2.0e-4`, exact `∫lr = 0.015000000`, `lr=0.0` at `1.00ep`).
    2. Marked `{0.50, 0.75, 0.875}` (`!= len(train_loader)`) as `telemetry_only` so ONLY the fully cooled `1.00 ep` (`lr=0.0`) snapshot enters `select_best_checkpoint()`.
- **Pre-registered Run 4 (`d91-foundry-17b-v4`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-24/`):**
  - [`ORDER.md`](ORDER.md) (`3,412 B < 4,096 B`, committed alone first at `b2a6fe285`).
  - [`PREP.md`](PREP.md): Pinned `d91-foundry-17b-v4`, unchanged Bars #1–#5 (`--max-new-tokens 448` explicit on `--mode eval`, `--mirror-url ...weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`), Charter §4.1/§4.2 Bar #4a no-`--calib-file` fallback (`v13_pieces_nocalib`, passing ALL Bars #1–#5 on the selected container), `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `1.5h / <= $1.00`, and non-author `TEST FITNESS: FIT` signoff (`77b31ca18` by `38889323-6868-40a6-a324-c8080a95adf7`).

## Measurements

- **Archived Run 3 receipts (`local/antigravity/d90-foundry-lora/results_v3_run3/d91-out/`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`.
  - `baseline_eval.json`: `tool_pass: 168/340` (`49.41%`: `167/288` `tool_call`, `1/52` `prism_write`), `chat_pass: 43/100` (`43.0%`).
  - `lora_eval.json` (`pred_sha256: eae264ab0ad0ccdb7c963d75c4510779abf0374d2da06f74b8f5b19af1d9934c`): `tool_pass: 329/340` (**`96.76%`**: `281/288` `tool_call`, `48/52` `prism_write`), `chat_pass: 86/100` (**`86.0%`**), `total_failures: 25`, `bars_passed: false`.
  - `train.log` (`17,432,576 / 1,738,007,552 = 1.0030%`): `0.50ep` (`1.46317, tool=0.41596, chat=2.53240`) -> `0.75ep` (`1.46905, 0.41924, 2.54093`) -> `0.875ep` (`1.45563, 0.41837, 2.51469`) -> `1.00ep` (`1.45866, 0.41910, 2.52009, train_loss=1.63281, restored`).
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Run 1 `vast-62298484` (`$0.508/h`, `1185s`): `$0.1672` + Run 2 `vast-62303346` (`$0.508/h`, `1729s`): `$0.2440` + Run 3 `vast-62309873` (`$0.508/h`, `1451s`): **`$0.2048`** = **`$0.6160` cumulative (`$2.3840` remaining of D91's `+$3.00` cap)**; `uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.
- **Pinned `v3.3` SHA-256:** `v3/train_foundry_lora_v3_3.py`: `f88736f925cf6ceab37a723ae1ebcfbbd39f52d1e732a66ba86ae6b281bb499c` (all 8 sibling scripts & `val.jsonl` `06814bd9...` unchanged).

## Checks run at the head

- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_3.py` — PASS (`0` errors).
- `train_foundry_lora_v3_3.py` offline host suite (`encode_sft_episode` `chat` history masking vs `multi_turn`, `epochs=1` trigger sweep across `[1, 2, 3, 4, 7, 8, 600]` + `1..600` inside `38889323-6868-40a6-a324-c8080a95adf7`, `select_best_checkpoint` 4/4 scenarios, exact cosine `∫lr = 0.015000000` + checkpoint LRs at `opt_step=75/112/131/150`, `HoloCycleGuardLogitsProcessor` 7/7 checks) — PASS.
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `0` overlap, `100.0%` gold).
- `38889323-6868-40a6-a324-c8080a95adf7` (`pro` non-author FIT review at `77b31ca1843377e12ab4abe553511030bff781eb`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` verified inside `.worktrees/antigravity-studio-phase5-d91-r4-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Push `antigravity/studio-phase5-d91-r4-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 4 (`d91-foundry-17b-v4`) GPU execution.

## Plain summary

Closed out D91 Run 3 (`d91-foundry-17b-v3`, `vast-62309873`, `$0.2048` spent; `$0.6160` cumulative of `+$3.00`, `$2.3840` remaining, `0` pods active), which passed Bar #1 (`compute_float_ops: 0`) and BF16 Bar #2 (**`96.76%`** = `329/340` tools/builds) while halting cleanly at Stop Rule Gate #1 on Chat (`86.0% < 95.0%`) due to under-converged `r=16` capacity under heavy regularization (`train_loss=1.633`). Created & offline-verified cooled capacity-restored `v3/train_foundry_lora_v3_3.py` (`r=32, alpha=64, dropout=0.05, wd=0.02, epochs=1, lr=2.0e-4, ∫lr=0.015`, `{0.50, 0.75, 0.875}` `telemetry_only` -> cooled `1.00 ep` restored), earned non-author `TEST FITNESS: FIT` (`38889323-6868-40a6-a324-c8080a95adf7`), and pre-registered `d91-foundry-17b-v4`.
