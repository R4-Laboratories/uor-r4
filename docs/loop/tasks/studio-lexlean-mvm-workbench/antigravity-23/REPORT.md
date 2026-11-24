# REPORT for D91 Run 2 Bar #4a Negative & Qwen3-1.7B `v3.2` Pre-Registration (`antigravity-23`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `41586a64c` (PR #4226, [`antigravity-22/REPORT.md`](../antigravity-22/REPORT.md); `ORDER.md` committed alone first at `0bc938856`).

## What did not work

- **D91 Run 2 (`d91-foundry-17b-v2`, `v3.1`, `vast-62303346`, `$0.2440` spent) passed Bar #1 WAT & BOTH BF16 Bars #2 (`97.35%`) and #3 (`96.0%`), then triggered Stop Rule Gate #2 on `UORR4V13` Bar #4a (`mean_kl_nats = 0.191195 > 0.08`):**
  - After passing Bar #1 WAT (`compute_float_ops: 0` across `248` funcs) and beating baseline (`49.41%` -> **`97.35%` [`331/340`]** tool/build; `43.0%` -> **`96.0%` [`96/100`]** Chat with `0` repetition/length misses), `recompile_holo_v13` compiled `1,316,411,550`-byte `UORR4V13` (`top1_parity_pct: 100.00%`, `top5_parity_pct: 100.00%`, coherent English), yet `mean_kl_nats` measured **`0.191195`** (`0.025569` `ood_03_math` .. `0.563900` `ood_06_rope`; index-6 `quantile(0.50)` = `0.157731`), halting cleanly at Stop Rule Gate #2 before `--mode pack`.
  - **Root cause (`results_v3_run2/d91-out/train.log` & `foundry_v13_receipt.json`):** `epochs=2, r=32, lr=1.2e-4, wd=0.01` hit its validation minimum already at `0.50ep` (`val_loss=1.44963`, `tool=0.41397`, `chat=2.50707`, `telemetry_only`, `∫lr=0.007336`), then overfit by `+0.504` nats to `1.00ep` (`val=1.74351`, `chat=3.01057`, `∫lr=0.014188`, **uncooled `lr=6.817e-5`**). Restoring the overfit, uncooled `1.00ep` `r=32` checkpoint perturbed base general weights (`ood_07_git` FP32 teacher leaked `uncommitted修改`).
- **Host test environment caveat (`$0.00` spent in `antigravity-23`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `train_foundry_lora_v3_2.py` unit tests ran against in-memory stubs while `Running: 0 pod(s)` is verified here (`$0.00` spent in this PR).

## What changed

- **Built & offline-verified `v3/train_foundry_lora_v3_2.py` (`$0.00`, untracked local toolchain only — zero repo code or scripts committed):**
  - Left `train_foundry_lora_v3.py`, `train_foundry_lora_v3_1.py`, `generate_foundry_sft_v3.py`, `eval_foundry_v13_v3.py`, `foundry_schemas_v2.json`, `foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) **100% byte-identical**.
  - Kept `v3_1.py`'s `chat`-history masking in `encode_sft_episode()`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, and `select_best_checkpoint()` while applying ONLY 2 targeted fixes in `v3/train_foundry_lora_v3_2.py` (`sha256: 73b58dd242c55477756ba16b4583dfdba4b2f990a772fe09ca50e521c3c7639d`):
    1. Restored `diag_v5`-matched OOD-safe LoRA capacity/regularization (`--epochs 1 --lr 1.6e-4 --lora-r 16 --lora-alpha 32 --lora-dropout 0.08 --weight-decay 0.03 --label-smoothing 0.02`, `∫lr = 0.012000`, full cooling to `lr = 0.0`).
    2. Added `0.875` (`7/8`) checkpointing in `epochs=1` (`{0.75 (2.81e-5), 0.875 (7.37e-6), 1.00 (0.0)}`) routed through `select_best_checkpoint()` (`0.50` at `9.09e-5` telemetry-only).
- **Pre-registered Run 3 (`d91-foundry-17b-v3`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-23/`):**
  - [`ORDER.md`](ORDER.md) (`3,329 B < 4,096 B`, committed alone first at `0bc938856`).
  - [`PREP.md`](PREP.md): Pinned `d91-foundry-17b-v3`, unchanged Bars #1–#5 (`--max-new-tokens 448` explicit on `--mode eval`, `--mirror-url ...weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`), bounded Charter §4.1/§4.2 pre-teardown Bar #4a diagnostic clause (`~15s` Bar #4b + optional `~3m` no-calib recompile to `/workspace/d91-out/v13_weights_nocalib`), `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `1.5h / <= $1.00`, and non-author `TEST FITNESS: FIT` signoff (`bd14425f8` by `85ef6380-4be2-4788-b8d7-2851f296cf53`).

## Measurements

- **Archived Run 2 receipts (`local/antigravity/d90-foundry-lora/results_v3_run2/d91-out/`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99...`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`.
  - `lora_eval.json`: `tool_pass: 331/340` (**`97.35%`**: `282/288` `tool_call`, `49/52` `prism_write`), `chat_pass: 96/100` (**`96.0%`**, `0` repetition/length misses), `total_failures: 13`, `bars_passed: true`.
  - `train.log` (`34,865,152 / 1,755,440,128 = 1.9861%`): `0.50ep val=1.44963 (tool=0.41397, chat=2.50707)` -> `0.75ep (1.54411, 0.46031, 2.65071)` -> `1.00ep (1.74351, 0.50254, 3.01057, restored)` -> `2.00ep (2.09108, 0.55869, 3.65568)`.
  - `foundry_v13_receipt.json`: `float_ops: 0`, `dense_matmul_ops: 0`, `total_bytes: 1,316,411,550`, `piece_count: 1,256`, `top1_parity_pct: 100.00%`, `top5_parity_pct: 100.00%`, **`mean_kl_nats: 0.191195`** (Bar #4a FAIL).
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Run 1 `vast-62298484` (`$0.508/h`, `1185s`): `$0.1672` + Run 2 `vast-62303346` (`$0.508/h`, `1729s`): **`$0.2440`** = **`$0.4112` cumulative (`$2.5888` remaining of D91's `+$3.00` cap)**; `uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.
- **Pinned `v3.2` SHA-256:** `v3/train_foundry_lora_v3_2.py`: `73b58dd242c55477756ba16b4583dfdba4b2f990a772fe09ca50e521c3c7639d` (all other `v3`, `v3_1`, `val.jsonl` `06814bd9...` SHAs unchanged).

## Checks run at the head

- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_2.py` — PASS (`0` errors).
- `train_foundry_lora_v3_2.py` offline host suite (`encode_sft_episode` `chat` history masking, `epochs=1` `{0.75, 0.875, 1.00}` triggers across `len=1..600`, `select_best_checkpoint` 4/4 scenarios, cosine `∫lr = 0.012000000`, `HoloCycleGuardLogitsProcessor` 7/7 + `2,500`-trial differential fuzzing inside `85ef6380-4be2-4788-b8d7-2851f296cf53` with `0` mismatches) — PASS.
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `0` overlap, `100.0%` gold).
- `85ef6380-4be2-4788-b8d7-2851f296cf53` (`pro` non-author FIT review at `bd14425f8c3c9fef6cea4a343642d9816428b961`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` verified inside `.worktrees/antigravity-studio-phase5-d91-r3-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Push `antigravity/studio-phase5-d91-r3-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 3 (`d91-foundry-17b-v3`) GPU launch.

## Plain summary

Closed out D91 Run 2 (`d91-foundry-17b-v2`, `vast-62303346`, `$0.2440` spent; `$0.4112` cumulative of `+$3.00`, `$2.5888` remaining, `0` pods active), which passed Bar #1 (`compute_float_ops: 0`), BF16 Bar #2 (**`97.35%`** = `331/340` tool/build), and BF16 Bar #3 (**`96.0%`** = `96/100` Chat, zero repetition/length misses), then halted at Stop Rule Gate #2 on `UORR4V13` Bar #4a (`mean_kl_nats = 0.191195 > 0.08` from restoring an uncooled overfit `1.00ep` `r=32` checkpoint despite `100%` top-1/top-5 parity). Created & offline-verified cooled OOD-safe `v3/train_foundry_lora_v3_2.py` (`r=16, alpha=32, dropout=0.08, wd=0.03, epochs=1, lr=1.6e-4, ∫lr=0.012`, `{0.75, 0.875, 1.00}` `select_best_checkpoint`, masked `chat` history), earned non-author `TEST FITNESS: FIT` (`85ef6380-4be2-4788-b8d7-2851f296cf53`), and pre-registered `d91-foundry-17b-v3`.
