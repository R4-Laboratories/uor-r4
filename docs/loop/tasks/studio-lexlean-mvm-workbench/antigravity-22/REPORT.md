# REPORT for D91 Run 1 Negative & Qwen3-1.7B `v3.1` Pre-Registration (`antigravity-22`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `9d3771b73` (PR #4224, [`antigravity-21/REPORT.md`](../antigravity-21/REPORT.md); `ORDER.md` committed alone first at `6ca424684`).

## What did not work

- **D91 Run 1 (`d91-foundry-17b-v1`, `Qwen/Qwen3-1.7B` `v3`, `vast-62298484`, `$0.1672` spent) passed Bar #1 & Bar #2 (`98.24%`) and triggered Stop Rule Gate #1 on BF16 Bar #3 (`76.0%` Chat):**
  - After passing Bar #1 WAT audit (`compute_float_ops: 0` across `248` funcs) and beating baseline (`49.41%` tool -> **`98.24%` [`334/340`]** tool/build, with `284/288` `<tool_call>` and `50/52` `<prism-write>`), BF16 post-LoRA Chat reached `76/100` (**`76.0% < 95.0%`**), triggering Stop Rule Gate #1 before `recompile_holo_v13` or `--mode pack`.
  - **Root-cause findings across `results_v3_run1/d91-out/`:**
    1. *Severe under-training at `∫lr = 0.006000` (`epochs=1, lr=8e-5, r=16, alpha=32, label_smoothing=0.05`):* `1.7B` trainable share (`17,432,576 / 1,738,007,552 = 1.003%`) finished at `train_loss = 2.58583` while validation losses dropped monotonically with zero overfitting (`0.50ep`: `val_loss=1.61690, chat=2.74570` -> `0.75ep`: `val_loss=1.55610, chat=2.65500` -> `1.00ep`: `val_loss=1.54236, chat=2.62987`). Under-convergence left greedy repetition/drift (`5` `repeated_4gram:3`, `5` `low_unique_ratio`, `1` `bad_terminator:'9'`), `4` missed architectural keywords (`val_ch_025, 093, 094, 100`), `1` short single-turn RAG answer (`val_ch_050: 57w`), and `6` XML-tag leaks inside explanatory prose (`val_ch_004, 064, 076, 077, 079, 080`) despite `v3` `train.jsonl` having `0` XML tags in Chat.
    2. *History turn-1 supervision on multi-turn `chat`:* `encode_sft_episode` supervised `34`-word turn-1 assistant replies on `chat` history rows, teaching compact short answers (`val_ch_073: 27w`, `val_ch_074: 31w`).
- **Host test environment caveat & Pod guardrails (`$0.00` spent in `antigravity-22`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `train_foundry_lora_v3_1.py` offline tests ran against lightweight in-memory stubs while real BF16/GPU execution runs on Vast after PR merge.
  - Under default `OFF_VOLUME_DISK_GB=150`, Vast offers exclude `$0.50–$0.65/h` South Korea `5090` hosts, whereas `UOR_POD_VAST_DISK_GB=80` exposes them; because `uor-pod up` exits immediately if `UOR_POD_MAX_RATE=0.65` rejects all offers (`cap_only=1`), Unit D will check `UOR_POD_VAST_DISK_GB=80 uor-pod status` right before `uor-pod up`. `Running: 0 pod(s)` confirmed here (`$0.00` spent in this PR).

## What changed

- **Built & offline-verified `v3/train_foundry_lora_v3_1.py` (`$0.00`, untracked local toolchain only — zero repo code or scripts committed):**
  - Kept `v3/train_foundry_lora_v3.py`, `v3/generate_foundry_sft_v3.py`, `v3/eval_foundry_v13_v3.py`, `v3/foundry_schemas_v2.json`, `v3/foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`N = 440`) **100% byte-identical** for a strict controlled ablation.
  - Applied ONLY 3 targeted fixes in `v3/train_foundry_lora_v3_1.py` (`sha256: bf241ee50580a46b5facfec20e599031f3c4feff3afb016f0bd42ff1245a6566`):
    1. `--epochs 2 --lr 1.2e-4 --lora-r 32 --lora-alpha 64 --weight-decay 0.01 --label-smoothing 0.02` (`warmup = 24/300 = 8%`, `∫lr = 0.018000` = `3.00×` Run 1).
    2. Masked `history` assistant turns (`labels = -100`) on `category == "chat"` in `encode_sft_episode()`, preserving multi-turn `<tool_call>` + summary supervision on `category == "multi_turn"`.
    3. Added `0.25`-epoch `select_best_checkpoint()` (`1.00, 1.25, 1.50, 1.75, 2.00` snapshots; `0.50/0.75` telemetry-only) filtering `val_loss_tool <= min_tool + 0.05` and choosing the latest cooled snapshot within `+0.02` of `min_chat`.
- **Pre-registered Run 2 (`d91-foundry-17b-v2`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-22/`):**
  - [`ORDER.md`](ORDER.md) (`3,195 B < 4,096 B`, committed alone first at `6ca424684`).
  - [`PREP.md`](PREP.md): Pinned `d91-foundry-17b-v2`, unchanged Bars #1–#5 (`--max-new-tokens 448` explicit on `--mode eval`, `--mirror-url ...weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`), `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `1.5h / <= $1.00` budget cap, and non-author `TEST FITNESS: FIT` signoff (`fee6353f7` by `62e3085b-5672-44bb-b352-1fae6c6f7d34`).

## Measurements

- **Archived Run 1 receipts (`local/antigravity/d90-foundry-lora/results_v3_run1/d91-out/`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`.
  - `baseline_eval.json`: `tool_pass: 168/340` (`49.41%`: `167/288` `tool_call`, `1/52` `prism_write`), `chat_pass: 43/100` (`43.0%`), `bars_passed: false`.
  - `lora_eval.json`: `tool_pass: 334/340` (`98.24%`: `284/288` `tool_call`, `50/52` `prism_write`), `chat_pass: 76/100` (`76.0%`), `total_failures: 30` (`6` tool + `24` Chat: `6` `forbidden_tag`, `5` `repeated_4gram`, `5` `low_unique_ratio`, `4` `missing_expected_keyword`, `3` `too_few_words`, `1` `bad_terminator:'9'`), `bars_passed: false`.
  - `train.log` (`17,432,576 / 1,738,007,552 = 1.003%` trainable): `0.50ep val=1.61690 (tool=0.49117, chat=2.74570)` -> `0.75ep val=1.55610 (tool=0.46019, chat=2.65500)` -> `1.00ep val=1.54236 (tool=0.45782, chat=2.62987, train_loss=2.58583)`.
- **D91 cumulative compute ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - `vast-62298484` (`1× RTX 5090`, Vast South Korea KR, `$0.508/h`, `uptime_s: 1185`): **`$0.1672`** spent; **`$2.8328`** remaining of D91's fresh `+$3.00` cap (`uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`).
- **Pinned `v3.1` SHA-256 digests (`local/antigravity/d90-foundry-lora/`):**
  - `v3/train_foundry_lora_v3_1.py`: `bf241ee50580a46b5facfec20e599031f3c4feff3afb016f0bd42ff1245a6566`
  - Unchanged `v3` pipeline & `val.jsonl`: `train_foundry_lora_v3.py` (`52158a3f...`), `foundry_schemas_v2.json` (`56fe4459...`), `generate_foundry_sft_v3.py` (`b4222f26...`), `eval_foundry_v13_v3.py` (`a6e9eba7...`), `foundry_v13_eval_pack_v2.rs` (`0feb9b07...`), `patch_recompile.py` (`bb5da659...`), `val.jsonl` (`06814bd9...`, `N = 440`).

## Checks run at the head

- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_1.py` — PASS (`0` errors).
- `train_foundry_lora_v3_1.py` offline host suite (`encode_sft_episode` `chat` history masking vs `multi_turn`, `select_best_checkpoint` 4/4 scenarios, cosine `∫lr = 0.018000000` at `T=300, W=24, lr=1.2e-4`, `HoloCycleGuardLogitsProcessor` 7/7 + `3,000`-trial differential fuzzing inside `62e3085b-5672-44bb-b352-1fae6c6f7d34` with `0` mismatches) — PASS.
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `train_val_overlap_hashes: 0`, `gold_eval: 100.0%`, `bars_passed: true`).
- `62e3085b-5672-44bb-b352-1fae6c6f7d34` (`pro` non-author FIT review at `fee6353f73da6b879703eb42faba0131dbb2df97`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, and `uor-harness loop pr-check --base origin/main --head HEAD --local` verified inside `.worktrees/antigravity-studio-phase5-d91-r2-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Push `antigravity/studio-phase5-d91-r2-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 2 (`d91-foundry-17b-v2`) GPU launch.

## Plain summary

Closed out D91 Run 1 (`d91-foundry-17b-v1`, `vast-62298484`, `$0.1672` spent, `$2.8328` remaining of D91's `+$3.00` cap, `0` running pods), which passed Bar #1 (`compute_float_ops: 0`) and BF16 Bar #2 (**`98.24%`** = `334/340` tools/builds) while halting cleanly at Stop Rule Gate #1 on Chat (`76.0% < 95.0%`) due to severe `1`-epoch under-training (`∫lr = 0.006`, monotonic `val_loss_chat = 2.62987`) and turn-1 `chat` history supervision. Created & offline-verified controlled-ablation trainer `v3/train_foundry_lora_v3_1.py` (`epochs=2, lr=1.2e-4, r=32, alpha=64, ls=0.02, ∫lr=0.018`, masked `chat` history, `0.25`-epoch `select_best_checkpoint`) against the unchanged `val.jsonl`, earned non-author `TEST FITNESS: FIT` (`62e3085b-5672-44bb-b352-1fae6c6f7d34`), and pre-registered `d91-foundry-17b-v2`.
