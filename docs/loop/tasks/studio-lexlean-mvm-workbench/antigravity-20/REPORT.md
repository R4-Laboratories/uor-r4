# REPORT for D90 Run 10 (`v2.3`) Stop-Rule & Phase 4D Closeout (`antigravity-20`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `ce35f93db` (PR #4221, [`antigravity-19/PREP.md`](../antigravity-19/PREP.md); `ORDER.md` committed alone first at `bf5ef0583`).

## What did not work

- **Aborted SSH-stalled pod `vast-62288296` (`2026-11-24T03:52:08Z` -> `03:59:03Z`):** Offer `52190822` (`$0.521/h`) vanished in the 4-second window between `UOR_POD_DRY_RUN=1` (`03:51:57Z`) and real `uor-pod up` (`03:52:01Z`), causing automatic fallback to Czechia offer `60223969` (`$0.908/h`). Because SSH never came up within the 6-minute cutoff (`SSH or metadata probe failed`), `task-9474` and `task-9475` were cancelled and `vast-62288296` was released and force-deleted (`uptime_s: 414`, `$0.1044`) with zero scripts uploaded and zero jobs run. Immediate fix applied for retry: `UOR_POD_MAX_RATE=0.60`, atomically enforcing `<= $0.60/h` inside `uor-pod`'s lock and provisioning `vast-62289287` (`1× RTX 5090`, Vast South Korea KR, `$0.521/h`, SSH ready in `47s`).
- **Run 10 (`d90-foundry-lora-v10`, `v2.3`, `vast-62289287`, `uptime_s: 744`, `$0.1077`) triggered the pre-registered Stop Rule before `recompile_holo_v13`:**
  - Merged BF16 evaluation (`lora_eval.json`, `--max-new-tokens 448` with `HoloCycleGuardLogitsProcessor`) passed Bar #2 (`>= 92.0%`) at **`317/340` (`93.24%`)** (`272/288` `tool_call` [`7` `wrong_tool`, `3` `invalid_tool_json`, `3` `unexpected_arg`, `3` `missing_tool_call`]; `45/52` `prism_write` [`3` `unexpected_tool_call`, `3` `invalid_path`, `1` `unclosed_prism_write`]), but reached only **`78/100` (`78.0%`)** on held-out Chat (failing Bar #3 `>= 95.0%`), halting cleanly at `2026-11-24T04:11:22Z` (`STOP_RULE_TRIGGERED: lora_eval failed Bar #2/#3`, `exit=0`).
  - `HoloCycleGuardLogitsProcessor` cut BF16 decoding loops sharply relative to Run 9 (`low_unique_ratio` dropped from `8` to `1` [`val_ch_056` at `448` tokens]; `repeated_4gram` dropped from `3` to `0`), exposing the true failure taxonomy across all `22` Chat misses: **`10` `missing_expected_keyword`**, **`6` `forbidden_tag`** (`3` `<tool_call>` on `val_ch_068/080/089`, `2` `<prism-write` on `val_ch_061/062`, `1` `\ufffd` on `val_ch_024`), **`5` `too_few_words`** (`26..57` words on `val_ch_051/052/069/074/092`, including concise RAG summaries on `051/052`), and **`1` `low_unique_ratio`** (`val_ch_056:0.165`).
  - `train.log` (`--epochs 1 --lr 1.3e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --weight-decay 0.03`, `20,185,088` trainable params = `3.2756%`) recorded monotonic held-out Chat overfitting while training loss fell:
    - `0.50 ep` (`batch=500/1000`, `opt_step=125/250`, `lr=7.385083e-05`, `telemetry_only`): `train_loss=2.68264 val_loss=1.87308 val_loss_tool=0.56176 val_loss_chat=3.18798`
    - `0.75 ep` (`batch=750/1000`, `opt_step=187/250`, `lr=2.261734e-05`): `train_loss=1.49902 val_loss=2.08830 val_loss_tool=0.60452 val_loss_chat=3.57613`
    - `1.00 ep` (`batch=1000/1000`, `opt_step=250/250`, `lr=0.000000e+00`): `train_loss=1.27753 val_loss=2.12447 val_loss_tool=0.60705 val_loss_chat=3.64605` (restored `cooled_1.00_epoch_default`).
  - Qualitative inspection of `results_v2_run10/d90-out/lora_predictions.jsonl` showed `Qwen3-0.6B` confabulates unseen Foundry architecture facts (e.g., claiming Foundry compiles TypeScript into `v13` int8 models on `val_ch_040`, isolates credentials inside `foundry.env` on `val_ch_035`, or scans `manifest.json` fields with `wasm-tools print` on `val_ch_024`). Across Runs 8 (`74%`), 9 (`77%`), and 10 (`78%`), `Qwen3-0.6B` hit a parametric self-knowledge ceiling on unseen multi-paragraph Foundry essays.
- **D90 `$3.00` cap respected; zero Run 11 launched:** Live Studio (`gh-pages`) and `Casey-allard/lexlean-r4@weights-v13-qwen3-0.6b-foundry` (`3b04cd460735af679d912ce749e6e4bb2bccc0b0`) remain untouched and operational on PR #4213 Phase 4A (`311/320` = `97.19%` compiled tool, `77/80` = `96.25%` Chat, `0.045925` 12-OOD KL; [`antigravity-15/REPORT.md`](../antigravity-15/REPORT.md)).

## What changed

- **Run 10 & Phase 4D Closeout Record ([`antigravity-20/`](ORDER.md)):**
  - [`ORDER.md`](ORDER.md): Task order committed alone at `bf5ef0583`.
  - [`REPORT.md`](REPORT.md): Records Run 10 verified receipts, failure breakdown, exact D90 ledger closeout (`$2.7427` spent / `$0.2573` remaining), and transition status toward user-approved Phase 5 (`Qwen3-1.7B`).
  - Untracked archive stored locally at `local/antigravity/d90-foundry-lora/results_v2_run10.tar.gz` (`1.6M`) and `local/antigravity/d90-foundry-lora/results_v2_run10/` (zero repo code, scripts, weights, or binaries committed).

## Measurements

- **Pod telemetry (`vast-62289287`, `1× RTX 5090`, Vast South Korea KR, `$0.521/h`):** Created `04:00:54Z`, SSH ready `04:01:40Z` (`47s`), launched `04:02:35Z`, exited `04:11:22Z`, released `04:13:13Z`, deleted `04:13:21Z` (`uptime_s: 744`, `$0.1077`).
- **Bar #1 WAT audit (`results_v2_run10/d90-out/wasm_float_audit.json`):** PASS (`wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
- **Baseline (`baseline_eval.json`):** `28/340` (`8.24%`) tool (`8/288` `tool_call`, `20/52` `prism_write`), `18/100` (`18.0%`) Chat, `394` total failures (`val_sha256: c8b14760fdae0382fb76093f56a2708062803374c5fcd7f91170c3cd40b01987`, `pred_sha256: fe94e7686ec581076a6526350ba0fa899c3fdb4ca53b65b2a851af5db59a1f20`).
- **Run 10 BF16 (`lora_eval.json`, `--max-new-tokens 448` with `HoloCycleGuardLogitsProcessor`):** `317/340` (`93.24%`) tool (`272/288` `tool_call`, `45/52` `prism_write`), `78/100` (`78.0%`) Chat, `45` total failures, `bars_passed: false` (`pred_sha256: dec044971a3ba886ed35eb94fd0eb69ce8b06a032c7beaea3738cb7b0c0b2663`; `parity_probe_export`: `576` positions, `vocab_size: 151936`, `logits_sha256: c9626c6bb03e1d960bbd86caae297a72dbfc4b6efd7153607f44b601b3304bb5`).
- **Cumulative [D90](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`, all `11` `down` entries from `2026-11-22T23:12:30Z` through `2026-11-24T04:13:21Z`):** Runs 1–9 `$2.5306` + `vast-62288296` (`414s @ $0.908/h = $0.1044`) + `vast-62289287` (`744s @ $0.521/h = $0.1077`) = **`$2.7427` spent (`$0.2573` remaining of `$3.00` cap)**. `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`.
- **Phase 5 (`Qwen3-1.7B`) verified `lexlean-r4` baseline evidence:** `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b` (`b35a6dcce1c527d91402a07dfa9ebd0928ee28f7`) and `Casey-allard/lexlean-r4@main:gpu/rtx4090_holo_v13_qwen3_1_7b_receipt.json` confirm pure-integer `UORR4V13` (`float_ops: 0`, `dense_matmul_ops: 0`, `total_bytes: 1,140,250,782`, `piece_count: 1088`, `top1_parity_pct: 91.67%`, `top5_parity_pct: 100.00%`, `mean_kl_nats: 0.033533`).

## Checks run at the head

- Pod run & local archive verification (`results_v2_run10.tar.gz`, `1.6M`, `wasm_float_audit.json`, `baseline_eval.json`, `lora_eval.json`, `train.log`, `lora_predictions.jsonl`) completed prior to `vast-62289287` deletion (`uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`).
- `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, and `uor-harness loop pr-check --local` run cleanly in `.worktrees/antigravity-studio-phase4d-run10-closeout` prior to push.

## Skills used

- `uor-pod` (with `UOR_POD_MAX_RATE=0.60` rate guard), `sha256sum`, Python offline prediction/ledger inspection, and `uor-harness` loop/ratchet verification.

## Still open

- Push `antigravity/studio-phase4d-run10-closeout`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`), post compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, and merge into `main`.
- **Owner [D90](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) amendment needed before Phase 5 (`Qwen3-1.7B`) GPU training:** D90 currently names `Qwen/Qwen3-0.6B` with a `$3.00` cumulative Vast.ai cap (`$2.7427` spent / `$0.2573` remaining), and [`antigravity-19/PREP.md`](../antigravity-19/PREP.md) closes `Qwen3-0.6B` at Run 10 (`no Run 11`). Transitioning to Phase 5 (`Qwen3-1.7B`) requires owner approval amending the model/budget cap (or continuing Studio on PR #4213 Phase 4A + external endpoint override until M4+).

## Plain summary

Completed Run 10 (`d90-foundry-lora-v10`, `v2.3`) on `vast-62289287` (`1× RTX 5090`, `$0.521/h`, `uptime_s: 744`, `$0.1077`) with zero active pods remaining (`Running: 0 pod(s)`) and `$2.7427` cumulative spend (`$0.2573` inside D90's `$3.00` cap): Bar #1 WAT (`compute_float_ops: 0`) and Bar #2 tool/build (`317/340` = `93.24%`) passed, while `HoloCycleGuardLogitsProcessor` cut loops (`8 -> 1`) and isolated Bar #3 Chat failure at `78/100` (`78.0%`, `val_loss_chat` climbing `3.188 -> 3.646` with unseen essay confabulations), cleanly halting before compile so PR #4213 Phase 4A stays untouched ahead of owner approval for Phase 5 (`Qwen3-1.7B`).
