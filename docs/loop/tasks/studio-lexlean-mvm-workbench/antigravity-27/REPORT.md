# REPORT for D91 Run 6 Stop-Rule Negative & Qwen3-1.7B `v3.5` Tuned Pre-Registration (`antigravity-27`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `890dd10be` (PR #4232, [`antigravity-26/REPORT.md`](../antigravity-26/REPORT.md); `ORDER.md` committed alone first at `ebadf3b6f`).

## What did not work

- **D91 Run 6 (`d91-foundry-17b-v6`, `v3.5`, `vast-62334503`, `uptime_s: 1603` @ `$0.508/h` = `$0.2262`) passed Bar #1 WAT (`compute_float_ops: 0`), confirmed KL anchoring (`anchor_val_kl_first: 0.021992`), and passed BF16 Bar #2 (`97.35%` = `331/340`), then halted cleanly at Stop Gate #1 on BF16 Bar #3 (`91.0%` = `91/100 < 95.0%`):**
  - The self-distillation KL anchor worked mechanistically (`anchor_kl_train`: `0.119180 -> 0.016234`, `anchor_val_kl_resp`: `0.108210 -> 0.072151`, `anchor_val_kl_first`: `0.042179 -> 0.021992`, `overall_mean delta_w_rel_fro = 0.008689`), and `<tool_call>` reached a Run 1–6 high of **`285/288` (`99.0%`)** alongside `46/52` `<prism-write>` (`331/340 = 97.35%` Bar #2 PASS).
  - However, `--kl-anchor-weight 1.0` imposed a `1:1` brake against SFT gradient updates: `train_loss` stopped at **`1.22854`** (vs Run 5's `0.77225`) and `val_loss_chat` stalled at **`2.78726`** (nearly matching underfitted Run 4's `2.78850` / `93.0%` Chat). BF16 Chat dropped from `98.0%` (Run 5) to **`91/100` (`91.0%`)** (`bars_passed: false`, `total_failures: 18`: `6` `missing_expected_keyword` — `val_ch_025, 028, 067, 081, 087, 093`, including `val_ch_087` reverting to ``Foundry's WebAssembly `v13` runtime enforces strict IEEE `64`-bit floating-point execution`` — plus `val_ch_010` `low_unique_ratio:0.327`, `val_ch_062` `too_few_words:55`, `val_ch_080` `forbidden_tag:<tool_call>`).
  - Run 6 stopped at Stop Gate #1 without running `recompile_holo_v13` (`run6_meta.json`: `stop_gate: "gate1_bf16"`, `active_mode: "none"`), saving downstream compile spend and tearing down cleanly (`Running: 0 pod(s)`).
- **Host environment caveat (`$0.00` spent in `antigravity-27`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `antigravity-27` verified all 6 unchanged pinned script SHA-256s, `py_compile`, `eval_foundry_v13_v3.py --self-test`, and local unpacked `results_v3_run6/` JSON receipts while `Running: 0 pod(s)` remained verified.

## What changed

- **Zero code edits — exact untouched `v3/train_foundry_lora_v3_5.py` (`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`) and byte-identical data/eval toolchain (`$0.00`):**
  - Left `train_foundry_lora_v3.py`..`v3_5.py`, `generate_foundry_sft_v3.py`, `eval_foundry_v13_v3.py`, `foundry_schemas_v2.json`, `foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) 100% unchanged.
- **Pre-registered Run 7 (`d91-foundry-17b-v7`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-27/`):**
  - [`ORDER.md`](ORDER.md) (`3,772 B <= 4,096 B`, committed alone first at `ebadf3b6f`).
  - [`PREP.md`](PREP.md): Pinned controlled 4-knob CLI tuning for `train_foundry_lora_v3_5.py` (`--kl-anchor-weight 0.25` for `4:1` SFT dominance to clear `>= 95.0%` Chat, `--label-smoothing 0.05` guided by the matched `0.6B` `diag_v3_i8` [`0.0`] vs `diag_v5` [`0.05`] comparison in [`antigravity-10/DECISION.md`](../antigravity-10/DECISION.md) where `0.05` cut `UORR4V13` Bar #4a KL `-38.8%` [`0.075072 -> 0.045925`] and lifted compiled tool `90.62% -> 97.19%` [hypothesis on `1.7B` `0.02 -> 0.05`], `--weight-decay 0.025`, `--lr 1.4e-4` with exact `∫lr = 0.015750000`, and `--skip-baseline` reusing verified baseline `a1ba2937...`), fixed Run 7 parity logits path `--logits /workspace/d91-out/teacher_parity_logits.bin` and explicit `--mode eval`, unchanged Bars #1–#5, primary no-`--calib-file` + bounded `--calib-file` compile, `2.0h / <= $1.00` ceiling inside `$1.5458` remaining D91 headroom, and non-author `TEST FITNESS: FIT` signoff (`13d31c562` by `731395cd-7c69-4122-b0f3-458a15299c63`).

## Measurements

- **Archived Run 6 receipts (`local/antigravity/d90-foundry-lora/results_v3_run6/d91-out/`, `results_v3_run6.tar.gz` `sha256: 3d39fff542fc4e4544b3d73be47fe16fd02548ed86b59eeb2dc8c1e5405e0439`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`. Note: `run6_meta.json` records `native_cpu_build: true`, `host_ram_gb: 30`, `host_avail_gb: 28`, `nproc: 16`, while pod build logs verified `/workspace/d91-out/foundry_v13_eval_pack_v2_native` (`5.2M`, `RUSTFLAGS="-C target-cpu=native"` on `AMD Ryzen 7 7800X3D` AVX-512) and `/workspace/d91-out/recompile_holo_v13` (`4.6M`) without altering `.wasm` (the compiled binaries themselves were not archived locally).
  - `baseline_eval.json` (`pred_sha256: a1ba29374053440d2ee4e43f0b120d7f255deeab1b0160488ac9d9cf891f8abd`): `tool_pass: 168/340` (`49.41%`: `167/288` `tool_call`, `1/52` `prism_write`), `chat_pass: 43/100` (`43.0%`).
  - `lora_eval.json` (`pred_sha256: 604e5c7cc7a9e466239111f41024d7aa462eaffea39442f1ad1e8df151d716db`): `tool_pass: 331/340` (`97.35%`: `285/288` `tool_call`, `46/52` `prism_write`), `chat_pass: 91/100` (`91.0%`), `total_failures: 18`, `bars_passed: false` (Stop Gate #1 FAIL).
  - `train.log` (`34,865,152 / 1,755,440,128 = 1.9861%`): `0.50ep` (`train=2.75563, val=1.49387, tool=0.41159, chat=2.59891, anc_tr=0.119180, anc_val_resp=0.108210, anc_val_first=0.042179`) -> `0.75ep` (`1.81490, 1.47844, 0.42726, 2.55173, 0.091621, 0.091059, 0.033859`) -> `1.00ep` (`1.62367, 1.53408, 0.45469, 2.63616, 0.045729, 0.080716, 0.029134`) -> `1.25ep` (`1.27443, 1.62290, 0.48549, 2.78423, 0.021430, 0.071893, 0.023764`) -> `1.50ep` (`batch=900/900, opt_step=225/225, lr=0.0, train=1.22854, val=1.62309, tool=0.48289, chat=2.78726, anc_tr=0.016234, anc_val_resp=0.072151, anc_val_first=0.021992`, restored); `[delta_w_rel_fro]`: `overall_mean=0.008689` (`down=0.004881, gate=0.013676, up=0.01119, q=0.007431, k=0.00743, v=0.007461, o=0.008751`).
  - `parity_probe_tokens.json`: `total_positions: 576`, `vocab_size: 151936`, `logits_sha256: 3fb5bc50bc9f7fb4162a7cf9fd1b34f71b182ddcb343fd24bca68072e889f88f` (`350,060,544` B).
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Runs 1–5 (`$1.2280`) + Run 6 `vast-62334503` (`$0.508/h`, `uptime_s: 1603`): **`$0.2262`** = **`$1.4542` cumulative (`$1.5458` remaining of D91's `+$3.00` cap)**; `uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.

## Checks run at the head

- Offline pinned script `shasum -a 256` verification (`6/6` matching `PREP.md`) — PASS.
- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_5.py` — PASS (`0` errors).
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `0` overlap, `100.0%` gold).
- `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status` (from main checkout) — `Running: 0 pod(s) ($0/h of $8.00/h)`.
- `731395cd-7c69-4122-b0f3-458a15299c63` (`pro` non-author FIT review at `13d31c562466737bb590881a5ebec74567f5796f`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` inside `.worktrees/antigravity-studio-phase5-d91-r7-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Complete non-author `pro` FIT review on `PREP.md`, commit signoff, push `antigravity/studio-phase5-d91-r7-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1`, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, enqueue via `gh pr merge --squash`, pull fresh `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 7 (`d91-foundry-17b-v7`) GPU execution.

## Plain summary

Closed out D91 Run 6 (`d91-foundry-17b-v6`, `vast-62334503`, `$0.2262` spent; `$1.4542` cumulative of `+$3.00`, `$1.5458` remaining, `0` pods active), where `v3.5` (`--kl-anchor-weight 1.0`) passed Bar #1 (`compute_float_ops: 0`), reduced `anchor_val_kl_first` to `0.021992`, and passed BF16 Bar #2 (**`97.35%`** = `331/340`), before halting cleanly at Stop Gate #1 on BF16 Bar #3 (**`91.0%`** = `91/100 < 95.0%`) due to `1:1` KL regularization over-braking SFT (`train_loss = 1.22854`, `val_loss_chat = 2.78726`). Pre-registered Run 7 (`d91-foundry-17b-v7`, `<= $1.00 / 2.0h`) on untouched `v3/train_foundry_lora_v3_5.py` with controlled CLI tuning (`--kl-anchor-weight 0.25 --label-smoothing 0.05 --weight-decay 0.025 --lr 1.4e-4 --skip-baseline`).
