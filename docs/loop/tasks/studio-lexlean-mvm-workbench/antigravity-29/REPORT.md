# REPORT for D91 Run 8 Stop-Rule Negative & Final `v3.5` Run 9 Pre-Registration (`antigravity-29`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `76dcf0f1e` (PR #4234, [`antigravity-28/REPORT.md`](../antigravity-28/REPORT.md); `ORDER.md` committed alone first at `9418a1753`).

## What did not work

- **D91 Run 8 (`d91-foundry-17b-v8`, `v3.5`, `vast-62347334`, `uptime_s: 1229` @ `$0.508/h` = `$0.1734`) passed Bar #1 WAT (`compute_float_ops: 0`) and BF16 Bar #2 (`97.65%` = `332/340`), then halted cleanly at Stop Gate #1 on BF16 Bar #3 (`91.0%` = `91/100 < 95.0%`), falsifying the `--label-smoothing 0.05`-only hypothesis:**
  - Reverting `--label-smoothing` from `0.05` back to `0.02` with `--kl-anchor-weight 0.30 --weight-decay 0.025 --lr 1.35e-4` lowered `train_loss` to `0.86049` (`val_loss_tool=0.51793`, `val_loss_chat=3.16485`, `delta_w_rel_fro=0.010622`), while keeping `anchor_val_kl_first` weaker at `0.055465` (`10` CJK rows across `440` validation predictions).
  - Run 8 still hit **`1` partial-UTF-8 `U+FFFD` (`\ufffd`) glitch** on `val_ch_016` (``"so inspectors can\ufffd\ufffd证 recomputing"`` — the same `"验证"` split seen in Run 7), alongside **`3` `<tool_call>` leaks inside Chat prose** (`val_ch_070`, `val_ch_076`, `val_ch_078`) and **`5` `missing_expected_keyword`** misses (`val_ch_025, 027, 035, 041, 087`), totaling `17` failures across all `440` validation items.
  - Across Runs 1–8 (`results_v3_run1..8/d91-out/lora_predictions.jsonl`, `440` rows/run), `\ufffd` counts are `0, 0, 0, 0, 0, 0, 2, 1` and CJK row counts are `0, 10, 0, 3, 9, 0, 10, 10`: weak/unanchored LoRA runs (`kl=0` in Runs 2/4/5 or `kl=0.25/0.30` in Runs 7–8) leak Chinese bilingual tokens (`cjk = 3..10`), whereas Run 6 (`kl_anchor_weight=1.0, weight_decay=0.015, label_smoothing=0.02`) achieved **`0` CJK rows, `0` `\ufffd`, `anchor_val_kl_first = 0.021992`, `delta_w_rel_fro = 0.008689`, and lowest `val_loss_chat = 2.78726`**, stalling at `91.0%` Chat only because `1.5` epochs (`train_loss = 1.22854`) was too short against `1:1` KL regularization.
  - Run 8 halted cleanly at Stop Gate #1 before `recompile_holo_v13` (`run8_meta.json`: `stop_gate: "gate1_bf16"`, `active_mode: "none"`), archived diagnostics locally, and tore down (`Running: 0 pod(s)`).
- **Host environment caveat (`$0.00` spent in `antigravity-29`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `antigravity-29` verified all 6 unchanged script SHA-256s, `py_compile`, `eval_foundry_v13_v3.py --self-test`, cross-run `\ufffd`/CJK counts across Runs 1–8, and local unpacked `results_v3_run8/` JSON receipts while `Running: 0 pod(s)` remained verified.

## What changed

- **Zero code edits — exact untouched `v3/train_foundry_lora_v3_5.py` (`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`) and byte-identical data/eval toolchain (`$0.00`):**
  - Left `train_foundry_lora_v3.py`..`v3_5.py`, `generate_foundry_sft_v3.py`, `eval_foundry_v13_v3.py`, `foundry_schemas_v2.json`, `foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) 100% unchanged.
- **Pre-registered final D91 configuration Run 9 (`d91-foundry-17b-v9`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-29/`):**
  - [`ORDER.md`](ORDER.md) (`4,027 B <= 4,096 B`, committed alone first at `9418a1753`).
  - [`PREP.md`](PREP.md): Restores Run 6's strong KL/weight-decay/smoothing settings (`--kl-anchor-weight 1.0 --weight-decay 0.015 --label-smoothing 0.02`) and raises SFT budget to `--epochs 2.5 --lr 1.4e-4 --skip-baseline` (`T = 375`, `W = 30`, exact `∫lr = 0.026250000` = `1.7949×` Run 6's `0.014625000`), keeps unchanged Bars #1–#5, primary no-`--calib-file` + bounded `--calib-file` fallback (`< $0.55`), `2.0h / <= $1.00` ceiling inside `$1.1887` remaining D91 headroom, explicit finality rule (no Run 10 absent an instrument defect; close D91 via §5.1 / D68 review and return to core native M2+ if Run 9 hits any Stop Gate), and non-author `TEST FITNESS: FIT` signoff (`1c93d9665` by `bc382ed2-c24c-4002-9fb0-5d62abb31577`).

## Measurements

- **Archived Run 8 receipts (`local/antigravity/d90-foundry-lora/results_v3_run8/d91-out/`, `results_v3_run8.tar.gz` `sha256: 7f11d887cbcb2963223b760513a68420f735cdadc9a731d9bf5805b4279c0ffa`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`. `pod_build.log` (`9,974 B`) and `job_gpu0.log` (`10,974 B`) archive pod toolchain/binary builds (`foundry_v13_eval_pack_v2_native` `5.2M`, `recompile_holo_v13` `4.6M`) with `.wasm` verified unchanged afterward (ELF binaries not archived locally); `run8_meta.json`: `native_cpu_build: true`, `host_ram_gb: 30`, `host_avail_gb: 28`, `nproc: 16`, `skip_baseline: true`, `stop_gate: "gate1_bf16"`.
  - `lora_eval.json` (`pred_sha256: f7c539eaa3316f92cf66ec527aa69123402137f0f92f3f65cf52a683e063799c`): `tool_pass: 332/340` (`97.65%`: `284/288` `tool_call`, `48/52` `prism_write`), `chat_pass: 91/100` (`91.0%`), `total_failures: 17`, `bars_passed: false` (Stop Gate #1 FAIL).
  - `train.log` (`34,865,152 / 1,755,440,128 = 1.9861%`): `0.50ep` (`train=2.61004, val=1.45358, tool=0.41020, chat=2.51890, anc_tr=0.121892, anc_val_resp=0.118462, anc_val_first=0.113603`) -> `0.75ep` (`1.52362, 1.52625, 0.44735, 2.62782, 0.084717, 0.094858, 0.052169`) -> `1.00ep` (`1.27237, 1.66137, 0.48570, 2.86176, 0.034845, 0.085661, 0.057008`) -> `1.25ep` (`0.90569, 1.82763, 0.51971, 3.16305, 0.022935, 0.086700, 0.050071`) -> `1.50ep` (`batch=900/900, opt_step=225/225, lr=0.0, train=0.86049, val=1.82762, tool=0.51793, chat=3.16485, anc_tr=0.020788, anc_val_resp=0.087008, anc_val_first=0.055465`, restored); `[delta_w_rel_fro]`: `overall_mean=0.010622` (`down=0.006062, gate=0.018294, up=0.013247, q=0.008939, k=0.009316, v=0.008445, o=0.010052`).
  - `parity_probe_tokens.json`: `total_positions: 576`, `vocab_size: 151936`, `logits_sha256: 7a865b69ead9dea0511c93c090a3f6de5c7e68e46655f56a71550d9a1f3f3550` (`350,060,544` B).
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Runs 1–7 (`$1.6379`) + Run 8 `vast-62347334` (`$0.508/h`, `uptime_s: 1229`): **`$0.1734`** = **`$1.8113` cumulative (`$1.1887` remaining of D91's `+$3.00` cap)**; `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.

## Checks run at the head

- Offline pinned script `shasum -a 256` verification (`6/6` matching `PREP.md`) — PASS.
- Cross-run `results_v3_run1..8` `\ufffd` / CJK audit (`\ufffd`: `[0, 0, 0, 0, 0, 0, 2, 1]`; `cjk`: `[0, 10, 0, 3, 9, 0, 10, 10]` across `440` rows/run) — PASS.
- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_5.py` — PASS (`0` errors).
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `0` overlap, `100.0%` gold).
- `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status` (from main checkout) — `Running: 0 pod(s) ($0/h of $8.00/h)`.
- `bc382ed2-c24c-4002-9fb0-5d62abb31577` (`pro` non-author FIT review at `1c93d96655c50676f4ce00d11a1e531bdfac301f`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local`, and `cargo run -q --locked -p uor-r4-claim-wording -- .` inside `.worktrees/antigravity-studio-phase5-d91-r9-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Obtain fresh-context `pro` exact-head `VERDICT: APPROVE` on PR #4235, post compliant `[antigravity] uor-review v1`, confirm `uor-harness loop pr-check --pr 4235` & GitHub CI green, enqueue via `gh pr merge 4235 --squash`, pull fresh `main`, clean worktree/branch, and hand back via `enter_plan_mode` for final Run 9 (`d91-foundry-17b-v9`) GPU execution.

## Plain summary

Closed out D91 Run 8 (`d91-foundry-17b-v8`, `vast-62347334`, `$0.1734` spent; `$1.8113` cumulative of `+$3.00`, `$1.1887` remaining, `0` pods active), where reverting `--label-smoothing 0.02` with `--kl-anchor-weight 0.30` passed Bar #1 (`compute_float_ops: 0`) and BF16 Bar #2 (`97.65%` = `332/340`), while halting at Stop Gate #1 on BF16 Bar #3 (`91.0%` = `91/100 < 95.0%`) with `3` `<tool_call>` Chat leaks and `1` `\ufffd` (`验证`) glitch (`val_ch_016`), falsifying the `0.05`-only hypothesis. Pre-registered final D91 configuration Run 9 (`d91-foundry-17b-v9`, `<= $1.00 / 2.0h`) on untouched `v3/train_foundry_lora_v3_5.py` restoring Run 6's `0`-CJK / `0`-`\ufffd` regularizers (`--kl-anchor-weight 1.0 --weight-decay 0.015 --label-smoothing 0.02`) at `--epochs 2.5 --lr 1.4e-4 --skip-baseline` (`∫lr = 0.026250000`).
