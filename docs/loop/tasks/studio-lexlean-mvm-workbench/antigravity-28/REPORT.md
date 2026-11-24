# REPORT for D91 Run 7 Stop-Rule Negative & Qwen3-1.7B `v3.5` Run 8 Pre-Registration (`antigravity-28`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `0abc5a85d` (PR #4233, [`antigravity-27/REPORT.md`](../antigravity-27/REPORT.md); `ORDER.md` committed alone first at `c10188d80`).

## What did not work

- **D91 Run 7 (`d91-foundry-17b-v7`, `v3.5`, `vast-62341307`, `uptime_s: 1302` @ `$0.508/h` = `$0.1837`) passed Bar #1 WAT (`compute_float_ops: 0`), reached a Run 1–7 best BF16 Bar #2 (**`98.82%`** = `336/340`) and lowest `total_failures: 10`, then halted cleanly at Stop Gate #1 missing BF16 Bar #3 by 1 item (`94.0%` = `94/100 < 95.0%`):**
  - Lowering `--kl-anchor-weight` from `1.0` to `0.25` resolved Run 6's `1:1` SFT stall (`val_loss_chat: 2.78726 -> 3.07777`, `anchor_val_kl_first: 0.040735`, `<prism-write>: 52/52 = 100.0%`, `<tool_call>: 284/288`).
  - However, raising `--label-smoothing` from `0.02` to `0.05` across Qwen3's `151,936`-token bilingual vocabulary triggered **`2` partial-UTF-8 `U+FFFD` (`\ufffd`) byte-fallback glitches** inside English Chat sentences (`val_ch_016`: ``"so\ufffd\ufffd证 that file pieces"``, `val_ch_070`: ``"can\ufffd\ufffd证 output integrity"`` — both splitting `"验证"`), failing Bar #3 at `94/100` (`total_failures: 10`, also `val_ch_025, 027, 045` `missing_expected_keyword` and `val_ch_048` `too_few_words:37` [`36` words]).
  - An exact cross-run audit of `results_v3_run1..7/d91-out/lora_predictions.jsonl` confirmed **`0/2,200`** `\ufffd` across Runs 2–6 (`--label-smoothing 0.02`: `\ufffd = 0` in all 5 runs, `cjk = 10, 0, 3, 9, 0`) and `0/440` `\ufffd` (`cjk = 0`) in Run 1 (`--label-smoothing 0.05`, under-trained `r=16, epochs=1, train_loss=2.58583`) — `0/2,640` combined across Runs 1–6 (`2/880` across both `0.05` runs) — whereas Run 7 (`--label-smoothing 0.05`) emitted `2/440` `\ufffd` (`cjk = 10`, `\ufffd = 2`).
  - Run 7 halted cleanly at Stop Gate #1 before `recompile_holo_v13` (`run7_meta.json`: `stop_gate: "gate1_bf16"`, `active_mode: "none"`), archived diagnostics locally, and tore down (`Running: 0 pod(s)`).
- **Host environment caveat (`$0.00` spent in `antigravity-28`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `antigravity-28` verified all 6 unchanged script SHA-256s, `py_compile`, `eval_foundry_v13_v3.py --self-test`, cross-run `\ufffd` counts over Runs 1–7, and local unpacked `results_v3_run7/` JSON receipts while `Running: 0 pod(s)` remained verified.

## What changed

- **Zero code edits — exact untouched `v3/train_foundry_lora_v3_5.py` (`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`) and byte-identical data/eval toolchain (`$0.00`):**
  - Left `train_foundry_lora_v3.py`..`v3_5.py`, `generate_foundry_sft_v3.py`, `eval_foundry_v13_v3.py`, `foundry_schemas_v2.json`, `foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) 100% unchanged.
- **Pre-registered Run 8 (`d91-foundry-17b-v8`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-28/`):**
  - [`ORDER.md`](ORDER.md) (`4,085 B <= 4,096 B`, committed alone first at `c10188d80`).
  - [`PREP.md`](PREP.md): Pinned surgical Run 8 CLI tuning on untouched `v3/train_foundry_lora_v3_5.py` (`--label-smoothing 0.02` restoring Runs 2–6's `0/2,200` `\ufffd` setting [`0/2,640` across Runs 1–6], `--kl-anchor-weight 0.30`, `--weight-decay 0.025`, `--lr 1.35e-4` with exact `∫lr = 0.015187500`, and `--skip-baseline` reusing verified baseline `a1ba2937...`), explicit `--mode eval` & `--logits /workspace/d91-out/teacher_parity_logits.bin` on Bar #4b, unchanged Bars #1–#5, primary no-`--calib-file` + bounded `--calib-file` compile, `2.0h / <= $1.00` ceiling inside `$1.3621` remaining D91 headroom, and non-author `TEST FITNESS: FIT` signoff (`3d2214aee` by `597c4007-4c60-4e37-8a16-1d5e055c46cb`).

## Measurements

- **Archived Run 7 receipts (`local/antigravity/d90-foundry-lora/results_v3_run7/d91-out/`, `results_v3_run7.tar.gz` `sha256: a99457ae2c37812d54da740d412fa40f341c1c9b0788dbd1d6805f5afc475c61`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`. `run7_meta.json` records `native_cpu_build: true`, `host_ram_gb: 30`, `host_avail_gb: 27`, `nproc: 16`, `skip_baseline: true`; ephemeral pod compilation of `/workspace/d91-out/foundry_v13_eval_pack_v2_native` (`5.2M`, `AMD Ryzen 7 7800X3D` AVX-512) and `/workspace/d91-out/recompile_holo_v13` (`4.6M`) ran without altering `.wasm` (neither Run 7 pod build logs nor the compiled binaries were archived locally).
  - `lora_eval.json` (`pred_sha256: 4052852bfe56210cd8d7c0ca15f3296436a73f679c73e94a795db199ea491dea`): `tool_pass: 336/340` (**`98.82%`**: `284/288` `tool_call`, `52/52` = `100.0%` `prism_write`), `chat_pass: 94/100` (`94.0%`), `total_failures: 10`, `bars_passed: false` (Stop Gate #1 FAIL).
  - `train.log` (`34,865,152 / 1,755,440,128 = 1.9861%`): `0.50ep` (`train=3.13932, val=1.47818, tool=0.44296, chat=2.53517, anc_tr=0.131449, anc_val_resp=0.126785, anc_val_first=0.092487`) -> `0.75ep` (`1.99016, 1.54173, 0.47944, 2.62635, 0.085346, 0.093553, 0.040930`) -> `1.00ep` (`1.74792, 1.66764, 0.51283, 2.84674, 0.034231, 0.088023, 0.047005`) -> `1.25ep` (`1.39924, 1.80351, 0.54447, 3.08903, 0.022955, 0.086794, 0.039975`) -> `1.50ep` (`batch=900/900, opt_step=225/225, lr=0.0, train=1.35823, val=1.79658, tool=0.54178, chat=3.07777, anc_tr=0.020531, anc_val_resp=0.086443, anc_val_first=0.040735`, restored); `[delta_w_rel_fro]`: `overall_mean=0.011328` (`down=0.006639, gate=0.019307, up=0.014073, q=0.009513, k=0.010048, v=0.009018, o=0.010699`).
  - `parity_probe_tokens.json`: `total_positions: 576`, `vocab_size: 151936`, `logits_sha256: a5a5f990a4a3afefbc41ad6c2d32fe7ad9478e152827218e5fcac6bb348f20b5` (`350,060,544` B).
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Runs 1–6 (`$1.4542`) + Run 7 `vast-62341307` (`$0.508/h`, `uptime_s: 1302`): **`$0.1837`** = **`$1.6379` cumulative (`$1.3621` remaining of D91's `+$3.00` cap)**; `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.

## Checks run at the head

- Offline pinned script `shasum -a 256` verification (`6/6` matching `PREP.md`) — PASS.
- Cross-run `results_v3_run1..7` `\ufffd` / CJK audit (`0/2,200` `\ufffd` across Runs 2–6 at `0.02`, `0/440` in Run 1 at `0.05`, `0/2,640` Runs 1–6 vs `2/440` in Run 7) — PASS.
- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_5.py` — PASS (`0` errors).
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `0` overlap, `100.0%` gold).
- `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status` (from main checkout) — `Running: 0 pod(s) ($0/h of $8.00/h)`.
- `597c4007-4c60-4e37-8a16-1d5e055c46cb` (`pro` non-author FIT review at `3d2214aeee0354e257f1f03269b27e7fb528fc0e`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` inside `.worktrees/antigravity-studio-phase5-d91-r8-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Complete non-author `pro` FIT review on `PREP.md`, commit signoff, push `antigravity/studio-phase5-d91-r8-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1`, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, enqueue via `gh pr merge --squash`, pull fresh `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 8 (`d91-foundry-17b-v8`) GPU execution.

## Plain summary

Closed out D91 Run 7 (`d91-foundry-17b-v7`, `vast-62341307`, `$0.1837` spent; `$1.6379` cumulative of `+$3.00`, `$1.3621` remaining, `0` pods active), where `v3.5` (`--kl-anchor-weight 0.25 --label-smoothing 0.05`) passed Bar #1 (`compute_float_ops: 0`), reached a Run 1–7 peak BF16 Bar #2 (**`98.82%`** = `336/340`, `52/52` `<prism-write>`) and lowest `total_failures: 10`, while halting cleanly at Stop Gate #1 on BF16 Bar #3 (`94.0%` = `94/100 < 95.0%`) because `--label-smoothing 0.05` induced `2` partial-UTF-8 `\ufffd` (`验证`) BPE glitches (`val_ch_016`, `val_ch_070`) vs `0/2,200` `\ufffd` across Runs 2–6 at `0.02` (`0/440` in Run 1 at `0.05`; `0/2,640` across Runs 1–6). Pre-registered Run 8 (`d91-foundry-17b-v8`, `<= $1.00 / 2.0h`) on untouched `v3/train_foundry_lora_v3_5.py` with `--label-smoothing 0.02 --kl-anchor-weight 0.30 --weight-decay 0.025 --lr 1.35e-4 --skip-baseline`.
