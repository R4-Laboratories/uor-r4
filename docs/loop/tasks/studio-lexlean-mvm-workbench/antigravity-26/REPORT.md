# REPORT for D91 Run 5 Stop-Rule Negative & Qwen3-1.7B `v3.5` Pre-Registration (`antigravity-26`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `04cd332c7` (PR #4231, [`antigravity-25/REPORT.md`](../antigravity-25/REPORT.md); `ORDER.md` committed alone first at `e5938ecdc`).

## What did not work

- **D91 Run 5 (`d91-foundry-17b-v5`, `v3.4`, `vast-62322100`, `$0.4321` spent) passed Bar #1 WAT and BOTH BF16 Stop Gate #1 bars (**`97.35%`** tool / **`98.0%`** Chat), then halted cleanly at Stop Gate #2 (`Bar #4a mean_kl_nats > 0.08`) on BOTH `--calib-file` (`0.197846`) and no-`--calib-file` (`0.243437`) `UORR4V13` builds:**
  - `v3.4` (`r=32, alpha=64, dropout=0.05, wd=0.015, ls=0.02, epochs=1.5, lr=1.3e-4, ∫lr=0.014625`, cooled `1.50 ep` restored at `lr=0.0`) solved BF16 completely on held-out `val.jsonl` (`N=440`): **`331/340` (`97.35%`: `280/288` `<tool_call>`, `51/52` `<prism-write>`)** Tool/Build and **`98/100` (`98.0%`)** Chat (`total_failures: 11`, zero loop/length/terminator/forbidden-tag errors).
  - However, primary `recompile_holo_v13` with `--calib-file calib_foundry.txt` reached `top1 = 75.00%`, `top5 = 100.00%`, `mean_kl_nats = 0.197846 > 0.08`, and the pre-registered no-`--calib-file` compile reached `top1 = 83.33%`, `top5 = 100.00%`, `mean_kl_nats = 0.243437 > 0.08` (`ood_06_rope=0.8891`, `ood_05_js=0.5470`, `ood_02_rust=0.4651`, `ood_07_git=0.2261`, `ood_12_chat=0.1810`, `ood_09_sql=0.1438`, `ood_04_opfs=0.1388`), halting before Bar #4b, compiled `N=440` eval, and `--mode pack`.
  - **Root cause:** Comparing per-prompt OOD KL against un-tuned `Qwen/Qwen3-1.7B` (`mean_kl_nats = 0.033533`, `7/12` prompts at `0.0000`) and `0.6B diag_v5` (`0.045925`), `v3.4` trained `2,304/2,400` (`96.0%`) rows on `FOUNDRY_SYSTEM_PROMPT` alongside `96` (`4.0%`) synthetic CS/tech SFT rows on `"You are a helpful assistant."` (`calib_foundry.txt` samples `5/96` user strings from `"You are a helpful assistant."` rows and `91/96` from Foundry rows, wrapped under `format_qwen3_prompt()` at compile time) with zero frozen-base KL anchoring, shifting general-assistant OOD phrasing/distributions (`"payload payload"`, `"header range header"`, `ood_06_rope = 0.8891`) even without Foundry calibration tokens (`0.243437`).
- **Baseline `x86_64` scalar CPU eval bottleneck observed on `vast-62322100` (`AMD Ryzen 7 7800X3D`):**
  - During the bounded diagnostic parity probe on `v13_pieces_raw` (`PID 2295`), `foundry_v13_eval_pack_v2` compiled with default `x86_64` baseline flags took `> 14 min` at `99.9%` of 1 core (`~2.83 GiB` RSS) for `6,625` probe tokens (`results_v3_run5/d91-out/parity_probe_tokens.json`, `576` eval positions) and was stopped (`kill 2295`) to unblock GPU 0 for the no-calib compile. Rebuilding unmodified `foundry_v13_eval_pack_v2` with `RUSTFLAGS="-C target-cpu=native"` into `/workspace/target-native` completed cleanly on pod (`1.1 MiB`, unlocking `AVX-512F / AVX2 / VNNI`) ahead of Run 6 while keeping `.wasm` and `.rs` sources untouched.
- **Host test environment caveat (`$0.00` spent in `antigravity-26`):**
  - Laptop system `python3` has no GPU `torch`/`transformers`/`peft`, so `train_foundry_lora_v3_5.py` host unit checks ran against numpy-backed stubs while `Running: 0 pod(s)` is verified here (`$0.00` spent in this PR).

## What changed

- **Built & offline-verified `v3/train_foundry_lora_v3_5.py` (`$0.00`, untracked local toolchain only — zero repo code or scripts committed):**
  - Left `train_foundry_lora_v3.py`..`v3_4.py`, `generate_foundry_sft_v3.py`, `eval_foundry_v13_v3.py`, `foundry_schemas_v2.json`, `foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) **100% byte-identical**.
  - Added zero-leakage general self-distillation KL anchor in `v3/train_foundry_lora_v3_5.py` (`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`): `--kl-anchor-weight 1.0 --kl-anchor-batch 2 --kl-anchor-tokens 32`, `64` unique train + `16` unique val `"You are a helpful assistant."` prompts enforced disjoint from `OOD_FORBIDDEN_SUBSTRINGS` by `assert_anchor_prompts_disjoint()`, frozen-base rollouts via `build_anchor_sequences()`, exact FP32 forward `KL(p_base || p_lora)` via `model.disable_adapter()` on response positions (`prompt_len - 1 .. n - 2`), token/prompt-weighted validation KL telemetry (`anchor_val_kl_resp`, `anchor_val_kl_first`), and `compute_delta_w_norms()` logging (`[delta_w_rel_fro]`).
- **Pre-registered Run 6 (`d91-foundry-17b-v6`) under Charter §4.5 (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-26/`):**
  - [`ORDER.md`](ORDER.md) (`3,640 B <= 4,096 B`, committed alone first at `e5938ecdc`).
  - [`PREP.md`](PREP.md): Pinned `d91-foundry-17b-v6`, unchanged Bars #1–#5 (`--max-new-tokens 448` on `--mode eval`, `--mirror-url ...weights-v13-qwen3-1.7b-foundry` on `--mode pack`), primary no-`--calib-file` compile + bounded `--calib-file` fallback, `RUSTFLAGS="-C target-cpu=native"` untouched-source native eval binary, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `2.0h / <= $1.10`, and non-author `TEST FITNESS: FIT` signoff (`75904ba77` by `14b01f27-e3af-466e-87a0-3be250ae165d`).

## Measurements

- **Archived Run 5 receipts (`local/antigravity/d90-foundry-lora/results_v3_run5/d91-out/`, `results_v3_run5.tar.gz` `sha256: 7fecc3834e9ca99d58d254d710e26626a9a94d068832d689b256a77a72d364f4`):**
  - `wasm_float_audit.json`: `wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`.
  - `baseline_eval.json` (`pred_sha256: a1ba2937...`): `tool_pass: 168/340` (`49.41%`: `167/288` `tool_call`, `1/52` `prism_write`), `chat_pass: 43/100` (`43.0%`).
  - `lora_eval.json` (`pred_sha256: 55f43eefdd61f5cbed58fbe57268f3bba546263596131801e851e3f5626b0fde`): `tool_pass: 331/340` (**`97.35%`**: `280/288` `tool_call`, `51/52` `prism_write`), `chat_pass: 98/100` (**`98.0%`**), `total_failures: 11`, **`bars_passed: true`**.
  - `train.log` (`34,865,152 / 1,755,440,128 = 1.9861%`): `0.50ep` (`1.44699, tool=0.41451, chat=2.50118`) -> `0.75ep` (`1.55537, 0.45883, 2.67496`) -> `1.00ep` (`1.70990, 0.49019, 2.95525`) -> `1.25ep` (`1.88482, 0.51917, 3.27918`) -> cooled `1.50ep` (`batch=900/900, opt_step=225/225, lr=0.0, train_loss=0.77225, val=1.87200, tool=0.51773, chat=3.25474, restored`).
  - `foundry_v13_receipt.json` (`--calib-file`, `0bfdb7c9...`): `float_ops: 0`, `dense_matmul_ops: 0`, `total_bytes: 1,316,411,550`, `piece_count: 1256`, `top1_parity_pct: 75.00%`, `top5_parity_pct: 100.00%`, `mean_kl_nats: 0.197846`.
  - `foundry_v13_receipt_nocalib.json` (no `--calib-file`, `5e267786...`): `float_ops: 0`, `dense_matmul_ops: 0`, `total_bytes: 1,316,411,550`, `piece_count: 1256`, `top1_parity_pct: 83.33%`, `top5_parity_pct: 100.00%`, `mean_kl_nats: 0.243437`.
- **Cumulative D91 ledger (`~/.local/share/uor-r4/compute/ledger.jsonl`):**
  - Runs 1–4 (`$0.7959`) + Run 5 `vast-62322100` (`$0.508/h`, `uptime_s: 3062`): **`$0.4321`** = **`$1.2280` cumulative (`$1.7720` remaining of D91's `+$3.00` cap)**; `uor-pod status`: `Running: 0 pod(s) ($0/h of $8.00/h)`.
- **Pinned `v3.5` SHA-256:** `v3/train_foundry_lora_v3_5.py`: `5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f` (all 10 sibling scripts & `val.jsonl` `06814bd9...` unchanged).

## Checks run at the head

- `python3 -B -m py_compile local/antigravity/d90-foundry-lora/v3/train_foundry_lora_v3_5.py` — PASS (`0` errors).
- `train_foundry_lora_v3_5.py` offline host unit suite (`assert_anchor_prompts_disjoint` + forbidden/duplicate/overlap negative tests, zero overlap against all `2400` train / `440` val / `96` calib rows, `build_anchor_sequences` EOS trimming & padding, `_prepare_anchor_tensors` / `compute_kl_anchor_loss` / `evaluate_anchor_val_kl` exact mask alignment + `0.0` identity + analytic FP32 `KL` parity + `model.train()` restoration, `compute_delta_w_norms` analytical check, `encode_sft_episode` `chat` masking, `(600, 1.5, 4)` cosine integral `0.014625`, `select_best_checkpoint`, `HoloCycleGuardLogitsProcessor`) — PASS (`V3_5_UNIT_SUITE_PASS`).
- `python3 -B local/antigravity/d90-foundry-lora/v3/eval_foundry_v13_v3.py --self-test` — PASS (`2400/440`, `0` overlap, `100.0%` gold).
- `14b01f27-e3af-466e-87a0-3be250ae165d` (`pro` non-author FIT review at `75904ba77b45523ecc2e99e936429198dd3b9d72`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` verified inside `.worktrees/antigravity-studio-phase5-d91-r6-prep`.

## Skills used

- Standard git, `python3 -B`, `uor-pod`, and `uor-harness` loop/ratchet verification.

## Still open

- Push `antigravity/studio-phase5-d91-r6-prep`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1`, confirm `uor-harness loop pr-check --pr <N>` & GitHub CI green, enqueue via `gh pr merge --squash`, pull fresh `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 6 (`d91-foundry-17b-v6`) GPU execution.

## Plain summary

Closed out D91 Run 5 (`d91-foundry-17b-v5`, `vast-62322100`, `$0.4321` spent; `$1.2280` cumulative of `+$3.00`, `$1.7720` remaining, `0` pods active), where `v3.4` passed Bar #1 (`compute_float_ops: 0`), BF16 Bar #2 (**`97.35%`** = `331/340`), and BF16 Bar #3 (**`98.0%`** = `98/100`), halting cleanly at Stop Gate #2 (`Bar #4a mean_kl_nats > 0.08`) on both `--calib-file` (`0.197846`) and no-`--calib-file` (`0.243437`) compiles. Diagnosed unanchored general-prior OOD drift (`96.0%` Foundry SFT / `4.0%` synthetic `"You are a helpful assistant."` SFT with zero frozen-base KL anchoring) plus baseline `x86_64` scalar CPU eval slowness, built and offline-verified KL-anchored `v3/train_foundry_lora_v3_5.py` (`64` disjoint train + `16` disjoint val general anchor prompts), and pre-registered `d91-foundry-17b-v6`.
