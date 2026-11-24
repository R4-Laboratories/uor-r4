# PREP Sheet for Qwen3-1.7B `v3.5` Tuned KL-Anchored Foundry LoRA (`d91-foundry-17b-v7`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-26/PREP.md`](../antigravity-26/PREP.md) & [`antigravity-26/REPORT.md`](../antigravity-26/REPORT.md) (PR #4232, `890dd10be`), [`antigravity-25/REPORT.md`](../antigravity-25/REPORT.md) (PR #4231, `04cd332c7`), [`antigravity-10/DECISION.md`](../antigravity-10/DECISION.md) (`0.6B diag_v5` `label_smoothing=0.05` + D68 Bar #4 split)
- Owner decision: [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`, extending Studio-only D90 exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap, tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger; followed by returning to core native UOR-R4 M2+ milestones)
- D91 cumulative ledger: **`$1.4542` spent (`$1.5458` remaining of fresh `+$3.00` D91 cap)** — Run 1 `vast-62298484` (`$0.1672`, `1185s`) + Run 2 `vast-62303346` (`$0.2440`, `1729s`) + Run 3 `vast-62309873` (`$0.2048`, `1451s`) + Run 4 `vast-62316259` (`$0.1799`, `1275s`) + Run 5 `vast-62322100` (`$0.4321`, `3062s`) + Run 6 `vast-62334503` (`$0.2262`, `1603s`); `10,305s` total @ `$0.508/h` South Korea `1× RTX 5090`; `Running: 0 pod(s) ($0/h of $8.00/h)`
- Run: `d91-foundry-17b-v7` (exact **untouched `v3/train_foundry_lora_v3_5.py`** [`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`] with controlled 4-knob CLI tuning `--kl-anchor-weight 0.25 --label-smoothing 0.05 --weight-decay 0.025 --lr 1.4e-4 --skip-baseline`; unchanged `64`/`16` zero-leakage `"You are a helpful assistant."` anchors, unchanged `val.jsonl`, primary no-`--calib-file` compile + bounded `--calib-file` fallback, `RUSTFLAGS="-C target-cpu=native"` untouched-source native eval binary)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving paths remain strictly §3-quarantined.

## Empirical Diagnosis from Run 6 (`d91-foundry-17b-v6`) & Exact `v3.5` CLI Tuning

1. **Verified Run 6 Outcomes (`results_v3_run6/d91-out/`, `results_v3_run6.tar.gz` `sha256: 3d39fff542fc4e4544b3d73be47fe16fd02548ed86b59eeb2dc8c1e5405e0439`):**
   - **Bar #1 (`wasm_float_audit.json`):** PASS (`wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`). Untouched-source `RUSTFLAGS="-C target-cpu=native"` `foundry_v13_eval_pack_v2_native` (`5.2M`, logged during pod build on `AMD Ryzen 7 7800X3D` AVX-512; `run6_meta.json` records `native_cpu_build: true`, `host_ram_gb: 30`, `host_avail_gb: 28`, `nproc: 16`) also built cleanly with `lexlean_r4_wasm.wasm` verified byte-identical afterward.
   - **KL Anchor Mechanism Confirmed (`train.log`):** General-prior drift dropped monotonically across training (`anchor_kl_train: 0.119180 -> 0.016234`, `anchor_val_kl_resp: 0.108210 -> 0.072151`, **`anchor_val_kl_first: 0.042179 -> 0.021992`**), and `[delta_w_rel_fro]` stayed tightly bounded (`overall_mean: 0.008689`; `down_proj: 0.004881`, `q_proj: 0.007431`, `k_proj: 0.007430`, `v_proj: 0.007461`, `o_proj: 0.008751`, `up_proj: 0.011190`, `gate_proj: 0.013676`).
   - **BF16 `lora_eval.json` (`pred_sha256: 604e5c7cc7a9e466239111f41024d7aa462eaffea39442f1ad1e8df151d716db`) Halted Cleanly at Stop Gate #1:**
     - **Bar #2 Tool/Build PASS:** **`331/340` (`97.35%`: `285/288` `<tool_call>` [Run 6 best], `46/52` `<prism-write>`)**.
     - **Bar #3 Chat FAIL (`91/100 = 91.0% < 95.0%`, `bars_passed: false`, `total_failures: 18`):** `val_ch_010` (`low_unique_ratio:0.327`), `val_ch_062` (`too_few_words:55`), `val_ch_080` (`forbidden_tag:<tool_call>`), and `6` `missing_expected_keyword` misses (`val_ch_025, 028, 067, 081, 087, 093`), halting cleanly before `recompile_holo_v13` (`run6_meta.json`: `stop_gate: "gate1_bf16"`, `uptime_s: 1603`, `$0.2262`).

2. **Root Cause Comparing Runs 4, 5, 6 & `diag_v3_i8` / `diag_v5`:**
   - Because `train_foundry_lora_v3_5.py:981` adds `(kl_anchor_weight * anc_kl).backward()` at weight `1.0` against SFT (`~1.2..2.8` CE loss on sparse LoRA adapters), `kl_anchor_weight = 1.0` acted as a `1:1` brake against SFT adaptation: `train_loss` stalled at **`1.22854`** (vs Run 5's `0.77225`) and `val_loss_chat` settled at **`2.78726`** — nearly identical to underfitted Run 4 (`val_loss_chat = 2.78850`, Chat `93.0%`), whereas Run 5 (`98.0%` Chat) and Run 2 (`96.0%` Chat) drove `train_loss <= 1.17`. Inspecting `val_ch_087` confirms underfitting pulled technical recall back toward base Qwen3 priors (predicting ``Foundry's WebAssembly `v13` runtime enforces strict IEEE `64`-bit floating-point execution`` instead of pure integer `compute_float_ops == 0`).
   - Simultaneously, [`antigravity-10/DECISION.md`](../antigravity-10/DECISION.md) recorded on `Qwen3-0.6B` in the matched comparison (`lora_alpha=32`, `96` user calib, `weight_decay=0.03`, `--promote-down-int8`) between **`diag_v3_i8` (`label_smoothing = 0.0`)** and **`diag_v5` (`label_smoothing = 0.05`)** that `0.05` smoothing cut `UORR4V13` Bar #4a `12-OOD` `mean_kl_nats` by **`-38.8%`** (`0.075072 -> 0.045925`, `top1 = 91.67% -> 83.33%`, `top5 = 100.00% -> 100.00%`) and lifted compiled `UORR4V13` tool validity (`90.62% -> 97.19%`), while `576`-pos Bar #4b moved `0.110754 -> 0.116247` (`tool_call = 0.015708`, `prism_write = 0.032622`, compiled Chat `97.50% -> 96.25%`) and BF16 held **`100.0%` tool / `98.75%` Chat**. (By contrast, `diag_v4` used unsmoothed `label_smoothing = 0.0` alongside `lora_alpha=16` and `128`-row ChatML calibration, so stepping `0.02 -> 0.05` on `Qwen3-1.7B` is a hypothesis guided by `diag_v5`'s `0.05` target rather than a prior `0.02`-vs-`0.05` ablation.)

3. **Exact Controlled CLI Tuning on Untouched `v3/train_foundry_lora_v3_5.py` (Zero Code Edits, Zero Data Leakage):**
   - **All 6 pinned local scripts and held-out `val.jsonl` (`06814bd9...`, `N = 440`) stay 100% byte-identical.** Only CLI hyperparameters on `v3/train_foundry_lora_v3_5.py` change:
     1. `--kl-anchor-weight 0.25` (reduced `4×` from `1.0`: SFT dominates `4:1` to clear `>= 95.0%` BF16 Chat while preserving general-prior OOD regularization).
     2. `--label-smoothing 0.05` (up from `0.02`, adopting `diag_v5`'s `0.05` smoothing hypothesis to temper teacher logit sharpness ahead of 4-bit Cholesky-GPTQ).
     3. `--weight-decay 0.025` (raised from `0.015` toward `diag_v5`'s `0.03` to limit Frobenius weight norms in RHT groups).
     4. `--lr 1.4e-4` (`T = 225`, `W = 18`, exact `∫lr = 0.015750000`, `+7.7%` peak LR over `1.3e-4` to compensate for `0.25` KL weight + `0.025` weight decay + `0.05` smoothing within the same `1.5` epoch schedule) + `--skip-baseline` (reuses verified untuned `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` baseline `a1ba2937...`: `49.41%` tool / `43.0%` Chat across Runs 1–6, saving `~2 min` of pod billing).

## Bar

Unchanged strict verification gates evaluated on the exact same **held-out `val.jsonl` v3** (`N = 440`, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`, `0` prompt-hash overlap with `train.jsonl`) on BOTH merged BF16 (`lora_eval.json`) AND compiled **`UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), alongside compiler receipt (`foundry_v13_receipt*.json`) and pack verification (`foundry_pack_verify.json`) prior to pod teardown:

1. **Bar #1 — Zero compute floats in compiled WASM (`compute_float_ops == 0`, fail-closed):** `eval_foundry_v13_v3.py --wasm` confirms `compute_float_ops == 0`, `holo_inference_functions_scanned > 0`, `holo_runtime_src_noncomment_float_lines == 0`, and `passed == true`.
2. **Bar #2 — Held-out 18-tool `<tool_call>` & `<prism-write>` schema validity (`>= 92.0%` on both BF16 & compiled `UORR4V13`):** `N_tool = 340` (`288` single-turn + `52` `<prism-write>` builds).
3. **Bar #3 — Held-out structured Chat quality & relevance (`>= 95.0%` on both BF16 & compiled `UORR4V13`):** `N_chat = 100` (`20` unseen topics × `5` unseen phrasings) evaluated with `--max-new-tokens 448` explicit on both BF16 and compiled `foundry_v13_eval_pack_v2_native --mode eval`.
4. **Bar #4 — Dual compiled `UORR4V13` teacher parity (D68 split):**
   - **Bar #4a (`foundry_v13_receipt*.json`, `12` OOD prompts):** `mean_kl_nats <= 0.08`, `top5_parity_pct >= 95.0%`, English output intact.
   - **Bar #4b (`foundry_v13_parity.json`, `576` held-out `val.jsonl` positions across `16 × 36`):** `mlp_mode_label == "DenseExact"`, `top1_parity_pct >= 90.0%`, `top5_parity_pct >= 95.0%`, `p50_kl_nats <= 0.05`, `by_category.tool_call.mean_kl_nats <= 0.05`, `by_category.prism_write.mean_kl_nats <= 0.05`, overall `mean_kl_nats <= 0.12`, `parity_bar_passed == true`.
5. **Bar #5 — Stitch XOR 2-of-3 pack integrity (`foundry_pack_verify.json`):** `pack_XX.pack <= 50,331,648` B (`48 MiB`), `all_packs_under_48mib == true`, `all_xor_groups_exact == true`, and `recovered_container_sha256 == container_sha256` with `--mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`.

## Base artifact & pinned local scripts

- **Pretrained BF16 model:** `Qwen/Qwen3-1.7B` pinned at HF SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (verified baseline `a1ba2937...`: `49.41%` tool / `43.0%` Chat).
- **Compiler & runtime source repo:** `Casey-allard/lexlean-r4@main` `e782e7bc83bc45ab1881b13c045d39592f4cca06`, Rust toolchain `1.97.1` (`wasm32-unknown-unknown`).
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (100% byte-identical to Run 6, zero repo commits):**
  - `v3/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a`
  - `v3/generate_foundry_sft_v3.py`: `sha256: b4222f26b82e6169a522a2e1585e91f13eb8ca1fe6c40754eddad28401c4a132`
  - `v3/train_foundry_lora_v3_5.py`: `sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f` (`TRAIN_SEED = 42`, `epochs = 1.5`, `lr = 1.4e-4`, `warmup = 8%`, `T = 225`, `W = 18`, `∫lr = 0.015750000`, `lora_r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, `weight_decay = 0.025`, `label_smoothing = 0.05`, `kl_anchor_weight = 0.25`, `kl_anchor_batch = 2`, `kl_anchor_tokens = 32`, `max_len = 1280`, `max_new_tokens = 448`, `--skip-baseline`)
  - `v3/eval_foundry_v13_v3.py`: `sha256: a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`

## Data split

Byte-identical to Runs 1–6 (`DATA_SEED = 20261127`, `train_val_overlap_hashes == 0`):
- **`train.jsonl` (`2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`)**
- **`val.jsonl` (`440` rows, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`)**
- **`calib_foundry.txt` (`96` rows, `sha256: 8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`)**
- **`parity_probe_prompts.json` (`16` rows, `sha256: 56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`)**

## Steps (for subsequent GPU execution after pre-run review)

1. Check `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65 ./scripts/pod/uor-pod status` confirming `Running: 0 pod(s)` and a `<= $0.65/h` verified `5090` offer; provision `1× RTX 5090` (`UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `--hours 2.0 --no-bootstrap`, `d91-17b-v7`) with strict `6`-min SSH cutoff + `25m` / `55m` / `95m` `L-004` watchdogs (`0` secrets copied, §3.13).
2. Upload the 6 pinned scripts (`v3/train_foundry_lora_v3_5.py`, `v3/foundry_schemas_v2.json`, `v3/generate_foundry_sft_v3.py`, `v3/eval_foundry_v13_v3.py`, `v3/foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`), assert exact SHA-256 digests, build `lexlean_r4_wasm.wasm`, `foundry_v13_eval_pack_v2_native` (with `RUSTFLAGS="-C target-cpu=native"` into `--target-dir /workspace/target-native`, leaving `.wasm` untouched), and `recompile_holo_v13` against `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`, and fail-closed pass Bar #1 WAT audit.
3. Generate `v3` data (`DATA_SEED = 20261127`, assert all 4 SHA-256 digests), train & evaluate BF16 `Qwen3-1.7B` KL-anchored LoRA via `v3/train_foundry_lora_v3_5.py` (`--epochs 1.5 --lr 1.4e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --weight-decay 0.025 --label-smoothing 0.05 --kl-anchor-weight 0.25 --kl-anchor-batch 2 --kl-anchor-tokens 32 --batch-size 4 --grad-accum 4 --eval-batch-size 8 --max-len 1280 --max-new-tokens 448 --seed 42 --skip-baseline`), gating downstream compile cleanly on `lora_eval.json` passing Bars #2 and #3 (`>= 92.0%` tool, `>= 95.0%` Chat).
4. If BF16 Bars #2 & #3 pass: renew lease, compile **primary no-`--calib-file` `UORR4V13`** (`--cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --out-weights-dir /workspace/d91-out/v13_weights --out-receipt /workspace/d91-out/foundry_v13_receipt.json --pieces-dir /workspace/d91-out/v13_pieces_raw --promote-down-int8`, `UOR_CUDA_ARCH=sm_120`).
   - If primary Bar #4a passes (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`, intact English): set `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_raw`, `ACTIVE_MODE=nocalib`. Run Bar #4b via `foundry_v13_eval_pack_v2_native --mode eval --pieces-dir $ACTIVE_PIECES --probe /workspace/d91-out/parity_probe_tokens.json --logits /workspace/d91-out/teacher_parity_logits.bin --parity-out /workspace/d91-out/foundry_v13_parity.json` (`--mode eval` explicit, recording `T_parity_s`).
   - Bounded fallback compile (**ONLY** once if primary fails Bar #4a alone with `top5 >= 95.0%` and elapsed cost `< $0.50`, OR if primary passes Bar #4a but fails Bar #4b on `val.jsonl` with elapsed cost `< $0.50`): compile `v13_weights_calib` / `v13_pieces_calib` / `foundry_v13_receipt_calib.json` WITH `--calib-file /workspace/d91-data/calib_foundry.txt`; if that passes ALL of Bar #4a + Bar #4b (`foundry_v13_parity_calib.json`), archive primary as `foundry_v13_receipt_nocalib.json` (and `foundry_v13_parity_nocalib.json` if run), promote `foundry_v13_receipt_calib.json -> foundry_v13_receipt.json` & `foundry_v13_parity_calib.json -> foundry_v13_parity.json`, and switch `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_calib`, `ACTIVE_MODE=calib`, else jump to Step 5.
   - On `$ACTIVE_PIECES` passing Bar #4a & Bar #4b: run sharded parallel `foundry_v13_eval_pack_v2_native --mode eval --pieces-dir $ACTIVE_PIECES --val ... --pred-out ... --max-new-tokens 448` (`W = min(7, nproc, max(1, (avail_gb - 4) // 3))`) + `eval_foundry_v13_v3.py` (`Bar #2/#3` -> `foundry_v13_eval.json`), and **ONLY** if Bars #1–#4 all pass run `--mode pack --pieces-dir $ACTIVE_PIECES --pack-dir /workspace/d91-out/v13_pieces_foundry_17b_run7 --verify-out /workspace/d91-out/foundry_pack_verify.json --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` (`Bar #5`).
5. **Mandatory Safe Archive Protocol:** Never `pkill -f` matching CLI names; stream `results_v3_run7.tar.gz` (and `v13_pieces_foundry_17b_run7.tar` from `$ACTIVE_PIECES` **ONLY** if Bars #1–#5 all pass), verify unpacked receipts & packs locally BEFORE `uor-pod release` & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel watchdogs, and hand back via `enter_plan_mode`.

## Stop rule

- Stop cleanly (archive diagnostics locally, release & tear down pod to `0` running, cancel watchdogs, hand back via `enter_plan_mode`) if `compute_float_ops != 0`, `lora_eval` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), both primary and bounded `--calib-file` fallback fail Bar #4a or Bar #4b, compiled `foundry_v13_eval` fails Bar #2 or Bar #3, `foundry_pack_verify` fails Bar #5, or elapsed pod cost approaches **`$1.00` / `2.0h`** inside D91's `$1.5458` remaining cap.
- **What pass unlocks:** Hand back with verified local `results_v3_run7/` digests, Bar #1–#5 receipts, and `v13_pieces_foundry_17b_run7.tar` to publish `weights-v13-qwen3-1.7b-foundry`, wire `Qwen3-1.7B` as an opt-in selector in `web/studio/` alongside default `0.6B`, verify E2E, and transition back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43).
- **What fail unlocks:** Live `gh-pages` and `0.6B` Studio remain untouched at PR #4213, leaving **`>= $0.54`** inside D91 or cleanly closing D91 via Charter §5.1 / D68 decision review using full Run 1–7 evidence across `>= $0.54` remaining headroom before returning to core native M2+.

## Cost

- **Target hardware:** `1× RTX 5090` (`32 GB VRAM`, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`2.0h` / `<= $1.00`** for Run 7 (`d91-foundry-17b-v7`), keeping cumulative D91 spend `<= $2.46` of `+$3.00`.

## Minimum test power

- Held-out `val.jsonl` (`440` unseen prompts: `340` tool/build `95% CI <= 2.9 pp`; `100` Chat across `20` disjoint topics × `5` phrasings `95% CI <= 4.3 pp`), `576 >= 512` `DenseExact` parity positions (`16 × 36`), `16` held-out general anchor validation prompts, and `12` OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at 13d31c562466737bb590881a5ebec74567f5796f by fresh-context research subagent 731395cd-7c69-4122-b0f3-458a15299c63.

Non-blocking execution notes from FIT review (no change to Run 7 spec or bars):
- Pass `--mode eval` explicitly on the Bar #4b `foundry_v13_eval_pack_v2_native` invocation (`--mode` defaults to `"all"`).
- Run `./scripts/pod/uor-pod` from main checkout `/Users/casey.allard/uor-r4` (where `local/pod-provider-target` resides).
