# PREP Sheet for Qwen3-1.7B Foundry Webharness LoRA v3.2 (`d91-foundry-17b-v3`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-22/PREP.md`](../antigravity-22/PREP.md) & [`antigravity-22/REPORT.md`](../antigravity-22/REPORT.md) (PR #4226, `41586a64c`), [`antigravity-21/REPORT.md`](../antigravity-21/REPORT.md) (PR #4224, `9d3771b73`), [`antigravity-10/REPORT.md`](../antigravity-10/REPORT.md) (D68 Bar #4 split)
- Owner decision: [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`, extending Studio-only D90 exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap, tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger; followed by returning to core native UOR-R4 M2+ milestones)
- D91 cumulative ledger: **`$0.4112` spent (`$2.5888` remaining of fresh `+$3.00` D91 cap)** — Run 1 `vast-62298484` (`$0.1672`, `1185s`) + Run 2 `vast-62303346` (`$0.2440`, `1729s`) both `$0.508/h` South Korea `1× RTX 5090`; `Running: 0 pod(s) ($0/h of $8.00/h)`
- Run: `d91-foundry-17b-v3` (`v3.2` — cooled OOD-safe LoRA regularization + fine-grained `1`-epoch checkpoint selection)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving paths remain strictly §3-quarantined.

## Empirical Diagnosis from Run 2 (`d91-foundry-17b-v2`) & Exact v3.2 Ablation Changes

1. **Verified Run 2 Outcomes (`results_v3_run2/d91-out/`):**
   - **Bar #1 (`wasm_float_audit.json`):** PASS (`compute_float_ops: 0` across `248` funcs, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
   - **BF16 Bars #2 & #3 PASS (`lora_eval.json`, `bars_passed: true`):** Tool/Build = **`97.35%`** (`331/340`: `282/288` `<tool_call>`, `49/52` `<prism-write>`) and Chat = **`96.0%`** (`96/100` with **`0`** repetition or length failures; only `2` `forbidden_tag` + `2` `missing_expected_keyword`, `total_failures: 13 / 440`). Masked `chat` history + increased effective LR solved BF16 Chat cleanly.
   - **Compile `UORR4V13` & Stop Rule Gate #2 (`foundry_v13_receipt.json`):** `float_ops: 0`, `dense_matmul_ops: 0`, `total_bytes: 1,316,411,550` (`1,255.43 MiB`), `piece_count: 1,256`, `parity_groups: 628`, **`top1_parity_pct: 100.00%` (`12/12`)**, **`top5_parity_pct: 100.00%` (`12/12`)**, coherent English on all 12 prompts, **`mean_kl_nats: 0.191195 > 0.08`** (median `0.157731`, range `0.025569` `ood_03_math` .. `0.563900` `ood_06_rope`). Halted cleanly at Stop Rule Gate #2 (`--mode pack` skipped).
   - **Root cause of Run 2 Bar #4a KL drift (`0.191195` vs `diag_v5` `0.045925` & `lexlean-r4` base `1.7B` `0.033533`):**
     - In Run 2 (`epochs=2, r=32, alpha=64, lr=1.2e-4, wd=0.01`), `val_loss` reached its minimum already at `0.50ep` (`val=1.44963`, `tool=0.41397`, `chat=2.50707`, `∫lr=0.007336`), then overfit by `+0.504` nats to `1.00ep` (`val=1.74351`, `chat=3.01057`, `∫lr=0.014188`, **uncooled `lr=6.817e-5`**).
     - Because `0.50/0.75ep` were `telemetry_only`, `select_best_checkpoint` restored the uncooled, overfit `1.00ep` `r=32` checkpoint, which perturbed base general weights (`ood_07_git` FP32 teacher leaked `uncommitted修改`, `ood_06_rope` jumped to `kl=0.563900`). By contrast, `diag_v5` (`0.6B`, `mean_kl_nats = 0.045925 <= 0.08`) trained with **`r=16, alpha=32, lora_dropout=0.08, weight_decay=0.03`** selected at its optimum with full cosine cooling.

2. **Exact Controlled Ablation in `v3/train_foundry_lora_v3_2.py` (Dataset & Prompts 100% Unchanged):**
   - `v3/foundry_schemas_v2.json`, `v3/generate_foundry_sft_v3.py`, `v3/eval_foundry_v13_v3.py`, `v3/foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, Run 1's `train_foundry_lora_v3.py`, Run 2's `train_foundry_lora_v3_1.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) remain **byte-identical**.
   - Keep `v3_1.py`'s proven `chat`-history masking in `encode_sft_episode()`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, and `select_best_checkpoint()`.
   - **Change 1 — Cooled OOD-safe defaults (`main()` argparse):** `--epochs 1`, `--lr 1.6e-4`, `--lora-r 16`, `--lora-alpha 32`, `--lora-dropout 0.08`, `--weight-decay 0.03`, `--label-smoothing 0.02` (`T = 150`, `W = 12` [`8%`], exact cosine `∫lr = 0.012000` cooling to `0.0` at `1.00` ep, matching `r=16` capacity of `diag_v5` while doubling Run 1's `∫lr = 0.006000` and preserving `chat` history masking).
   - **Change 2 — Fine-grained cooled 1-epoch checkpoint grid (`0.75, 0.875, 1.00`):** Keep `0.50` (`batch=300`) as `telemetry_only` (`lr=1.12e-4`) and evaluate/snapshot `{0.75 (batch=450, lr=2.81e-5), 0.875 (batch=525, lr=7.37e-6), 1.00 (batch=600, lr=0.0)}` into unified `select_best_checkpoint(ckpt_list)` (`val_loss_tool <= min_tool + 0.05`, then latest cooled snapshot within `+0.02` nats of `min_chat`).

## Bar

Unchanged strict verification gates evaluated on the exact same **held-out `val.jsonl` v3** (`N = 440`, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`, `0` prompt-hash overlap with `train.jsonl`) on BOTH merged BF16 (`lora_eval.json`) AND compiled **`UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), alongside untuned baseline (`baseline_eval.json`), compiler receipt (`foundry_v13_receipt.json`), and pack verification (`foundry_pack_verify.json`) prior to pod teardown:

1. **Bar #1 — Zero compute floats in compiled WASM (`compute_float_ops == 0`, fail-closed):** `eval_foundry_v13_v3.py --wasm` confirms `compute_float_ops == 0`, `holo_inference_functions_scanned > 0`, `holo_runtime_src_noncomment_float_lines == 0`, and `passed == true`.
2. **Bar #2 — Held-out 18-tool `<tool_call>` & `<prism-write>` schema validity (`>= 92.0%` on both BF16 & compiled `UORR4V13`):** `N_tool = 340` (`288` single-turn + `52` `<prism-write>` builds).
3. **Bar #3 — Held-out structured Chat quality & relevance (`>= 95.0%` on both BF16 & compiled `UORR4V13`):** `N_chat = 100` (`20` unseen topics × `5` unseen phrasings) evaluated with `--max-new-tokens 448` explicit on both BF16 and compiled `foundry_v13_eval_pack_v2 --mode eval`.
4. **Bar #4 — Dual compiled `UORR4V13` teacher parity (D68 split):**
   - **Bar #4a (`foundry_v13_receipt.json`, `12` OOD prompts):** `mean_kl_nats <= 0.08`, `top5_parity_pct >= 95.0%`, English output intact.
   - **Bar #4b (`foundry_v13_parity.json`, `576` held-out `val.jsonl` positions across `16 × 36`):** `mlp_mode_label == "DenseExact"`, `top1_parity_pct >= 90.0%`, `top5_parity_pct >= 95.0%`, `p50_kl_nats <= 0.05`, `by_category.tool_call.mean_kl_nats <= 0.05`, `by_category.prism_write.mean_kl_nats <= 0.05`, overall `mean_kl_nats <= 0.12`, `parity_bar_passed == true`.
5. **Bar #5 — Stitch XOR 2-of-3 pack integrity (`foundry_pack_verify.json`):** `pack_XX.pack <= 50,331,648` B (`48 MiB`), `all_packs_under_48mib == true`, `all_xor_groups_exact == true`, and `recovered_container_sha256 == container_sha256` with `--mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`.

## Base artifact & pinned local scripts

- **Pretrained BF16 model:** `Qwen/Qwen3-1.7B` pinned at HF SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **Compiler & runtime source repo:** `Casey-allard/lexlean-r4@main` `e782e7bc83bc45ab1881b13c045d39592f4cca06`, Rust toolchain `1.97.1` (`wasm32-unknown-unknown`).
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (verified offline at `$0.00`, zero repo commits):**
  - `v3/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a`
  - `v3/generate_foundry_sft_v3.py`: `sha256: b4222f26b82e6169a522a2e1585e91f13eb8ca1fe6c40754eddad28401c4a132`
  - `v3/train_foundry_lora_v3_2.py` (**new Run 3 trainer**): `sha256: 73b58dd242c55477756ba16b4583dfdba4b2f990a772fe09ca50e521c3c7639d` (`TRAIN_SEED = 42`, `epochs = 1`, `lr = 1.6e-4`, `warmup = 8%`, `lora_r = 16`, `lora_alpha = 32`, `lora_dropout = 0.08`, `weight_decay = 0.03`, `label_smoothing = 0.02`, masked `chat` history, `{0.75, 0.875, 1.00}` `select_best_checkpoint`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, `max_len = 1280`, `max_new_tokens = 448`)
  - `v3/train_foundry_lora_v3_1.py` (untouched Run 2 reference): `sha256: bf241ee50580a46b5facfec20e599031f3c4feff3afb016f0bd42ff1245a6566`
  - `v3/train_foundry_lora_v3.py` (untouched Run 1 reference): `sha256: 52158a3fc0ac2ef971e2d31902cbf18ed5f1b0997fb00fbd3b6c15baf59dba0e`
  - `v3/eval_foundry_v13_v3.py`: `sha256: a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`

## Data split

Byte-identical to Runs 1–2 (`DATA_SEED = 20261127`, `train_val_overlap_hashes == 0`):
- **`train.jsonl` (`2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`)**
- **`val.jsonl` (`440` rows, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`)**
- **`calib_foundry.txt` (`96` rows, `sha256: 8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`)**
- **`parity_probe_prompts.json` (`16` rows, `sha256: 56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`)**

## Steps (for subsequent GPU execution after pre-run review)

1. Check `UOR_POD_VAST_DISK_GB=80 uor-pod status` confirming `Running: 0 pod(s)` and a `<= $0.65/h` offer; provision `1× RTX 5090` (`UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `--hours 1.5 --no-bootstrap`, `d91-17b-v3`) with strict `6`-min SSH cutoff + `25m`/`75m` `L-004` watchdogs (`0` secrets copied, §3.13).
2. Upload the 6 pinned scripts (`v3/train_foundry_lora_v3_2.py` + `v3/{foundry_schemas_v2.json,generate_foundry_sft_v3.py,eval_foundry_v13_v3.py,foundry_v13_eval_pack_v2.rs}` + `patch_recompile.py`), assert exact SHA-256 digests, build `lexlean_r4_wasm.wasm`, `recompile_holo_v13`, and `foundry_v13_eval_pack_v2` against `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`, and fail-closed pass Bar #1 WAT audit.
3. Generate `v3` data (`DATA_SEED = 20261127`, assert all 4 SHA-256 digests), train & evaluate BF16 `Qwen3-1.7B` LoRA via `v3/train_foundry_lora_v3_2.py` (`--epochs 1 --lr 1.6e-4 --lora-r 16 --lora-alpha 32 --lora-dropout 0.08 --weight-decay 0.03 --label-smoothing 0.02 --eval-batch-size 8 --max-len 1280 --max-new-tokens 448`), gating downstream compile cleanly on `lora_eval.json` passing Bars #2 and #3 (`>= 92.0%` tool, `>= 95.0%` Chat).
4. If BF16 Bars #2 & #3 pass: renew lease, compile `UORR4V13` via `recompile_holo_v13 --cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --promote-down-int8 --calib-file /workspace/d91-data/calib_foundry.txt` (`UOR_CUDA_ARCH=sm_120`), check Bar #4a receipt & Bar #4b `DenseExact` `576`-position parity, run sharded parallel `foundry_v13_eval_pack_v2 --mode eval --max-new-tokens 448` (`W = min(8, nproc, max(1, free_ram_gb // 4))`, unmodified binary) + `eval_foundry_v13_v3.py` (`Bar #2/#3`), and ONLY if Bars #1–#4 all pass run `--mode pack --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` (`Bar #5`).
5. **Mandatory Safe Archive Protocol:** Never `pkill -f` with matched CLI name; stream `results_v3_run3.tar.gz` (and `v13_pieces_foundry_17b_run3.tar` ONLY if Bars #1–#5 all pass), verify unpacked receipts & packs locally BEFORE `uor-pod release` & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel watchdogs, and hand back via `enter_plan_mode`.

## Stop rule

- Stop cleanly (archive diagnostics locally, release & tear down pod to `0` running, cancel watchdogs, hand back via `enter_plan_mode`) if `compute_float_ops != 0`, `lora_eval` or compiled `foundry_v13_eval` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), `foundry_v13_receipt` fails Bar #4a (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`), `foundry_v13_parity` fails Bar #4b (`parity_bar_passed != true`), `foundry_pack_verify` fails Bar #5, or elapsed pod cost approaches **`$1.00` / `1.5h`** inside D91's `$2.5888` remaining cap.
- **Pre-teardown Bar #4a diagnostic clause (Charter §4.1/§4.2):** If Bar #4a alone exceeds `0.08` while BF16 Bars #2/#3 passed and `top1/top5` are intact, BEFORE skipping straight to teardown (and ONLY while elapsed pod spend stays well below `$0.60`), run two bounded read-only/diagnostic measurements on the already-provisioned pod:
  1. The `~15s` Bar #4b `DenseExact` `576`-position check (`foundry_v13_eval_pack_v2 --mode eval --parity-out /workspace/d91-out/foundry_v13_parity.json`), AND
  2. One `~3m` no-`--calib-file` diagnostic recompile (`--out-receipt /workspace/d91-out/foundry_v13_receipt_nocalib.json`, `--pieces-dir /workspace/d91-out/v13_pieces_nocalib`) — skipping `--mode pack` on both unless a single compiled container passes ALL Bars #1–#5 — so `results_v3_run3.tar.gz` definitively distinguishes `calib_foundry.txt` Hessian skew from merged-weight quantization sensitivity.
- **What pass unlocks:** Hand back with verified local `results_v3_run3/` digests & Bar #1–#5 receipts to publish `weights-v13-qwen3-1.7b-foundry`, wire `Qwen3-1.7B` as an opt-in selector in `web/studio/` alongside default `0.6B`, verify E2E, and transition back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43).
- **What fail unlocks:** Live `gh-pages` and `0.6B` Studio remain untouched at PR #4213, with full Run 3 (`v3.2`) + Bar #4 diagnostic data saved inside remaining D91 headroom (`>= $1.59` left).

## Cost

- **Target hardware:** `1× RTX 5090` (`32 GB VRAM`, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`1.5h` / `<= $1.00`** for Run 3 (`d91-foundry-17b-v3`), keeping cumulative D91 spend `<= $1.42` of `+$3.00`.

## Minimum test power

- Held-out `val.jsonl` (`440` unseen prompts: `340` tool/build `95% CI <= 2.9 pp`; `100` Chat across `20` disjoint topics × `5` phrasings `95% CI <= 4.3 pp`) plus `576 >= 512` `DenseExact` parity positions (`16 × 36`) and `12` OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at bd14425f8c3c9fef6cea4a343642d9816428b961 by fresh-context research subagent 85ef6380-4be2-4788-b8d7-2851f296cf53.

Non-blocking execution notes from FIT review (no change to `v3.2` spec or bars):
- At telemetry step `0.50` (`opt_step=75` of `150`, `W=12`), the exact cosine scheduler LR is `9.089e-5` (cosmetic `1.12e-4` typo above has no execution effect; checkpoint candidates `{0.75, 0.875, 1.00}` are `2.81e-5`, `7.37e-6`, `0.0`). On `foundry_v13_receipt.json`, `0.157731` is exact `quantile(0.50)` (`ood_08_http`), middle-two mean `0.1559155`; and `diag_v5/train.log` ran `epochs=2` restoring `best_epoch=1`.
- If the optional Bar #4a no-`--calib-file` diagnostic recompile runs, pass explicit `--out-weights-dir /workspace/d91-out/v13_weights_nocalib`, `--out-receipt /workspace/d91-out/foundry_v13_receipt_nocalib.json`, and `--pieces-dir /workspace/d91-out/v13_pieces_nocalib` (plus `--cache-dir`, `--hf-repo`, `--mirror-url`, `--promote-down-int8`). Keep `--max-new-tokens 448` explicit on `--mode eval` and `--mirror-url ...weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`.
