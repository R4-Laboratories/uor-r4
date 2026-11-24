# PREP Sheet for Qwen3-1.7B `v3.5` Final Run 9 Strong KL-Anchored Foundry LoRA (`d91-foundry-17b-v9`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-28/PREP.md`](../antigravity-28/PREP.md) & [`antigravity-28/REPORT.md`](../antigravity-28/REPORT.md) (PR #4234, `76dcf0f1e`), [`antigravity-27/REPORT.md`](../antigravity-27/REPORT.md) (PR #4233, `0abc5a85d`), [`antigravity-26/REPORT.md`](../antigravity-26/REPORT.md) (PR #4232, `890dd10be`), [`antigravity-10/DECISION.md`](../antigravity-10/DECISION.md) (D68 Bar #4 split)
- Owner decision: [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`, extending Studio-only D90 exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap, tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger; followed by returning to core native UOR-R4 M2+ milestones)
- D91 cumulative ledger: **`$1.8113` spent (`$1.1887` remaining of fresh `+$3.00` D91 cap)** — Run 1 `vast-62298484` (`$0.1672`, `1185s`) + Run 2 `vast-62303346` (`$0.2440`, `1729s`) + Run 3 `vast-62309873` (`$0.2048`, `1451s`) + Run 4 `vast-62316259` (`$0.1799`, `1275s`) + Run 5 `vast-62322100` (`$0.4321`, `3062s`) + Run 6 `vast-62334503` (`$0.2262`, `1603s`) + Run 7 `vast-62341307` (`$0.1837`, `1302s`) + Run 8 `vast-62347334` (`$0.1734`, `1229s`); `12,836s` total @ `$0.508/h` South Korea `1× RTX 5090`; `Running: 0 pod(s) ($0/h of $8.00/h)`
- Run: `d91-foundry-17b-v9` (**final D91 `1.7B` configuration**, exact **untouched `v3/train_foundry_lora_v3_5.py`** [`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`] restoring Run 6's strong KL anchor `--kl-anchor-weight 1.0 --weight-decay 0.015 --label-smoothing 0.02` with extended SFT step budget `--epochs 2.5 --lr 1.4e-4 --skip-baseline`; unchanged `64`/`16` zero-leakage `"You are a helpful assistant."` anchors, unchanged `val.jsonl`, primary no-`--calib-file` compile + bounded `--calib-file` fallback, `RUSTFLAGS="-C target-cpu=native"` untouched-source native eval binary)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving paths remain strictly §3-quarantined.

## Empirical Diagnosis Across Runs 1–8 & Exact `v3.5` Run 9 Configuration

1. **Verified Run 8 Outcomes (`results_v3_run8/d91-out/`, `results_v3_run8.tar.gz` `sha256: 7f11d887cbcb2963223b760513a68420f735cdadc9a731d9bf5805b4279c0ffa`):**
   - **Bar #1 (`wasm_float_audit.json`):** PASS (`wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`). Pod build logs are archived locally in `results_v3_run8/d91-out/pod_build.log` (`9,974 B`) and `job_gpu0.log` (`10,974 B`), confirming compilation of `/workspace/d91-out/foundry_v13_eval_pack_v2_native` (`5.2M`) and `/workspace/d91-out/recompile_holo_v13` (`4.6M`) with `.wasm` re-verified unchanged (`a6f57c99...`); the compiled ELF binaries were not archived locally. `run8_meta.json` records `native_cpu_build: true`, `host_ram_gb: 30`, `host_avail_gb: 28`, `nproc: 16`, `skip_baseline: true`, `stop_gate: "gate1_bf16"`.
   - **KL Anchor & Regularization Telemetry (`train.log`):** `ep=0.50 -> 1.50` (`opt_step=225/225`, `lr=0.0` restored): `train_loss: 2.61004 -> 0.86049`, `val_loss: 1.45358 -> 1.82762`, `val_loss_tool: 0.41020 -> 0.51793`, `val_loss_chat: 2.51890 -> 3.16485`, `anchor_kl_train: 0.121892 -> 0.020788`, `anchor_val_kl_resp: 0.118462 -> 0.087008`, `anchor_val_kl_first: 0.113603 -> 0.055465`; `[delta_w_rel_fro]`: `overall_mean=0.010622` (`down_proj=0.006062, gate_proj=0.018294, up_proj=0.013247, q_proj=0.008939, k_proj=0.009316, v_proj=0.008445, o_proj=0.010052`).
   - **BF16 `lora_eval.json` (`pred_sha256: f7c539eaa3316f92cf66ec527aa69123402137f0f92f3f65cf52a683e063799c`) Halted Cleanly at Stop Gate #1:**
     - **Bar #2 Tool/Build PASS (`332/340 = 97.65%`):** `284/288` `<tool_call>` + `48/52` `<prism-write>`.
     - **Bar #3 Chat FAIL (`91/100 = 91.0% < 95.0%`, `bars_passed: false`, `total_failures: 17`):** `3` `forbidden_tag:<tool_call>` prose leaks (`val_ch_070`, `val_ch_076`, `val_ch_078`), `1` `forbidden_tag:\ufffd` (`val_ch_016`: ``"so inspectors can\ufffd\ufffd证 recomputing"`` — splitting Qwen3's multi-byte Chinese token `"验证"`), and `5` `missing_expected_keyword` misses (`val_ch_025, 027, 035, 041, 087`); `10` CJK rows across all `440` validation predictions.

2. **Cross-Run Synthesis Comparing Runs 1–8 (`results_v3_run1..8/d91-out/`):**
   - **Run 8 falsified the `label_smoothing=0.05`-only hypothesis:** Across Runs 1–8 (`440` predictions/run), `\ufffd` counts are `0, 0, 0, 0, 0, 0, 2, 1` and CJK row counts are `0, 10, 0, 3, 9, 0, 10, 10`. With `--label-smoothing 0.02` in Run 8 (`kl=0.30, wd=0.025, lr=1.35e-4`), weak anchoring still allowed the same Qwen3 bilingual `"验证"` subtoken glitch on `val_ch_016` (`10` CJK rows) AND weakened SFT boundary separation enough to re-open `3` `<tool_call>` leaks inside Chat (`91.0%`).
   - **Why Run 6 (`kl_anchor_weight=1.0, weight_decay=0.015, label_smoothing=0.02`) is the clean missing lever:** Run 6 achieved **zero CJK rows (`0/440`), zero `\ufffd` (`0/440`), the strongest OOD first-token anchoring (`anchor_val_kl_first = 0.021992` vs `0.040735` in Run 7 / `0.055465` in Run 8), the smallest adapter drift (`delta_w_rel_fro = 0.008689`), and lowest `val_loss_chat = 2.78726`**, failing BF16 Chat (`91.0%`) only because competing `1:1` KL gradients left `1.5` epochs under-converged (`train_loss = 1.22854` vs `0.77225` in Run 5 / `0.52853` in Run 2). Giving Run 6's exact regularizers (`--kl-anchor-weight 1.0 --weight-decay 0.015 --label-smoothing 0.02`) **`2.5` epochs at `lr=1.4e-4`** (`T = 375`, `W = 30`, exact `∫lr = 0.026250000` = `1.7949×` Run 6's `0.014625000`) drives SFT convergence across Chat boundaries and keyword recall while the `1.0` KL anchor continuously suppresses bilingual CJK drift and OOD Bar #4a KL inflation.

3. **Exact Controlled CLI Tuning on Untouched `v3/train_foundry_lora_v3_5.py` (Zero Code Edits, Zero Data Leakage):**
   - **All 6 pinned local scripts and held-out `val.jsonl` (`06814bd9...`, `N = 440`) remain 100% byte-identical.** Only CLI flags change for final D91 candidate `d91-foundry-17b-v9`:
     1. `--kl-anchor-weight 1.0 --weight-decay 0.015 --label-smoothing 0.02` (exact Run 6 regularizers that delivered `0` CJK, `0` `\ufffd`, and `anchor_val_kl_first = 0.021992`).
     2. `--epochs 2.5 --lr 1.4e-4` (`batches_per_epoch = 600`, `total_batches = 1500`, `T = 375`, `W = 30`, exact `∫lr = 0.026250000`; `train_foundry_lora_v3_5.py` logs telemetry at `{0.50, 0.75, 1.00, 1.25}` epochs and restores the cooled final `2.50ep` checkpoint) + `--skip-baseline` (reuses verified `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` baseline `a1ba2937...`: `49.41%` tool / `43.0%` Chat).

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
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (100% byte-identical to Runs 6–8, zero repo commits):**
  - `v3/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a`
  - `v3/generate_foundry_sft_v3.py`: `sha256: b4222f26b82e6169a522a2e1585e91f13eb8ca1fe6c40754eddad28401c4a132`
  - `v3/train_foundry_lora_v3_5.py`: `sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f` (`TRAIN_SEED = 42`, `epochs = 2.5`, `lr = 1.4e-4`, `warmup = 8%`, `T = 375`, `W = 30`, `∫lr = 0.026250000`, `lora_r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, `weight_decay = 0.015`, `label_smoothing = 0.02`, `kl_anchor_weight = 1.0`, `kl_anchor_batch = 2`, `kl_anchor_tokens = 32`, `max_len = 1280`, `max_new_tokens = 448`, `--skip-baseline`)
  - `v3/eval_foundry_v13_v3.py`: `sha256: a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`

## Data split

Byte-identical to Runs 1–8 (`DATA_SEED = 20261127`, `train_val_overlap_hashes == 0`):
- **`train.jsonl` (`2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`)**
- **`val.jsonl` (`440` rows, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`)**
- **`calib_foundry.txt` (`96` rows, `sha256: 8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`)**
- **`parity_probe_prompts.json` (`16` rows, `sha256: 56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`)**

## Steps (for subsequent GPU execution after pre-run review)

1. From main checkout `/Users/casey.allard/uor-r4`, check `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65 ./scripts/pod/uor-pod status` confirming `Running: 0 pod(s)` and a `<= $0.65/h` verified `5090` offer; provision `1× RTX 5090` (`UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `--hours 2.0 --no-bootstrap`, `d91-17b-v9`) with strict `6`-min SSH cutoff + `25m` / `55m` / `95m` `L-004` watchdogs (`0` secrets copied, §3.13).
2. Upload the 6 pinned scripts (`v3/train_foundry_lora_v3_5.py`, `v3/foundry_schemas_v2.json`, `v3/generate_foundry_sft_v3.py`, `v3/eval_foundry_v13_v3.py`, `v3/foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`), assert exact SHA-256 digests, build `lexlean_r4_wasm.wasm`, `foundry_v13_eval_pack_v2_native` (with `RUSTFLAGS="-C target-cpu=native"` into `--target-dir /workspace/target-native`, leaving `.wasm` untouched), and `recompile_holo_v13` against `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06` (logging build output to `/workspace/d91-out/pod_build.log`), and fail-closed pass Bar #1 WAT audit.
3. Generate `v3` data (`DATA_SEED = 20261127`, assert all 4 SHA-256 digests), train & evaluate BF16 `Qwen3-1.7B` strong KL-anchored LoRA via `v3/train_foundry_lora_v3_5.py` (`--data-dir /workspace/d91-data --out-dir /workspace/d91-out --schemas /workspace/d91-scripts/foundry_schemas_v2.json --epochs 2.5 --lr 1.4e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --weight-decay 0.015 --label-smoothing 0.02 --kl-anchor-weight 1.0 --kl-anchor-batch 2 --kl-anchor-tokens 32 --batch-size 4 --grad-accum 4 --eval-batch-size 8 --max-len 1280 --max-new-tokens 448 --seed 42 --skip-baseline`), gating downstream compile cleanly on `lora_eval.json` passing Bars #2 and #3 (`>= 92.0%` tool, `>= 95.0%` Chat).
4. If BF16 Bars #2 & #3 pass: renew lease, compile **primary no-`--calib-file` `UORR4V13`** (`--cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --out-weights-dir /workspace/d91-out/v13_weights --out-receipt /workspace/d91-out/foundry_v13_receipt.json --pieces-dir /workspace/d91-out/v13_pieces_raw --promote-down-int8`, `UOR_CUDA_ARCH=sm_120`).
   - If primary Bar #4a passes (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`, intact English): set `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_raw`, `ACTIVE_MODE=nocalib`. Run Bar #4b via `foundry_v13_eval_pack_v2_native --mode eval --pieces-dir $ACTIVE_PIECES --probe /workspace/d91-out/parity_probe_tokens.json --logits /workspace/d91-out/teacher_parity_logits.bin --parity-out /workspace/d91-out/foundry_v13_parity.json` (`--mode eval` explicit, recording `T_parity_s`).
   - Bounded fallback compile (**ONLY** once if primary fails Bar #4a alone with `top5 >= 95.0%` and elapsed cost `< $0.55`, OR if primary passes Bar #4a but fails Bar #4b on `val.jsonl` with elapsed cost `< $0.55`): compile `v13_weights_calib` / `v13_pieces_calib` / `foundry_v13_receipt_calib.json` WITH `--calib-file /workspace/d91-data/calib_foundry.txt`; if that passes ALL of Bar #4a + Bar #4b (`foundry_v13_parity_calib.json`), archive primary as `foundry_v13_receipt_nocalib.json` (and `foundry_v13_parity_nocalib.json` if run), promote `foundry_v13_receipt_calib.json -> foundry_v13_receipt.json` & `foundry_v13_parity_calib.json -> foundry_v13_parity.json`, and switch `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_calib`, `ACTIVE_MODE=calib`, else jump to Step 5.
   - On `$ACTIVE_PIECES` passing Bar #4a & Bar #4b: run sharded parallel `foundry_v13_eval_pack_v2_native --mode eval --pieces-dir $ACTIVE_PIECES --val ... --pred-out ... --max-new-tokens 448` (`W = min(7, nproc, max(1, (avail_gb - 4) // 3))`) + `eval_foundry_v13_v3.py` (`Bar #2/#3` -> `foundry_v13_eval.json`), and **ONLY** if Bars #1–#4 all pass run `--mode pack --pieces-dir $ACTIVE_PIECES --pack-dir /workspace/d91-out/v13_pieces_foundry_17b_run9 --verify-out /workspace/d91-out/foundry_pack_verify.json --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` (`Bar #5`).
5. **Mandatory Safe Archive Protocol:** Never `pkill -f` matching CLI names; stream `results_v3_run9.tar.gz` (including `pod_build.log` and `run9_meta.json`, plus `v13_pieces_foundry_17b_run9.tar` from `$ACTIVE_PIECES` **ONLY** if Bars #1–#5 all pass), verify unpacked receipts & packs locally BEFORE `uor-pod release` & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel watchdogs, and hand back via `enter_plan_mode`.

## Stop rule & finality

- Stop cleanly (archive diagnostics locally, release & tear down pod to `0` running, cancel watchdogs, hand back via `enter_plan_mode`) if `compute_float_ops != 0`, `lora_eval` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), both primary and bounded `--calib-file` fallback fail Bar #4a or Bar #4b, compiled `foundry_v13_eval` fails Bar #2 or Bar #3, `foundry_pack_verify` fails Bar #5, or elapsed pod cost approaches **`$1.00` / `2.0h`** inside D91's `$1.1887` remaining cap.
- **Explicit D91 Finality:** Run 9 (`d91-foundry-17b-v9`) is the final `1.7B` training configuration under D91's `+$3.00` cap (`<= $2.82 / $3.00`). No Run 10 absent an unambiguous instrument defect.
- **What pass unlocks:** Hand back with verified local `results_v3_run9/` digests, Bar #1–#5 receipts, and `v13_pieces_foundry_17b_run9.tar` to publish `weights-v13-qwen3-1.7b-foundry`, wire `Qwen3-1.7B` as an opt-in selector in `web/studio/` alongside default `0.6B`, verify E2E, and transition back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43).
- **What fail unlocks:** Close D91 `1.7B` training via Charter §5.1 / D68 decision review across Runs 1–9 (leaving live `gh-pages` and fast `0.6B` Studio untouched at PR #4213) and return to core native UOR-R4 M2+ milestones.

## Cost

- **Target hardware:** `1× RTX 5090` (`32 GB VRAM`, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`2.0h` / `<= $1.00`** for Run 9 (`d91-foundry-17b-v9`), keeping cumulative D91 spend `<= $2.82` of `+$3.00`.

## Minimum test power

- Held-out `val.jsonl` (`440` unseen prompts: `340` tool/build `95% CI <= 2.9 pp`; `100` Chat across `20` disjoint topics × `5` phrasings `95% CI <= 4.3 pp`), `576 >= 512` `DenseExact` parity positions (`16 × 36`), `16` held-out general anchor validation prompts, and `12` OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at 1c93d96655c50676f4ce00d11a1e531bdfac301f by fresh-context research subagent bc382ed2-c24c-4002-9fb0-5d62abb31577.

Non-blocking execution notes from FIT review (no change to Run 9 spec or bars):
- Keep the 6 uploaded scripts colocated in `/workspace/d91-scripts/` so `train_foundry_lora_v3_5.py`'s relative import of `eval_foundry_v13_v3` resolves directly.
- `train_foundry_lora_v3_5.py` logs telemetry at `{0.50, 0.75, 1.00, 1.25}` epochs before restoring the cooled `2.50ep` (`batch=1500/1500`) checkpoint (no telemetry lines print at `1.50..2.25` epochs).
- Execute `./scripts/pod/uor-pod` from `/Users/casey.allard/uor-r4` with `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`.
