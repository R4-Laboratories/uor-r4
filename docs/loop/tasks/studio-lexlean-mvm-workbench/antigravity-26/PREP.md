# PREP Sheet for Qwen3-1.7B KL-Anchored Foundry LoRA v3.5 (`d91-foundry-17b-v6`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-25/PREP.md`](../antigravity-25/PREP.md) & [`antigravity-25/REPORT.md`](../antigravity-25/REPORT.md) (PR #4231, `04cd332c7`), [`antigravity-23/REPORT.md`](../antigravity-23/REPORT.md) (PR #4229, `e575959fb`), [`antigravity-10/REPORT.md`](../antigravity-10/REPORT.md) (D68 Bar #4 split)
- Owner decision: [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`, extending Studio-only D90 exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap, tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger; followed by returning to core native UOR-R4 M2+ milestones)
- D91 cumulative ledger: **`$1.2280` spent (`$1.7720` remaining of fresh `+$3.00` D91 cap)** — Run 1 `vast-62298484` (`$0.1672`, `1185s`) + Run 2 `vast-62303346` (`$0.2440`, `1729s`) + Run 3 `vast-62309873` (`$0.2048`, `1451s`) + Run 4 `vast-62316259` (`$0.1799`, `1275s`) + Run 5 `vast-62322100` (`$0.4321`, `3062s`), all `$0.508/h` South Korea `1× RTX 5090`; `Running: 0 pod(s) ($0/h of $8.00/h)`
- Run: `d91-foundry-17b-v6` (`v3.5` — zero-leakage general self-distillation KL anchor `--kl-anchor-weight 1.0 --kl-anchor-batch 2 --kl-anchor-tokens 32` over `64` disjoint train + `16` disjoint val `"You are a helpful assistant."` prompts alongside unchanged `v3.4` LoRA hyperparams & `val.jsonl`; primary no-`--calib-file` compile + bounded `--calib-file` fallback; `RUSTFLAGS="-C target-cpu=native"` untouched-source native eval binary)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving paths remain strictly §3-quarantined.

## Empirical Diagnosis from Run 5 (`d91-foundry-17b-v5`) & Exact `v3.5` Changes

1. **Verified Run 5 Outcomes (`results_v3_run5/d91-out/`):**
   - **Bar #1 (`wasm_float_audit.json`):** PASS (`wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
   - **BF16 `v3.4` (`lora_eval.json`, `bars_passed: true`) PASSED Stop Gate #1:**
     - **Bar #2 Tool/Build PASS:** **`331/340` (`97.35%`: `280/288` `<tool_call>`, `51/52` `<prism-write>`)**.
     - **Bar #3 Chat PASS:** **`98/100` (`98.0%`)**, `total_failures: 11` (`0` repetition loops, `0` length misses, `0` bad terminators, `0` forbidden tags; only `2` `missing_expected_keyword` on `val_ch_025, 043`), cooled `1.50 ep` (`batch=900/900`, `opt_step=225/225`, `lr=0.0`, `train_loss=0.77225`, `val_loss=1.87200`, `val_loss_chat=3.25474`) restored cleanly.
   - **Stop Gate #2 Bar #4a FAIL on BOTH compiles (falsifying calibration-dilution alone):**
     - **Primary `--calib-file calib_foundry.txt` (`foundry_v13_receipt.json`, `container_sha256: 0bfdb7c9...`):** `top1 = 75.00%`, `top5 = 100.00%`, **`mean_kl_nats = 0.197846 > 0.08`**.
     - **Fallback no-`--calib-file` (`foundry_v13_receipt_nocalib.json`, `container_sha256: 5e267786...`):** `top1 = 83.33%`, `top5 = 100.00%`, **`mean_kl_nats = 0.243437 > 0.08`** (`ood_06_rope=0.8891`, `ood_05_js=0.5470`, `ood_02_rust=0.4651`, `ood_07_git=0.2261`, `ood_12_chat=0.1810`, `ood_09_sql=0.1438`, `ood_04_opfs=0.1388`, `ood_10_css=0.0855`, `ood_08_http=0.0762`, `ood_11_sec=0.0717`, `ood_01_wasm=0.0604`, `ood_03_math=0.0367`).

2. **Root Cause Comparing Per-Prompt OOD KL (`base 1.7B` vs `0.6B diag_v5` vs `1.7B Runs 2 & 5`):**
   - `Casey-allard/lexlean-r4@main:gpu/rtx4090_holo_v13_qwen3_1_7b_receipt.json` (`e782e7bc8`) proves un-tuned `Qwen/Qwen3-1.7B` compiles through the exact same `recompile_holo_v13.rs` at `mean_kl_nats = 0.033533` (`top1 = 91.67%`, `top5 = 100.00%`), with `7/12` OOD prompts at `0.0000` KL and only high-entropy `ood_02_rust` at `0.3476`. By contrast, `diag_v5` (`0.6B`) achieved `mean_kl_nats = 0.045925`.
   - `v3.4` trained `2,304/2,400` (`96.0%`) rows on `FOUNDRY_SYSTEM_PROMPT` alongside only `96` (`4.0%`) synthetic CS/tech SFT rows on `"You are a helpful assistant."` (`calib_foundry.txt` samples `5/96` user strings from those `"You are a helpful assistant."` rows and `91/96` from Foundry rows, storing plain user strings with `0/96` explicit system headers until `format_qwen3_prompt()` wraps all `96/96` at compile time) and **zero frozen-base self-distillation KL anchoring**. Without a distributional anchor to the frozen `"You are a helpful assistant."` prior, `r=32` LoRA drifted general OOD activations/representations, introducing word-echo artifacts in teacher rollouts (`"payload payload"`, `"If-Range header range header"`, `"data inside mounted OPFS"`) and large OOD KL spikes (`ood_06_rope = 0.8891`, `ood_05_js = 0.5470`), whereas `ood_03_math` (which preserved base phrasing) stayed at `0.0367 <= 0.08`.
   - In parallel, Run 5 measured that `foundry_v13_eval_pack_v2` compiled with baseline `x86_64` scalar flags (`#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]`) spent `> 14 min` on `6,625` probe tokens (`results_v3_run5/d91-out/parity_probe_tokens.json`, `576` eval positions, `~2.83 GiB` RSS/worker) on `AMD Ryzen 7 7800X3D` (`AVX-512F / AVX2 / VNNI`). Building unmodified source directly on the pod with `RUSTFLAGS="-C target-cpu=native"` into `--target-dir /workspace/target-native` unlocks host SIMD while leaving `holo_geometric_runtime.rs` (`6b84de03...`), `foundry_v13_eval_pack_v2.rs` (`0feb9b07...`), and `lexlean_r4_wasm.wasm` (`a6f57c99...`) untouched and integer arithmetic bit-identical.

3. **Exact Controlled Changes in `v3/train_foundry_lora_v3_5.py` (Zero Data Leakage, Zero Foundry Change):**
   - `v3/foundry_schemas_v2.json`, `v3/generate_foundry_sft_v3.py`, `v3/eval_foundry_v13_v3.py`, `v3/foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, `v3.py`..`v3_4.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) remain **100% byte-identical**.
   - **Self-distillation KL anchor (`--kl-anchor-weight 1.0 --kl-anchor-batch 2 --kl-anchor-tokens 32`):**
     1. Define `GENERAL_SYSTEM_PROMPT = "You are a helpful assistant."`, `64` unique `ANCHOR_TRAIN_PROMPTS`, and `16` unique `ANCHOR_VAL_PROMPTS` covering non-Foundry, non-OOD everyday topics (cooking, botany, music, weather, geography, literature, zoology, sports, photography, woodworking, architecture, history, unit arithmetic without right triangles, nutrition, geology, textiles).
     2. `assert_anchor_prompts_disjoint()` runs at startup (and offline unit tests) asserting `64`/`16` counts, `0` duplicates, `0` train/val anchor overlap, and `0` case-insensitive substring hits across `36` `OOD_FORBIDDEN_SUBSTRINGS` (`wasm`, `lexlean`, `foundry`, `rust`, `euclid`, `pythagorean`, `opfs`, `javascript`, `rope`, `git diff`, `http`, `206`, `sql`, `flexbox`, `css`, `hmac`, `solar system`, `planet`, etc.).
     3. Before LoRA wrapping (`base_model.eval()`, `use_cache=True`), `build_anchor_sequences()` generates deterministic greedy frozen-base rollouts (`<= 32` tokens, trimmed cleanly at EOS) on `ANCHOR_TRAIN_PROMPTS` and `ANCHOR_VAL_PROMPTS`.
     4. On every optimizer step (before `clip_grad_norm_`), sample cyclic mini-batch of `kl_anchor_batch = 2` train anchors, run frozen base (`model.eval()`, `torch.no_grad()`, `model.disable_adapter()`) and LoRA forward (`model.train()` restored), compute exact FP32 forward `KL(p_base || p_lora)` across all response positions (including first assistant token at `t == prompt_len - 1`), and backprop `(kl_anchor_weight * kl_resp).backward()`.
     5. Log `anchor_kl_train`, token-weighted `anchor_val_kl_resp`, and prompt-weighted `anchor_val_kl_first` at `{0.50, 0.75, 1.00, 1.25, 1.50}` checkpoints, plus `[delta_w_rel_fro]` relative Frobenius norms grouped by module suffix prior to `merge_and_unload()`.

## Bar

Unchanged strict verification gates evaluated on the exact same **held-out `val.jsonl` v3** (`N = 440`, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`, `0` prompt-hash overlap with `train.jsonl`) on BOTH merged BF16 (`lora_eval.json`) AND compiled **`UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), alongside untuned baseline (`baseline_eval.json`), compiler receipt (`foundry_v13_receipt*.json`), and pack verification (`foundry_pack_verify.json`) prior to pod teardown:

1. **Bar #1 — Zero compute floats in compiled WASM (`compute_float_ops == 0`, fail-closed):** `eval_foundry_v13_v3.py --wasm` confirms `compute_float_ops == 0`, `holo_inference_functions_scanned > 0`, `holo_runtime_src_noncomment_float_lines == 0`, and `passed == true`.
2. **Bar #2 — Held-out 18-tool `<tool_call>` & `<prism-write>` schema validity (`>= 92.0%` on both BF16 & compiled `UORR4V13`):** `N_tool = 340` (`288` single-turn + `52` `<prism-write>` builds).
3. **Bar #3 — Held-out structured Chat quality & relevance (`>= 95.0%` on both BF16 & compiled `UORR4V13`):** `N_chat = 100` (`20` unseen topics × `5` unseen phrasings) evaluated with `--max-new-tokens 448` explicit on both BF16 and compiled `foundry_v13_eval_pack_v2 --mode eval`.
4. **Bar #4 — Dual compiled `UORR4V13` teacher parity (D68 split):**
   - **Bar #4a (`foundry_v13_receipt*.json`, `12` OOD prompts):** `mean_kl_nats <= 0.08`, `top5_parity_pct >= 95.0%`, English output intact.
   - **Bar #4b (`foundry_v13_parity.json`, `576` held-out `val.jsonl` positions across `16 × 36`):** `mlp_mode_label == "DenseExact"`, `top1_parity_pct >= 90.0%`, `top5_parity_pct >= 95.0%`, `p50_kl_nats <= 0.05`, `by_category.tool_call.mean_kl_nats <= 0.05`, `by_category.prism_write.mean_kl_nats <= 0.05`, overall `mean_kl_nats <= 0.12`, `parity_bar_passed == true`.
5. **Bar #5 — Stitch XOR 2-of-3 pack integrity (`foundry_pack_verify.json`):** `pack_XX.pack <= 50,331,648` B (`48 MiB`), `all_packs_under_48mib == true`, `all_xor_groups_exact == true`, and `recovered_container_sha256 == container_sha256` with `--mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`.

## Base artifact & pinned local scripts

- **Pretrained BF16 model:** `Qwen/Qwen3-1.7B` pinned at HF SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **Compiler & runtime source repo:** `Casey-allard/lexlean-r4@main` `e782e7bc83bc45ab1881b13c045d39592f4cca06`, Rust toolchain `1.97.1` (`wasm32-unknown-unknown`).
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (verified offline at `$0.00`, zero repo commits):**
  - `v3/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a`
  - `v3/generate_foundry_sft_v3.py`: `sha256: b4222f26b82e6169a522a2e1585e91f13eb8ca1fe6c40754eddad28401c4a132`
  - `v3/train_foundry_lora_v3_5.py` (**new Run 6 trainer**): `sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f` (`TRAIN_SEED = 42`, `epochs = 1.5`, `lr = 1.3e-4`, `warmup = 8%`, `T = 225`, `W = 18`, `∫lr = 0.014625000`, `lora_r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, `weight_decay = 0.015`, `label_smoothing = 0.02`, `kl_anchor_weight = 1.0`, `kl_anchor_batch = 2`, `kl_anchor_tokens = 32`, masked `chat` history, `{0.50, 0.75, 1.00, 1.25}` `telemetry_only` -> cooled `1.50 ep` restored, `delta_w_rel_fro` logged, `max_len = 1280`, `max_new_tokens = 448`)
  - `v3/train_foundry_lora_v3_4.py` (untouched Run 5 reference): `sha256: eed988d5eafd9c7b59be174085d06c115ae14e2af4b49eb9ce091e2ca4457d35`
  - `v3/eval_foundry_v13_v3.py`: `sha256: a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`

## Data split

Byte-identical to Runs 1–5 (`DATA_SEED = 20261127`, `train_val_overlap_hashes == 0`):
- **`train.jsonl` (`2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`)**
- **`val.jsonl` (`440` rows, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`)**
- **`calib_foundry.txt` (`96` rows, `sha256: 8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`)**
- **`parity_probe_prompts.json` (`16` rows, `sha256: 56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`)**

## Steps (for subsequent GPU execution after pre-run review)

1. Check `UOR_POD_VAST_DISK_GB=80 uor-pod status` confirming `Running: 0 pod(s)` and a `<= $0.65/h` verified `5090` offer; provision `1× RTX 5090` (`UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `--hours 2.0 --no-bootstrap`, `d91-17b-v6`) with strict `6`-min SSH cutoff + `25m` / `55m` / `105m` `L-004` watchdogs (`0` secrets copied, §3.13).
2. Upload the 6 pinned scripts (`v3/train_foundry_lora_v3_5.py` + `v3/{foundry_schemas_v2.json,generate_foundry_sft_v3.py,eval_foundry_v13_v3.py,foundry_v13_eval_pack_v2.rs}` + `patch_recompile.py`), assert exact SHA-256 digests, build `lexlean_r4_wasm.wasm`, `foundry_v13_eval_pack_v2` (with `RUSTFLAGS="-C target-cpu=native"` into `--target-dir /workspace/target-native` leaving source & WASM untouched), and `recompile_holo_v13` (`--edition 2021 -L .../deps --extern sha2=...`) against `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`, and fail-closed pass Bar #1 WAT audit.
3. Generate `v3` data (`DATA_SEED = 20261127`, assert all 4 SHA-256 digests), train & evaluate BF16 `Qwen3-1.7B` KL-anchored LoRA via `v3/train_foundry_lora_v3_5.py` (`--epochs 1.5 --lr 1.3e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --weight-decay 0.015 --label-smoothing 0.02 --kl-anchor-weight 1.0 --kl-anchor-batch 2 --kl-anchor-tokens 32 --batch-size 4 --grad-accum 4 --eval-batch-size 8 --max-len 1280 --max-new-tokens 448 --seed 42`), gating downstream compile cleanly on `lora_eval.json` passing Bars #2 and #3 (`>= 92.0%` tool, `>= 95.0%` Chat).
4. If BF16 Bars #2 & #3 pass: renew lease, compile **primary no-`--calib-file` `UORR4V13`** (`--cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --out-weights-dir /workspace/d91-out/v13_weights --out-receipt /workspace/d91-out/foundry_v13_receipt.json --pieces-dir /workspace/d91-out/v13_pieces_raw --promote-down-int8`, `UOR_CUDA_ARCH=sm_120`).
   - If primary Bar #4a passes (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`, intact English): set `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_raw`, `ACTIVE_MODE=nocalib`. Run Bar #4b (`576`-pos `DenseExact` parity -> `/workspace/d91-out/foundry_v13_parity.json`, measuring wall-clock `T_parity`).
   - Bounded fallback compile (**ONLY** once if primary fails Bar #4a alone with `top5 >= 95%` and elapsed cost `< $0.50`, OR if primary passes Bar #4a but fails Bar #4b on `val.jsonl` with elapsed cost `< $0.50`): compile `v13_weights_calib` / `v13_pieces_calib` / `foundry_v13_receipt_calib.json` WITH `--calib-file /workspace/d91-data/calib_foundry.txt`; if that passes ALL of Bar #4a + Bar #4b (`foundry_v13_parity_calib.json`), archive primary as `foundry_v13_receipt_nocalib.json` (and `foundry_v13_parity_nocalib.json` if run), promote `foundry_v13_receipt_calib.json -> foundry_v13_receipt.json` & `foundry_v13_parity_calib.json -> foundry_v13_parity.json`, and switch `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_calib`, `ACTIVE_MODE=calib`, else jump to Step 5.
   - On `$ACTIVE_PIECES` passing Bar #4a & Bar #4b: run sharded parallel `foundry_v13_eval_pack_v2 --mode eval --pieces-dir $ACTIVE_PIECES --val ... --pred-out ... --max-new-tokens 448` (`W = min(7, nproc, max(1, (avail_gb - 4) // 3))`) + `eval_foundry_v13_v3.py` (`Bar #2/#3` -> `foundry_v13_eval.json`), and **ONLY** if Bars #1–#4 all pass run `--mode pack --pieces-dir $ACTIVE_PIECES --pack-dir /workspace/d91-out/v13_pieces_foundry_17b_run6 --verify-out /workspace/d91-out/foundry_pack_verify.json --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` (`Bar #5`).
5. **Mandatory Safe Archive Protocol:** Never `pkill -f` matching CLI names; stream `results_v3_run6.tar.gz` (and `v13_pieces_foundry_17b_run6.tar` from `$ACTIVE_PIECES` **ONLY** if Bars #1–#5 all pass), verify unpacked receipts & packs locally BEFORE `uor-pod release` & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel watchdogs, and hand back via `enter_plan_mode`.

## Stop rule

- Stop cleanly (archive diagnostics locally, release & tear down pod to `0` running, cancel watchdogs, hand back via `enter_plan_mode`) if `compute_float_ops != 0`, `lora_eval` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), both primary and bounded `--calib-file` fallback fail Bar #4a or Bar #4b, compiled `foundry_v13_eval` fails Bar #2 or Bar #3, `foundry_pack_verify` fails Bar #5, or elapsed pod cost approaches **`$1.10` / `2.0h`** inside D91's `$1.7720` remaining cap.
- **What pass unlocks:** Hand back with verified local `results_v3_run6/` digests, Bar #1–#5 receipts, and `v13_pieces_foundry_17b_run6.tar` to publish `weights-v13-qwen3-1.7b-foundry`, wire `Qwen3-1.7B` as an opt-in selector in `web/studio/` alongside default `0.6B`, verify E2E, and transition back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43).
- **What fail unlocks:** Live `gh-pages` and `0.6B` Studio remain untouched at PR #4213, with full Run 6 (`v3.5`, `anchor_val_kl_*`, `delta_w_rel_fro`, `T_parity`) diagnostic data saved inside remaining D91 headroom (`>= $0.67` left).

## Cost

- **Target hardware:** `1× RTX 5090` (`32 GB VRAM`, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`2.0h` / `<= $1.10`** for Run 6 (`d91-foundry-17b-v6`), keeping cumulative D91 spend `<= $2.33` of `+$3.00`.

## Minimum test power

- Held-out `val.jsonl` (`440` unseen prompts: `340` tool/build `95% CI <= 2.9 pp`; `100` Chat across `20` disjoint topics × `5` phrasings `95% CI <= 4.3 pp`), `576 >= 512` `DenseExact` parity positions (`16 × 36`), `16` held-out general anchor validation prompts, and `12` OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at 75904ba77b45523ecc2e99e936429198dd3b9d72 by fresh-context research subagent 14b01f27-e3af-466e-87a0-3be250ae165d.

Non-blocking execution notes from FIT review (no change to `v3.5` spec or bars):
- Verify `--target-dir /workspace/target-native` host SIMD build leaves `lexlean_r4_wasm.wasm` (`a6f57c99...`) untouched and audited separately in `--target wasm32-unknown-unknown --release`.
- Keep `--max-new-tokens 448` explicit on `foundry_v13_eval_pack_v2 --mode eval` and `--mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`.
