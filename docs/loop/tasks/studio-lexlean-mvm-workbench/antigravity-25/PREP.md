# PREP Sheet for Qwen3-1.7B Foundry Webharness LoRA v3.4 (`d91-foundry-17b-v5`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-24/PREP.md`](../antigravity-24/PREP.md) & [`antigravity-24/REPORT.md`](../antigravity-24/REPORT.md) (PR #4230, `9af20178c`), [`antigravity-23/REPORT.md`](../antigravity-23/REPORT.md) (PR #4229, `e575959fb`), [`antigravity-22/REPORT.md`](../antigravity-22/REPORT.md) (PR #4226, `41586a64c`), [`antigravity-21/REPORT.md`](../antigravity-21/REPORT.md) (PR #4224, `9d3771b73`), [`antigravity-10/REPORT.md`](../antigravity-10/REPORT.md) (D68 Bar #4 split)
- Owner decision: [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`, extending Studio-only D90 exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap, tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger; followed by returning to core native UOR-R4 M2+ milestones)
- D91 cumulative ledger: **`$0.7959` spent (`$2.2041` remaining of fresh `+$3.00` D91 cap)** — Run 1 `vast-62298484` (`$0.1672`, `1185s`) + Run 2 `vast-62303346` (`$0.2440`, `1729s`) + Run 3 `vast-62309873` (`$0.2048`, `1451s`) + Run 4 `vast-62316259` (`$0.1799`, `1275s`), all `$0.508/h` South Korea `1× RTX 5090`; `Running: 0 pod(s) ($0/h of $8.00/h)`
- Run: `d91-foundry-17b-v5` (`v3.4` — gentle Run 2-matched LR schedule `1.3e-4` over `1.50` epochs [`∫lr=0.014625`, `wd=0.015`, cooled `1.50 ep`], with Charter §4.1/§4.2 Bar #4a no-`--calib-file` fallback)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving paths remain strictly §3-quarantined.

## Empirical Diagnosis from Run 4 (`d91-foundry-17b-v4`), Runs 1–3 & Exact v3.4 Changes

1. **Verified Run 4 Outcomes (`results_v3_run4/d91-out/`):**
   - **Bar #1 (`wasm_float_audit.json`):** PASS (`wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_src_sha256: 6b84de0364efe6dd80fe8146c9bf1cc08631e3069e13bb6414c358a0c00907ae`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
   - **Baseline (`baseline_eval.json`):** `tool_pass: 168/340` (`49.41%`: `167/288` `<tool_call>`, `1/52` `<prism-write>`), `chat_pass: 43/100` (`43.0%`).
   - **Post-LoRA BF16 (`lora_eval.json`, `bars_passed: false` -> Stop Gate #1):**
     - **Bar #2 Tool/Build PASS:** `328/340` (**`96.47%`**: `279/288` `<tool_call>`, `49/52` `<prism-write>`).
     - **Bar #3 Chat FAIL (just 2 prompts short):** `93/100` (**`93.0% < 95.0%`**), `total_failures: 19` (`12` tool + `7` Chat: `2` `missing_expected_keyword` [`val_ch_025, 045`], `2` `too_few_words` [`val_ch_046:46`, `val_ch_073:37`], `1` `repeated_4gram:4` [`val_ch_008`], `1` `bad_terminator:'*'` [`val_ch_048`], `1` `forbidden_tag:<tool_call>` [`val_ch_071`]). Halted cleanly at Stop Rule Gate #1 (`recompile_holo_v13` & `--mode pack` skipped).
   - **Root cause comparing Run 4 (`93.0%`) vs Run 2 (`96.0%` Chat, `97.35%` Tool):**
     - Run 4 restored `r=32, alpha=64, dropout=0.05` (`1.9861%` trainable) and lifted Chat from Run 3's `86.0%` to **`93.0%`** while keeping `val_loss_chat = 2.78850` (well below Run 2's `3.01057`).
     - However, compressing `∫lr = 0.015000` into `1` epoch (`T=150, W=12, lr=2.0e-4`) spent **`79%` of its total LR budget by `0.50 ep`** (`cum_∫lr = 0.011795` vs Run 2's gentle `0.007336`), raising early `val_loss_tool` (`0.43108` vs Run 2's `0.41397` at `0.50 ep`) and leaving occasional tail noise (`val_ch_048` ended on markdown `*`, `val_ch_046` & `val_ch_073` terminated concisely at `46`/`37` `_words()` [`45` whitespace words on `046`], `val_ch_008` repeated `('uor', 'r4', 'studio', 'mvm')` 4 times + `make, git` bigram).
     - Pacing virtually the same total LR budget (`∫lr = 0.014625000` vs Run 2's `0.014188` and Run 4's `0.015000`) at Run 2's gentle peak LR (`1.3e-4`, `W=18`) over **`1.50` epochs** (`900` batches, `225` opt steps) matches Run 2's gentle early trajectory (`cum_∫lr` at `0.50/0.75/1.00 ep` = `0.008081 / 0.011481 / 0.013619` vs Run 2's `0.007336 / 0.011114 / 0.014188`) and then cools smoothly all the way to `lr = 0.0` at `1.50 ep` (`0.0010` residual cooling tail vs Run 2's uncooled `lr = 6.817e-5` stop).

2. **Exact Controlled Changes in `v3/train_foundry_lora_v3_4.py` (Dataset & Prompts 100% Unchanged):**
   - `v3/foundry_schemas_v2.json`, `v3/generate_foundry_sft_v3.py`, `v3/eval_foundry_v13_v3.py`, `v3/foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, `v3.py`, `v3_1.py`, `v3_2.py`, `v3_3.py`, and held-out `val.jsonl` (`06814bd9...`, `N = 440`) remain **byte-identical**.
   - Keep `v3_1.py`/`v3_2.py`/`v3_3.py`'s `chat`-history masking in `encode_sft_episode()`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, and `select_best_checkpoint()`.
   - **Change 1 — Gentle `1.50`-epoch cooled defaults (`main()` argparse):** `--epochs 1.5` (`type=float`, `assert args.epochs > 0`), `--lr 1.3e-4`, `--lora-r 32`, `--lora-alpha 64`, `--lora-dropout 0.05`, `--weight-decay 0.015`, `--label-smoothing 0.02` (`T = 225`, `W = 18` [`8%`], exact optimizer-step cosine integral `∫lr = sum(lr_s for s=0..224) = 0.014625000`, cooling strictly to `lr = 0.0` at `1.50 ep`).
   - **Change 2 — Fractional-epoch `global_batch`/`accum_count` training loop:** Stop cleanly at `total_batches = round(len(train_loader) * args.epochs)` (`900`), scale loss by exact `remaining_in_group`, log `{0.50 (batch=300, opt_step=75, lr=1.071578e-4), 0.75 (batch=450, opt_step=112, lr=7.433922e-5), 1.00 (batch=600, opt_step=150, lr=3.775353e-5), 1.25 (batch=750, opt_step=187, lr=1.051329e-5)}` as `telemetry_only`, and restore ONLY the fully cooled `1.50 ep` (`batch=900, opt_step=225, lr=0.0`) endpoint.

## Bar

Unchanged strict verification gates evaluated on the exact same **held-out `val.jsonl` v3** (`N = 440`, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`, `0` prompt-hash overlap with `train.jsonl`) on BOTH merged BF16 (`lora_eval.json`) AND compiled **`UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), alongside untuned baseline (`baseline_eval.json`), compiler receipt (`foundry_v13_receipt.json` or `foundry_v13_receipt_nocalib.json`), and pack verification (`foundry_pack_verify.json`) prior to pod teardown:

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
  - `v3/train_foundry_lora_v3_4.py` (**new Run 5 trainer**): `sha256: eed988d5eafd9c7b59be174085d06c115ae14e2af4b49eb9ce091e2ca4457d35` (`TRAIN_SEED = 42`, `epochs = 1.5`, `lr = 1.3e-4`, `warmup = 8%`, `T = 225`, `W = 18`, `∫lr = 0.014625000`, `lora_r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, `weight_decay = 0.015`, `label_smoothing = 0.02`, masked `chat` history, `{0.50, 0.75, 1.00, 1.25}` `telemetry_only` -> cooled `1.50 ep` restored, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, `max_len = 1280`, `max_new_tokens = 448`)
  - `v3/train_foundry_lora_v3_3.py` (untouched Run 4 reference): `sha256: f88736f925cf6ceab37a723ae1ebcfbbd39f52d1e732a66ba86ae6b281bb499c`
  - `v3/train_foundry_lora_v3_2.py` (untouched Run 3 reference): `sha256: 73b58dd242c55477756ba16b4583dfdba4b2f990a772fe09ca50e521c3c7639d`
  - `v3/train_foundry_lora_v3_1.py` (untouched Run 2 reference): `sha256: bf241ee50580a46b5facfec20e599031f3c4feff3afb016f0bd42ff1245a6566`
  - `v3/train_foundry_lora_v3.py` (untouched Run 1 reference): `sha256: 52158a3fc0ac2ef971e2d31902cbf18ed5f1b0997fb00fbd3b6c15baf59dba0e`
  - `v3/eval_foundry_v13_v3.py`: `sha256: a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`

## Data split

Byte-identical to Runs 1–4 (`DATA_SEED = 20261127`, `train_val_overlap_hashes == 0`):
- **`train.jsonl` (`2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`)**
- **`val.jsonl` (`440` rows, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`)**
- **`calib_foundry.txt` (`96` rows, `sha256: 8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`)**
- **`parity_probe_prompts.json` (`16` rows, `sha256: 56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`)**

## Steps (for subsequent GPU execution after pre-run review)

1. Check `UOR_POD_VAST_DISK_GB=80 uor-pod status` confirming `Running: 0 pod(s)` and a `<= $0.65/h` offer; provision `1× RTX 5090` (`UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `--hours 1.5 --no-bootstrap`, `d91-17b-v5`) with strict `6`-min SSH cutoff + `25m`/`75m` `L-004` watchdogs (`0` secrets copied, §3.13).
2. Upload the 6 pinned scripts (`v3/train_foundry_lora_v3_4.py` + `v3/{foundry_schemas_v2.json,generate_foundry_sft_v3.py,eval_foundry_v13_v3.py,foundry_v13_eval_pack_v2.rs}` + `patch_recompile.py`), assert exact SHA-256 digests, build `lexlean_r4_wasm.wasm`, `recompile_holo_v13`, and `foundry_v13_eval_pack_v2` against `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`, and fail-closed pass Bar #1 WAT audit.
3. Generate `v3` data (`DATA_SEED = 20261127`, assert all 4 SHA-256 digests), train & evaluate BF16 `Qwen3-1.7B` LoRA via `v3/train_foundry_lora_v3_4.py` (`--epochs 1.5 --lr 1.3e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --weight-decay 0.015 --label-smoothing 0.02 --batch-size 4 --grad-accum 4 --eval-batch-size 8 --max-len 1280 --max-new-tokens 448 --seed 42`), gating downstream compile cleanly on `lora_eval.json` passing Bars #2 and #3 (`>= 92.0%` tool, `>= 95.0%` Chat).
4. If BF16 Bars #2 & #3 pass: renew lease, compile primary `UORR4V13` (`--cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --out-weights-dir /workspace/d91-out/v13_weights --out-receipt /workspace/d91-out/foundry_v13_receipt.json --pieces-dir /workspace/d91-out/v13_pieces_raw --promote-down-int8 --calib-file /workspace/d91-data/calib_foundry.txt`, `UOR_CUDA_ARCH=sm_120`).
   - If primary Bar #4a passes (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`, English output intact), set `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_raw` and `ACTIVE_MODE=calib`.
   - Else if primary Bar #4a alone exceeds `0.08` while `top5 >= 95.0%` and elapsed pod cost is `< $0.60`: run `~15s` Bar #4b parity on `v13_pieces_raw` (`foundry_v13_parity_calib.json`), then run one `~3m` no-`--calib-file` compile (`--cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --out-weights-dir /workspace/d91-out/v13_weights_nocalib --out-receipt /workspace/d91-out/foundry_v13_receipt_nocalib.json --pieces-dir /workspace/d91-out/v13_pieces_nocalib --promote-down-int8`); if that passes Bar #4a set `ACTIVE_PIECES=/workspace/d91-out/v13_pieces_nocalib` and `ACTIVE_MODE=nocalib`, else jump to Step 5.
   - Run Bar #4b (`576`-pos `DenseExact` parity -> `/workspace/d91-out/foundry_v13_parity.json` with `--probe`, `--logits`, `--parity-out`), sharded parallel `foundry_v13_eval_pack_v2 --mode eval --pieces-dir $ACTIVE_PIECES --val ... --pred-out ... --max-new-tokens 448` (`W = min(8, nproc, max(1, free_ram_gb // 4))`, unmodified binary) + `eval_foundry_v13_v3.py` (`Bar #2/#3` -> `foundry_v13_eval.json`), and **ONLY** if Bars #1–#4 all pass run `--mode pack --pieces-dir $ACTIVE_PIECES --pack-dir ... --verify-out ... --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` (`Bar #5` -> `foundry_pack_verify.json`).
5. **Mandatory Safe Archive Protocol:** Never `pkill -f` with matched CLI name; stream `results_v3_run5.tar.gz` (and `v13_pieces_foundry_17b_run5.tar` from `$ACTIVE_PIECES` **ONLY** if Bars #1–#5 all pass), verify unpacked receipts & packs locally BEFORE `uor-pod release` & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel watchdogs, and hand back via `enter_plan_mode`.

## Stop rule

- Stop cleanly (archive diagnostics locally, release & tear down pod to `0` running, cancel watchdogs, hand back via `enter_plan_mode`) if `compute_float_ops != 0`, `lora_eval` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), both primary and bounded no-`--calib-file` recompile fail Bar #4a (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`), `foundry_v13_parity` fails Bar #4b (`parity_bar_passed != true`), compiled `foundry_v13_eval` fails Bar #2 or Bar #3, `foundry_pack_verify` fails Bar #5, or elapsed pod cost approaches **`$1.00` / `1.5h`** inside D91's `$2.2041` remaining cap.
- **What pass unlocks:** Hand back with verified local `results_v3_run5/` digests, Bar #1–#5 receipts, and `v13_pieces_foundry_17b_run5.tar` to publish `weights-v13-qwen3-1.7b-foundry`, wire `Qwen3-1.7B` as an opt-in selector in `web/studio/` alongside default `0.6B`, verify E2E, and transition back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43).
- **What fail unlocks:** Live `gh-pages` and `0.6B` Studio remain untouched at PR #4213, with full Run 5 (`v3.4`) + Bar #4 diagnostic data saved inside remaining D91 headroom (`>= $1.20` left).

## Cost

- **Target hardware:** `1× RTX 5090` (`32 GB VRAM`, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`1.5h` / `<= $1.00`** for Run 5 (`d91-foundry-17b-v5`), keeping cumulative D91 spend `<= $1.80` of `+$3.00`.

## Minimum test power

- Held-out `val.jsonl` (`440` unseen prompts: `340` tool/build `95% CI <= 2.9 pp`; `100` Chat across `20` disjoint topics × `5` phrasings `95% CI <= 4.3 pp`) plus `576 >= 512` `DenseExact` parity positions (`16 × 36`) and `12` OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at 0115eb7d167bf3682c8d40e04eef2fba49c7cd58 by fresh-context research subagent b70740ed-05e0-4fad-b16c-d48ef3c810c5.

Non-blocking execution notes from FIT review (no change to `v3.4` spec or bars):
- Corrected `val_ch_008` (`('uor', 'r4', 'studio', 'mvm')` 4-gram trigger) and `val_ch_046` (`46` `_words()` / `45` whitespace words) prose details above.
- Keep `--max-new-tokens 448` explicit on `foundry_v13_eval_pack_v2 --mode eval`, `--mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` explicit on `--mode pack`, and verify all 3 Bar #4a criteria (including intact English output) before selecting `$ACTIVE_PIECES` (whether `v13_pieces_raw` or `v13_pieces_nocalib`). Note `DataLoader(..., shuffle=True, generator=gen_rng)` draws epoch 1's first `300` batches (`global_batch = 601..900`) cleanly on GPU (`torch` is absent locally).
