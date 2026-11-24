# PREP Sheet for Qwen3-1.7B Foundry Webharness LoRA v3.1 (`d91-foundry-17b-v2`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-21/PREP.md`](../antigravity-21/PREP.md) & [`antigravity-21/REPORT.md`](../antigravity-21/REPORT.md) (PR #4224, `9d3771b73`), [`antigravity-20/REPORT.md`](../antigravity-20/REPORT.md) (PR #4222, `0.6B` closeout), [`antigravity-10/REPORT.md`](../antigravity-10/REPORT.md) (D68 Bar #4 split)
- Owner decision: [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`, extending Studio-only D90 exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap, tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger; followed by returning to core native UOR-R4 M2+ milestones)
- D91 cumulative ledger: **`$0.1672` spent (`$2.8328` remaining of fresh `+$3.00` D91 cap)** — Run 1 `vast-62298484` (`1× RTX 5090`, South Korea, `$0.508/h`, `uptime_s: 1185`); `Running: 0 pod(s) ($0/h of $8.00/h)`
- Run: `d91-foundry-17b-v2` (`v3.1` — controlled ablation after Run 1 Stop-Rule diagnosis)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving paths remain strictly §3-quarantined.

## Empirical Diagnosis from Run 1 (`d91-foundry-17b-v1`) & Exact v3.1 Ablation Changes

1. **Verified Run 1 Outcomes (`results_v3_run1/d91-out/`):**
   - **Bar #1 (`wasm_float_audit.json`):** PASS (`wasm_sha256: a6f57c99f62957e09d651679a0a406b76662803bfffe03af68bb104a711fc4b6`, `holo_inference_functions_scanned: 248`, `compute_float_ops: 0`, `float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
   - **Baseline Untuned `Qwen/Qwen3-1.7B` (`baseline_eval.json`):** `168/340` (`49.41%`: `167/288` `tool_call`, `1/52` `prism_write`), `43/100` (`43.0%`) Chat.
   - **Post-LoRA BF16 (`lora_eval.json`, `epochs=1, lr=8e-5, r=16, label_smoothing=0.05, ∫lr=0.006000`):**
     - **Bar #2 Tool/Build PASS:** `334/340` (**`98.24%`**: `284/288` `tool_call`, `50/52` `prism_write`; `>= 92.0%`). Only 6 fails: `github_pull` vs `github_clone` (2), `lexlean_check`/`lexlean_verify` swap (2), `unclosed_prism_write` hitting 448 tokens (2).
     - **Bar #3 Chat FAIL (`Stop Rule Gate #1`):** `76/100` (**`76.0%`** `< 95.0%`), `bars_passed: false`. `recompile_holo_v13` and `--mode pack` skipped cleanly per Stop Rule.
   - **Training trajectory (strictly monotonic decrease — zero overfitting):** `0.50ep` (`val_loss=1.61690`, `tool=0.49117`, `chat=2.74570`) -> `0.75ep` (`val_loss=1.55610`, `tool=0.46019`, `chat=2.65500`) -> `1.00ep` (`val_loss=1.54236`, `tool=0.45782`, `chat=2.62987`; `train_loss=2.58583`).
   - **Root causes identified across all 24 Chat failures (`results_v3_run1/d91-out/lora_predictions.jsonl`):**
     1. *Severe under-training at `∫lr = 0.006000` (`epochs=1, lr=8e-5, r=16`):* Compared to `0.6B` (`1.6B`-effective capacity needs >2× rank/budget; `1.7B` trainable ratio was only `1.00%`), `val_loss_chat=2.62987` stopped early mid-descent, leaving greedy loops (`5` `repeated_4gram:3`, `5` `low_unique_ratio`, `1` `bad_terminator:'9'`), `4` missed architectural keywords (`val_ch_025, 093, 094, 100`), and `6` `forbidden_tag` leaks (`val_ch_004, 064, 076, 077, 079, 080`) despite `v3` `train.jsonl` containing `0` literal XML tags in Chat and `23..306` clean rows covering those concepts.
     2. *History turn-1 supervision on multi-turn `chat` rows (`val_ch_073: 27w`, `val_ch_074: 31w`, `val_ch_050: 57w`):* `encode_sft_episode` supervised short (`20+`-word) turn-1 `history` assistant messages on `category == "chat"`, teaching the model to emit compact single-paragraph replies on follow-up turns (`min_words=60`).
     3. *Coarse checkpoint selection:* When scaling to `epochs=2`, `v3`'s multi-epoch branch only checked `1.50` and `2.00` on combined `val_loss`, missing any `val_loss_chat` valley at `1.25` or `1.75` and risking optimizing `val_loss_tool` over `val_loss_chat`.

2. **Exact Controlled Ablation in `v3/train_foundry_lora_v3_1.py` (Dataset `val.jsonl` 100% Unchanged):**
   - `v3/foundry_schemas_v2.json`, `v3/generate_foundry_sft_v3.py`, `v3/eval_foundry_v13_v3.py`, `v3/foundry_v13_eval_pack_v2.rs`, `patch_recompile.py`, and Run 1's `v3/train_foundry_lora_v3.py` remain **byte-identical**.
   - **Change 1 — Training capacity & schedule (`main()` defaults):** `--epochs 2`, `--lr 1.2e-4`, `--lora-r 32`, `--lora-alpha 64`, `--lora-dropout 0.05`, `--weight-decay 0.01`, `--label-smoothing 0.02` (`warmup = 24/300` = `8%`, exact cosine `∫lr = 0.018000` = `3.0×` Run 1, keeping `--batch-size 4 --grad-accum 4 --eval-batch-size 8 --max-len 1280 --max-new-tokens 448 --seed 42`, gradient checkpointing, and `HoloCycleGuardLogitsProcessor` unchanged).
   - **Change 2 — Mask `history` assistant turns in `category == "chat"` (`encode_sft_episode()`):** Keep `history` assistant turns masked (`labels = -100`) on `category == "chat"` so only the final structured `80..134`-word `target + <|im_end|>` is supervised, while preserving full multi-turn turn-1+turn-2 supervision on `category == "multi_turn"`.
   - **Change 3 — `0.25`-epoch `val_loss_chat`-minimizing selector (`select_best_checkpoint()`):** In `ep_idx >= 1` (`1.25, 1.50, 1.75, 2.00` alongside `1.00`), keep `0.50` and `0.75` as `telemetry_only`, filter snapshots with `val_loss_tool <= min_tool + 0.05`, then select the latest (most cooled) checkpoint within `+0.02` nats of `min_chat`.

## Bar

Unchanged strict verification gates evaluated on the exact same **held-out `val.jsonl` v3** (`N = 440`, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`, `0` prompt-hash overlap with `train.jsonl`) on BOTH merged BF16 (`lora_eval.json`) AND compiled **`UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), alongside untuned baseline (`baseline_eval.json`), compiler receipt (`foundry_v13_receipt.json`), and pack verification (`foundry_pack_verify.json`) prior to pod teardown:

1. **Bar #1 — Zero compute floats in compiled WASM (`compute_float_ops == 0`, fail-closed):** `eval_foundry_v13_v3.py --wasm` confirms `compute_float_ops == 0`, `holo_inference_functions_scanned > 0`, `holo_runtime_src_noncomment_float_lines == 0`, and `passed == true`.
2. **Bar #2 — Held-out 18-tool `<tool_call>` & `<prism-write>` schema validity (`>= 92.0%` on both BF16 & compiled `UORR4V13`):** `N_tool = 340` (`288` single-turn + `52` `<prism-write>` builds).
3. **Bar #3 — Held-out structured Chat quality & relevance (`>= 95.0%` on both BF16 & compiled `UORR4V13`):** `N_chat = 100` (`20` unseen topics × `5` unseen phrasings) evaluated with `--max-new-tokens 448` (unchanged rules).
4. **Bar #4 — Dual compiled `UORR4V13` teacher parity (D68 split):**
   - **Bar #4a (`foundry_v13_receipt.json`, `12` OOD prompts):** `mean_kl_nats <= 0.08`, `top5_parity_pct >= 95.0%`, English output intact.
   - **Bar #4b (`foundry_v13_parity.json`, `576` held-out `val.jsonl` positions across `16 × 36`):** `mlp_mode_label == "DenseExact"`, `top1_parity_pct >= 90.0%`, `top5_parity_pct >= 95.0%`, `p50_kl_nats <= 0.05`, `by_category.tool_call.mean_kl_nats <= 0.05`, `by_category.prism_write.mean_kl_nats <= 0.05`, overall `mean_kl_nats <= 0.12`, `parity_bar_passed == true`.
5. **Bar #5 — Stitch XOR 2-of-3 pack integrity (`foundry_pack_verify.json`):** `pack_XX.pack <= 50,331,648` B (`48 MiB`), `all_packs_under_48mib == true`, `all_xor_groups_exact == true`, and `recovered_container_sha256 == container_sha256` with `manifest.json` pinned to `weights-v13-qwen3-1.7b-foundry`.

## Base artifact & pinned local scripts

- **Pretrained BF16 model:** `Qwen/Qwen3-1.7B` pinned at HF SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **Compiler & runtime source repo:** `Casey-allard/lexlean-r4@main` `e782e7bc83bc45ab1881b13c045d39592f4cca06`, Rust toolchain `1.97.1` (`wasm32-unknown-unknown`).
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (verified offline at `$0.00`, zero repo commits):**
  - `v3/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a`
  - `v3/generate_foundry_sft_v3.py`: `sha256: b4222f26b82e6169a522a2e1585e91f13eb8ca1fe6c40754eddad28401c4a132`
  - `v3/train_foundry_lora_v3_1.py` (**new Run 2 trainer**): `sha256: bf241ee50580a46b5facfec20e599031f3c4feff3afb016f0bd42ff1245a6566` (`TRAIN_SEED = 42`, `epochs = 2`, `lr = 1.2e-4`, `warmup = 8%`, `lora_r = 32`, `lora_alpha = 64`, `lora_dropout = 0.05`, `weight_decay = 0.01`, `label_smoothing = 0.02`, masked `chat` history, `0.25`-ep `select_best_checkpoint`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, `max_len = 1280`, `max_new_tokens = 448`)
  - `v3/train_foundry_lora_v3.py` (untouched Run 1 reference): `sha256: 52158a3fc0ac2ef971e2d31902cbf18ed5f1b0997fb00fbd3b6c15baf59dba0e`
  - `v3/eval_foundry_v13_v3.py`: `sha256: a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`

## Data split

Byte-identical to Run 1 (`DATA_SEED = 20261127`, verified `train_val_overlap_hashes == 0`, strict even `(user, assistant)` history pairs, `0` `ood_01..12` / `.lex.tex` leakage, `0` repeated 4-grams):
- **`train.jsonl` (`2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`)**
- **`val.jsonl` (`440` rows, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`)**
- **`calib_foundry.txt` (`96` rows, `sha256: 8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`)**
- **`parity_probe_prompts.json` (`16` rows, `sha256: 56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`)**

## Steps (for subsequent GPU execution after pre-run review)

1. Check `UOR_POD_VAST_DISK_GB=80 uor-pod status`; provision `1× RTX 5090` (`UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `--hours 1.5 --no-bootstrap`, `d91-17b-v2`) with strict `6`-min SSH cutoff + `25m`/`75m` `L-004` watchdog timers (`0` secrets copied, §3.13).
2. Upload the 6 pinned scripts (`v3/train_foundry_lora_v3_1.py` + `v3/{foundry_schemas_v2.json,generate_foundry_sft_v3.py,eval_foundry_v13_v3.py,foundry_v13_eval_pack_v2.rs}` + `patch_recompile.py`), assert exact SHA-256 digests, build `lexlean_r4_wasm.wasm`, `recompile_holo_v13`, and `foundry_v13_eval_pack_v2` against `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`, and fail-closed pass Bar #1 WAT audit.
3. Generate `v3` data (`DATA_SEED = 20261127`, assert all 4 SHA-256 digests), train & evaluate BF16 `Qwen3-1.7B` LoRA via `v3/train_foundry_lora_v3_1.py` (with `--epochs 2 --lr 1.2e-4 --lora-r 32 --lora-alpha 64 --weight-decay 0.01 --label-smoothing 0.02 --eval-batch-size 8 --max-len 1280 --max-new-tokens 448`), gating downstream compile/eval cleanly on `lora_eval.json` passing Bars #2 and #3 (`>= 92.0%` tool, `>= 95.0%` Chat).
4. If BF16 Bars #2 & #3 pass: renew lease, compile `UORR4V13` via `recompile_holo_v13 --cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --promote-down-int8 --calib-file /workspace/d91-data/calib_foundry.txt` (`UOR_CUDA_ARCH=sm_120`), check Bar #4a receipt & Bar #4b `DenseExact` `576`-position parity, run sharded parallel `foundry_v13_eval_pack_v2 --mode eval` (`W = min(8, nproc, max(1, free_ram_gb // 4))`, unmodified binary) + `eval_foundry_v13_v3.py` (`Bar #2/#3`), and ONLY if Bars #1–#4 all pass run `--mode pack --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` (`Bar #5`).
5. **Mandatory Safe Archive Protocol:** Never `pkill -f` with matched CLI name; stream `results_v3_run2.tar.gz` (and `v13_pieces_foundry_17b_run2.tar` ONLY if Bars #1–#5 all pass), verify unpacked receipts & packs locally BEFORE `uor-pod release` & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel watchdogs, and hand back via `enter_plan_mode`.

## Stop rule

- Stop cleanly (archive diagnostics locally, release & tear down pod to `0` running, cancel watchdogs, hand back via `enter_plan_mode`) if `compute_float_ops != 0`, `lora_eval` or compiled `foundry_v13_eval` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), `foundry_v13_receipt` fails Bar #4a (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`), `foundry_v13_parity` fails Bar #4b (`parity_bar_passed != true`), `foundry_pack_verify` fails Bar #5, or elapsed pod cost approaches **`$1.00` / `1.5h`** inside D91's `$2.8328` remaining cap.
- **What pass unlocks:** Hand back with verified local `results_v3_run2/` digests & Bar #1–#5 receipts to publish `weights-v13-qwen3-1.7b-foundry`, wire `Qwen3-1.7B` as an opt-in selector in `web/studio/` alongside default `0.6B`, verify E2E, and transition back to core native UOR-R4 M2+ milestones per [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43).
- **What fail unlocks:** Live `gh-pages` and `0.6B` Studio remain untouched at PR #4213, with full Run 2 (`v3.1`) diagnostics saved inside remaining D91 headroom (`>= $1.83` left for final iteration or M2 transition).

## Cost

- **Target hardware:** `1× RTX 5090` (`32 GB VRAM`, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`1.5h` / `<= $1.00`** for Run 2 (`d91-foundry-17b-v2`), keeping cumulative D91 spend `<= $1.17` of `+$3.00`.

## Minimum test power

- Held-out `val.jsonl` (`440` unseen prompts: `340` tool/build `95% CI <= 2.9 pp`; `100` Chat across `20` disjoint topics × `5` phrasings `95% CI <= 4.3 pp`) plus `576 >= 512` `DenseExact` parity positions (`16 × 36`) and `12` OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at fee6353f73da6b879703eb42faba0131dbb2df97 by fresh-context research subagent 62e3085b-5672-44bb-b352-1fae6c6f7d34.

Non-blocking execution notes from FIT review (no change to `v3.1` spec or bars):
- Pass `--max-new-tokens 448` explicitly to `foundry_v13_eval_pack_v2 --mode eval` (its compiled default is `384`) and pass `--mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry` explicitly to `--mode pack`.
- `val_ch_050` (`57` words) is a single-turn `[Standards]` RAG episode (`history: []`), whereas `val_ch_073` (`27` words) and `val_ch_074` (`31` words) carry `34`-word `chat` turn-1 assistant history.
- Verify `UOR_POD_VAST_DISK_GB=80 uor-pod status` shows a `<= $0.65/h` RTX 5090 offer immediately before running `uor-pod up`.
