# PREP Sheet for Qwen3-1.7B Foundry Webharness LoRA v3 (`d91-foundry-17b-v1`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-20/REPORT.md`](../antigravity-20/REPORT.md) (`d90-foundry-lora-v10`, PR #4222), [`antigravity-19/PREP.md`](../antigravity-19/PREP.md) (PR #4221), [`antigravity-15/REPORT.md`](../antigravity-15/REPORT.md) (PR #4213 Phase 4A), [`antigravity-10/REPORT.md`](../antigravity-10/REPORT.md) (D68 Bar #4 split)
- Owner decision: [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-24T04:32:05Z`, extending Studio-only D90 exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap, tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger; followed by returning to core native UOR-R4 M2+ milestones)
- D91 cumulative ledger: **`$0.0000` spent (`$3.0000` remaining of fresh D91 cap)**; D90 `0.6B` closed cleanly at `$2.7427` across 11 pods (`Running: 0 pod(s)`)
- Run: `d91-foundry-17b-v1` (`v3` — first `Qwen3-1.7B` Studio LoRA + pure-integer `UORR4V13` run)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving paths remain strictly §3-quarantined.

## Why Phase 5 (`Qwen3-1.7B`) & v3 Improvements Over Run 10

1. **Empirical Motivation from D90 (`Qwen3-0.6B`, Runs 8–10):**
   - Across `v2.1` (Run 8: `74/100` Chat), `v2.2` (Run 9: `77/100` Chat, `333/340` tool = `97.94%`), and `v2.3` (Run 10: `78/100` Chat, `317/340` tool = `93.24%`), mirroring `HoloV13Model`'s cycle guard (`HoloCycleGuardLogitsProcessor`) cut BF16 decoding loops (`low_unique_ratio: 8 -> 1`, `repeated_4gram: 3 -> 0`), isolating `0.6B`'s hard parametric ceiling (`10` `missing_expected_keyword`, `6` `forbidden_tag`, `5` `too_few_words`, monotonic Chat overfitting `val_loss_chat: 3.18798 -> 3.57613 -> 3.64605` across `15..16×` target repeats).
   - Meanwhile, `Casey-allard/lexlean-r4` already verifies `Qwen/Qwen3-1.7B` compiles cleanly to zero-float `UORR4V13` (`Casey-allard/lexlean-r4@main:gpu/rtx4090_holo_v13_qwen3_1_7b_receipt.json`: `float_ops: 0`, `dense_matmul_ops: 0`, `total_bytes: 1,140,250,782`, `top1_parity_pct: 91.67%`, `top5_parity_pct: 100.00%`, `mean_kl_nats: 0.033533`).

2. **Exact v3 Dataset & Trainer Fixes Verified Offline at `$0.00`:**
   - **Cut Chat Memorization in Half (`2,400` train rows, `DATA_SEED = 20261127`, `7..8×` repeats):** Split `train.jsonl` into `540` `tool_call` (`30 × 18` tools) + `320` `prism_write` (`33` disjoint specs, `0` `.lex.tex` leakage) + `288` `multi_turn` (`16 × 18` across `36` chains) + `1,252` `chat` (`192` general CS/tech rows = `24 × 8` split `96` `"You are a helpful assistant."` + `96` `FOUNDRY_SYSTEM_PROMPT`, plus `1,060` Foundry/RAG/multi-turn Chat rows across `94` `EXTRA_FOUNDRY_TOPICS_TRAIN` [`+6` prose-only negative boundary topics], `20` `STANDARDS_RAG_TRAIN`, `12` `CHAT_TOPICS_TRAIN`, and `14` `mt_chat` targets). All `164` unique train Chat answers repeat uniformly **`7..8×`** (`104` at `8×`, `60` at `7×`), with `1,252/1,252` unique prompts, `0` repeated 4-grams, `0` `Admission.lex.tex` / `ood_01..12` leakage, and structurally unchanged held-out `val.jsonl` (`N = 440`). Also clarified Run 10 train-side prompt boundaries (`workspace_read` vs `stitch_fetch`, `github_pull` vs `github_clone`, `github_list_repos`, `prismpm_verify`, `standards_lookup`).
   - **`Qwen3-1.7B` Memory-Safe LoRA & Cooled Selection (`train_foundry_lora_v3.py`):** Enabled `use_cache=False` + gradient checkpointing (`use_reentrant=False`) + `enable_input_require_grads()` during LoRA training, `use_cache=True` + `gc.collect()` + `torch.cuda.empty_cache()` after `merge_and_unload()`, `--lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --weight-decay 0.03 --label-smoothing 0.05`, `--epochs 1 --lr 8e-5` (`150` opt steps at `batch_size=4, grad_accum=4`, `8%` warmup `12/150` -> exact cosine `∫lr = 0.006000`), `--eval-batch-size 8`, `--max-len 1280`, `--max-new-tokens 448`, `HoloCycleGuardLogitsProcessor`, and `1.00`-epoch cooled checkpoint selection restoring `0.75` if `val_loss(1.00) > val_loss(0.75) + 0.02` OR `val_loss_chat(1.00) > val_loss_chat(0.75) + 0.05`.

## Bar

Unchanged strict verification gates evaluated on deterministic **held-out `val.jsonl` v3** (`N = 440`, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`, `0` prompt-hash overlap with `train.jsonl`) on BOTH merged BF16 (`lora_eval.json`) AND compiled **`UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), alongside untuned baseline (`baseline_eval.json`), compiler receipt (`foundry_v13_receipt.json`), and pack verification (`foundry_pack_verify.json`) prior to pod teardown:

1. **Bar #1 — Zero compute floats in compiled WASM (`compute_float_ops == 0`, fail-closed):** `eval_foundry_v13_v3.py --wasm target/wasm32-unknown-unknown/release/lexlean_r4_wasm.wasm --holo-src crates/wasm/src/holo_geometric_runtime.rs --out wasm_float_audit.json` confirms `compute_float_ops == 0`, `holo_inference_functions_scanned > 0`, `holo_runtime_src_noncomment_float_lines == 0`, and `passed == true`.
2. **Bar #2 — Held-out 18-tool `<tool_call>` & `<prism-write>` schema validity (`>= 92.0%` on both BF16 & compiled `UORR4V13`):** `N_tool = 340` (`288` single-turn + `52` `<prism-write>` builds) validated deterministically (`TOOL_VALIDITY_BAR_PCT = 92.0`).
3. **Bar #3 — Held-out structured Chat quality & relevance (`>= 95.0%` on both BF16 & compiled `UORR4V13`):** `N_chat = 100` (`20` unseen topics × `5` unseen phrasings) evaluated with `--max-new-tokens 448` (`CHAT_PASS_BAR_PCT = 95.0`, unchanged quality rules).
4. **Bar #4 — Dual compiled `UORR4V13` teacher parity (D68 split):**
   - **Bar #4a (`foundry_v13_receipt.json`, `12` OOD prompts):** `mean_kl_nats <= 0.08`, `top5_parity_pct >= 95.0%`, English output intact.
   - **Bar #4b (`foundry_v13_parity.json`, `576` held-out `val.jsonl` positions across `16 × 36`):** `mlp_mode_label == "DenseExact"`, `top1_parity_pct >= 90.0%`, `top5_parity_pct >= 95.0%`, `p50_kl_nats <= 0.05`, `by_category.tool_call.mean_kl_nats <= 0.05`, `by_category.prism_write.mean_kl_nats <= 0.05`, overall `mean_kl_nats <= 0.12`, `parity_bar_passed == true`.
5. **Bar #5 — Stitch XOR 2-of-3 pack integrity (`foundry_pack_verify.json`):** `pack_XX.pack <= 50,331,648` B (`48 MiB`), `all_packs_under_48mib == true`, `all_xor_groups_exact == true`, and `recovered_container_sha256 == container_sha256`.

## Base artifact & pinned local scripts

- **Pretrained BF16 model:** `Qwen/Qwen3-1.7B` pinned at HF SHA `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- **Compiler & runtime source repo:** `Casey-allard/lexlean-r4@main` `e782e7bc83bc45ab1881b13c045d39592f4cca06`, Rust toolchain `1.97.1` (`wasm32-unknown-unknown`).
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (verified offline at `$0.00`, zero repo commits):**
  - `v3/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, `18` tools byte-identical to `r4-agent-loop.mjs`)
  - `v3/generate_foundry_sft_v3.py`: `sha256: b4222f26b82e6169a522a2e1585e91f13eb8ca1fe6c40754eddad28401c4a132`
  - `v3/train_foundry_lora_v3.py`: `sha256: 52158a3fc0ac2ef971e2d31902cbf18ed5f1b0997fb00fbd3b6c15baf59dba0e` (`TRAIN_SEED = 42`, `epochs = 1`, `lr = 8e-5`, `warmup = 8%`, `lora_r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`, `weight_decay = 0.03`, `label_smoothing = 0.05`, gradient checkpointing, `HoloCycleGuardLogitsProcessor`, cooled `1.00`-epoch selection with `val+0.02 / chat+0.05` guard, `max_len = 1280`, `max_new_tokens = 448`)
  - `v3/eval_foundry_v13_v3.py`: `sha256: a6e9eba7fd0d09535c132b3fa060911cbfbdf538f91da22cf1a006527b78c47d`
  - `v3/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5`
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de`

## Data split

- Deterministic seed-fixed generator (`DATA_SEED = 20261127`, verified `train_val_overlap_hashes == 0`, strict even `(user, assistant)` history pairs, `0` `ood_01..12` leakage, `0` `.lex.tex` leakage, `0` repeated 4-grams):
  - **`train.jsonl` (`2,400` rows, `sha256: a2e44fa69496e4fb99802e40be1fce051bd5ccc66c74bdd3d227a2c54ff6ccef`):** `540` `tool_call` (`30 × 18`) + `320` `prism_write` (`33` disjoint specs) + `288` `multi_turn` (`16 × 18`) + `1,252` `chat` (`164` unique targets repeated `7..8×`, `1,252/1,252` unique Chat prompts; `96` default-system + `2,304` `FOUNDRY_SYSTEM_PROMPT`).
  - **`val.jsonl` (`440` rows, `sha256: 06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`):** `440/440` `FOUNDRY_SYSTEM_PROMPT`; `288` `tool_call` (`16 × 18`) + `52` `prism_write` (`13` disjoint specs) + `100` `chat` (`20` unseen topics × `5` unseen phrasings).
  - **`calib_foundry.txt` (`96` rows, `sha256: 8524a836d5a44a6c240929defe876ffa588fcdaf2451f1d81a2ccef199c652a6`):** strict subset sampled from `train.jsonl` (`36` Chat, `30` tool_call, `16` prism_write, `14` multi_turn).
  - **`parity_probe_prompts.json` (`16` rows, `sha256: 56500d4b5deffed61a5f4840987a158af0d92d7fabb4cdb765318fffa99e8be9`):** `10` `tool_call` + `2` `prism_write` + `4` `chat` from `val.jsonl` (`576` parity positions).

## Steps (for subsequent GPU execution after pre-run review)

1. Check `uor-pod status`; provision `1× RTX 5090` (`UOR_POD_MAX_RATE=0.65`, `--hours 1.5 --no-bootstrap`, `d91-foundry-17b-v1`) with strict `6`-min SSH cutoff + `L-004` watchdog timer. Zero secrets/tokens copied (§3.13).
2. Upload the 6 pinned scripts (`v3/` 5 files + `patch_recompile.py`), assert exact SHA-256 digests, build `lexlean_r4_wasm.wasm`, `recompile_holo_v13`, and `foundry_v13_eval_pack_v2`, and fail-closed pass Bar #1 WAT audit.
3. Generate `v3` data (`DATA_SEED = 20261127`, assert all 4 SHA-256 digests), train & evaluate BF16 `Qwen3-1.7B` LoRA (`train_foundry_lora_v3.py` saving `merged_qwen3_17b`), gating downstream compile/eval cleanly on `lora_eval.json` passing Bars #2 and #3.
4. Run `recompile_holo_v13 --cache-dir /workspace/d91-out/merged_qwen3_17b --hf-repo Qwen/Qwen3-1.7B --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-1.7b-foundry --promote-down-int8 --calib-file /workspace/d91-data/calib_foundry.txt` (`UOR_CUDA_ARCH=sm_120`), run `foundry_v13_eval_pack_v2 --mode eval --max-new-tokens 448` + `eval_foundry_v13_v3.py`, and gate `--mode pack` (`v13_pieces_foundry_17b/`) cleanly on Bars #2/#3/#4a/#4b passing.
5. **Mandatory Safe Archive Protocol:** Never `pkill -f` with matched CLI name; stream `gzip -1` archive (including `v13_pieces_foundry_17b/` ONLY if Bars #1–#5 all pass), verify tarball (`> 100 KB`), unpack & inspect receipts locally BEFORE `uor-pod release` & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel watchdogs, and hand back via `enter_plan_mode`.

## Stop rule

- Stop cleanly (archive diagnostics locally, release & tear down pod to `0` running, hand back via `enter_plan_mode`) if `compute_float_ops != 0`, `lora_eval` or compiled `foundry_v13_eval` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), `foundry_v13_receipt` fails Bar #4a (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`), `foundry_v13_parity` fails Bar #4b (`parity_bar_passed != true`), `foundry_pack_verify` fails Bar #5, or elapsed pod cost approaches **`$1.00` / `1.5h`** inside D91's fresh `+$3.00` cap.
- **What pass unlocks:** Hand back with verified local `results_v3_run1/` digests & Bar #1–#5 receipts to approve publishing `weights-v13-qwen3-1.7b-foundry` and wiring `1.7B` as an opt-in selector in `web/studio/` alongside default `0.6B`, before returning to core native UOR-R4 M2+ milestones.
- **What fail unlocks:** Live `gh-pages` and `0.6B` Studio remain untouched at PR #4213, with full Run 1 (`1.7B`) diagnostics saved locally inside `+$3.00` D91 headroom.

## Cost

- **Target hardware:** `1× RTX 5090` (`32 GB VRAM`, `UOR_POD_MAX_RATE=0.65`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`1.5h` / `<= $1.00`** for Run 1 (`d91-foundry-17b-v1`, inside D91's fresh `+$3.00` cap).

## Minimum test power

- Held-out `val.jsonl` (`440` unseen prompts: `340` tool/build `95% CI <= 2.9 pp`; `100` Chat across `20` disjoint topics × `5` phrasings `95% CI <= 4.3 pp`) plus `576 >= 512` `DenseExact` parity positions (`16 × 36`) and `12` OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at 02bb184873b98fc655a351aeae627b7a208f6d82 by fresh-context research subagent 35f1e7c6-fc5d-46c3-be08-7b27cfcd5abd.
