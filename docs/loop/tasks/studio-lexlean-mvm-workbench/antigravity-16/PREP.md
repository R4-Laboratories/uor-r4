# PREP Sheet for Qwen3-0.6B Foundry Webharness LoRA v2 (`d90-foundry-lora-v7`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-6/PREP.md`](../antigravity-6/PREP.md) (`d90-foundry-lora-v3`), [`antigravity-9/PREP.md`](../antigravity-9/PREP.md) (`d90-foundry-lora-v6`), [`antigravity-10/REPORT.md`](../antigravity-10/REPORT.md) (Run 6 & D68 Bar #4 split), [`antigravity-15/REPORT.md`](../antigravity-15/REPORT.md) (PR #4213 Phase 4A)
- Owner decision: [D90](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-22T21:39:18Z`, `$3.00` cumulative Vast.ai cap; prior Runs 1–6 spent `~$1.85` [`$1.8481`], remaining `~$1.15`)
- Run: `d90-foundry-lora-v7`
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-0.6b-foundry`); `crates/**` and M1–M8 native training/serving remain strictly quarantined under Charter §3.

## Phase 4D Rationale & v1 Root Causes Fixed in v2

1. **Prompt & 18-Tool Alignment:** Run 5's `foundry_schemas.json` only trained 16 tools and an older system prompt string, whereas live Phase 4A (`web/studio/foundry/studio/shell/r4-agent-loop.mjs`) dispatches **18** `FOUNDRY_TOOL_NAMES` (adding `holo_publish` and `holo_inspect`) and passes an updated `FOUNDRY_SYSTEM_PROMPT`. `v2/foundry_schemas_v2.json` matches `FOUNDRY_SYSTEM_PROMPT` and all 18 tool names byte-for-byte.
2. **Short/Boilerplate Chat Replies & Truncation (`192 -> 384` tokens):** Run 5 trained on short (~45–85 word) Chat outputs with repetitive workflow boilerplate tails and evaluated with `max_new_tokens = 192` (`diag_v5` failed `val_ch_009` `unterminated` and `val_pw_*` `missing_prism_write`), causing brief two-line Chat answers in live Studio. `v2` eliminates all boilerplate tails, trains on structured 100–235 word multi-paragraph Studio self-knowledge, `[Standards]` RAG, multi-turn Q&A, technical coding anchors (preventing OOD forgetting seen on `diag_v5` `ood_02`/`ood_07`), and single/multi-file `<prism-write>` examples, while evaluating at `--max-new-tokens 384` across both BF16 and compiled `UORR4V13`.
3. **Run 5 vs Run 6 Calibration Lesson (`antigravity-9`/`10`):** Run 5's `96` user-only prompts (without `<|im_start|>`) achieved `0.045925` on 12-OOD KL and passed D68 Bar #4b (`p50=0.012868`, `tool_call=0.015708`, `prism_write=0.032622`, `mean=0.116247`), whereas Run 6's assistant-continuation calibration (`128` rows with `<|im_start|>assistant`) regressed 12-OOD KL to `0.158847`. Therefore `v2` retains Run 5's `96` user-only prompt calibration regime (`DATA_SEED`-seeded sample from `train.jsonl` only) alongside `--promote-down-int8` and `DenseExact` MLP evaluation.

## Bar

Every gate below is evaluated on the deterministic **held-out `val.jsonl` v2 split** (`N = 440`, `sha256: 3119a5ebf8289c0dea5bd082e2c866c9ceeec5e8317ed6e9e40a977324ea7c9b`, `0` prompt-hash overlap with `train.jsonl`), on BOTH merged BF16 (`lora_eval.json`) AND the **compiled `UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), plus untuned baseline (`baseline_eval.json`), compiler receipt (`foundry_v13_receipt.json`), and pack verification (`foundry_pack_verify.json`) before pod teardown:

1. **Bar #1 — Zero compute floats in compiled WASM (`compute_float_ops == 0`, fail-closed):** `eval_foundry_v13_v2.py --wasm target/wasm32-unknown-unknown/release/lexlean_r4_wasm.wasm --holo-src crates/wasm/src/holo_geometric_runtime.rs --out wasm_float_audit.json` confirms `compute_float_ops == 0`, `holo_inference_functions_scanned > 0`, `holo_runtime_src_noncomment_float_lines == 0`, and `passed == true` (closing `f64.convert_i32_u` to JS ABI bridge functions only).
2. **Bar #2 — Held-out 18-tool `<tool_call>` & `<prism-write>` schema validity (`>= 92.0%` on both `lora_eval.json` and compiled `UORR4V13` `foundry_v13_eval.json`):** `N_tool = 340` (`16 × 18 = 288` unseen single-turn tool calls + `52` single/multi-file `<prism-write>` builds) validated deterministically against `foundry_schemas_v2.json`.
3. **Bar #3 — Held-out structured Chat quality & relevance (`>= 95.0%` on both `lora_eval.json` and compiled `UORR4V13` `foundry_v13_eval.json`):** `N_chat = 100` (`20` unseen topics × `5` unseen phrasings) validated by all deterministic D28 / §3.10 + richness checks: `20..=512` generated tokens, `>= 60` words and `>= 240` non-whitespace chars, clean terminator, zero spurious XML/ChatML tags or `\ufffd`, zero banned boilerplate substrings, max 4-gram count `<= 2`, unique word ratio `>= 0.50`, and `>= 1` `expected_keywords` match.
4. **Bar #4 — Dual compiled `UORR4V13` teacher parity (D68 split):**
   - **Bar #4a (`foundry_v13_receipt.json`, `12` OOD general prompts):** `mean_kl_nats <= 0.08`, `top5_parity_pct >= 95.0%`.
   - **Bar #4b (`foundry_v13_parity.json`, `576` held-out `val.jsonl` positions across `16` stratified episodes × `36` positions = `10` `tool_call` + `2` `prism_write` + `4` `chat`):** `mlp_mode_label == "DenseExact"`, `top1_parity_pct >= 90.0%`, `top5_parity_pct >= 95.0%`, `p50_kl_nats <= 0.05`, `by_category.tool_call.mean_kl_nats <= 0.05`, `by_category.prism_write.mean_kl_nats <= 0.05`, overall `mean_kl_nats <= 0.12`, `parity_bar_passed == true`.
5. **Bar #5 — Stitch XOR 2-of-3 pack integrity (`foundry_pack_verify.json`):** `foundry_v13_eval_pack_v2 --mode pack` outputs `<= 50,331,648` B (`48 MiB`) `pack_XX.pack` bundles, `manifest.json` (`uor-r4.holo-pieces/1`), `manifest.stitch.json` (`stitch/xor-2-of-3/1`), `all_packs_under_48mib == true`, `all_xor_groups_exact == true` (100% drop-and-recover across every group), and `recovered_container_sha256 == container_sha256`.

## Base artifact & pinned local scripts

- **Pretrained BF16 model:** `Qwen/Qwen3-0.6B` pinned at HF SHA `c1899de289a04d12100db370d81485cdf75e47ca`.
- **Compiler & runtime source repo:** `Casey-allard/lexlean-r4@main` `e782e7bc83bc45ab1881b13c045d39592f4cca06` (`uor-foundry-v13-wasm` v3 deployed reference `0b5b6d30a019286269d432a625eb7d9c7a339b1b`), Rust toolchain `1.97.1` (`rustup target add --toolchain 1.97.1 wasm32-unknown-unknown`).
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (verified offline at `$0.00`, zero repo commits):**
  - `v2/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, 18 tools byte-identical to `r4-agent-loop.mjs`)
  - `v2/generate_foundry_sft_v2.py`: `sha256: 011559b23ab9f4ac1f3498eee764924c76b9531bc95af16a10e86ad215902b68`
  - `v2/train_foundry_lora_v2.py`: `sha256: 80c3db4f0cfc3b1fc54303dcd988067ec06f039930c76b2182b9f7d28f339827` (`TRAIN_SEED = 42`, `epochs = 2`, `lr = 1.0e-4`, `lora_r = 16`, `lora_alpha = 32`, `lora_dropout = 0.08`, `weight_decay = 0.03`, train `label_smoothing = 0.05`, unsmoothed best-`val_loss` adapter snapshot, `max_len = 1280`, `max_new_tokens = 384`, greedy eval + 576-position FP32 teacher export)
  - `v2/eval_foundry_v13_v2.py`: `sha256: de98ed2c19471c539921dc82f8e22ec7e89ffbb839606392753d461a3a3832ac` (`--self-test` verified `340/340` tool + `100/100` Chat gold pass; local WAT audit `passed: true`, `compute_float_ops: 0`)
  - `v2/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5` (`--max-new-tokens 384` default, D68 Bar #4b breakdown, `cargo check -p lexlean-r4-wasm --bin foundry_v13_eval_pack_v2` zero warnings)
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de` (dynamic CUDA arch + `--calib-file` loader, unchanged from Run 5)

## Data split

- Deterministic seed-fixed generator (`DATA_SEED = 20261123`, verified `train_val_overlap_hashes == 0`, strict even `(user, assistant)` history pairs across all rows):
  - **`train.jsonl` (`7,200` rows, `sha256: ecc7d94955baf63c93e742364048e3f106cacf2da789b5e2c0977e5417bc67d8`):** `2,700` single-turn tool calls (`150 × 18` tools, ~25% `[Foundry Context]`) + `1,400` single/multi-file `<prism-write>` builds + `1,260` multi-turn `<tool_response>` continuation/synthesis/2-turn follow-ups (`70 × 18` tools) + `1,840` Chat rows (`100–235` word structured Studio/architecture, `[Standards]` RAG, multi-turn Q&A, and technical coding anchors; `0` XML/ChatML tags or boilerplate tails).
  - **`val.jsonl` (`440` rows, `sha256: 3119a5ebf8289c0dea5bd082e2c866c9ceeec5e8317ed6e9e40a977324ea7c9b`):** `340` tool/build (`288` unseen single-turn `16 × 18` + `52` `<prism-write>`) + `100` unseen structured Chat (`20` disjoint topics × `5` unseen phrasings).
  - **`calib_foundry.txt` (`96` rows, `sha256: 64bd52451695e0031dc04ced70261fe91e8ab421562c8349dfc059689ed24ff9`):** `96` user-only plain prompts sampled strictly from `train.jsonl` (`36` Chat, `30` tool_call, `16` prism_write, `14` multi_turn), matching Run 5's calibration regime.
  - **`parity_probe_prompts.json` (`16` rows, `sha256: ddff8616e20eb53f5f507a0c420045875387ae63c8ed74bd68cab96ac0c51cc8`):** `10` `tool_call` + `2` `prism_write` + `4` `chat` from `val.jsonl` (`16 × 36 = 576` parity positions).

## Steps (for subsequent GPU execution after pre-run review)

1. Check `uor-pod status`; provision `1× RTX 5090` (`<= $1.05/h`, `uor-pod up --lab antigravity --session d90-foundry-lora-v7 --count 1 --hours 1.5 --no-bootstrap --purpose "D90 v7 Qwen3-0.6B v2 Foundry LoRA + UORR4V13"`) and arm an `L-004` watchdog timer (`schedule` 45-min alert). Copy ZERO secrets/tokens to the pod (§3.13).
2. Upload the 6 pinned local scripts (`v2/` 5 files + `patch_recompile.py`) to `/workspace/d90-v2/`, assert all 6 SHA-256 digests match this sheet, install Python deps + Rust `1.97.1` (`wasm32-unknown-unknown`) + `wasm-tools`, clone `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`, run `patch_recompile.py`, stage `foundry_v13_eval_pack_v2.rs`, and build `lexlean_r4_wasm.wasm`, `recompile_holo_v13`, and `foundry_v13_eval_pack_v2`.
3. Run Bar #1 WAT float audit (`eval_foundry_v13_v2.py --wasm ... --holo-src ... --out wasm_float_audit.json`), failing immediately if `passed != true` or `compute_float_ops != 0`.
4. Generate datasets (`generate_foundry_sft_v2.py --schemas ... --out-dir /workspace/d90-data` verifying `train_sha256`, `val_sha256`, `calib_sha256`, `probe_prompts_sha256` match above), then run `train_foundry_lora_v2.py --data-dir /workspace/d90-data --out-dir /workspace/d90-out --epochs 2 --lr 1.0e-4 --max-len 1280 --max-new-tokens 384` (saving `baseline_eval.json`, `lora_eval.json`, `merged_qwen3_06b/`, `parity_probe_tokens.json`, `teacher_parity_logits.bin`).
5. Run `UOR_CUDA_ARCH=sm_120` (fallback `sm_89` if RTX 4090) `recompile_holo_v13 ... --promote-down-int8 --calib-file /workspace/d90-data/calib_foundry.txt --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-0.6b-foundry`, then `foundry_v13_eval_pack_v2 --mode eval --max-new-tokens 384 ...`, `eval_foundry_v13_v2.py --schemas ... --val ... --pred foundry_v13_predictions.jsonl --out foundry_v13_eval.json`, and `foundry_v13_eval_pack_v2 --mode pack ...` (`v13_pieces_foundry/`, `foundry_pack_verify.json`).
6. Archive `v13_pieces_foundry/` and all JSON logs/eval receipts off the pod via SCP into `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/results_v2/`, verify local SHA-256 checksums, IMMEDIATELY release lease & `uor-pod down <POD> --confirm-archived`, confirm `Running: 0 pod(s)`, and cancel `L-004` watcher.
7. Only if ALL Bars #1–#5 pass, open a separate execution PR/step to publish `weights-v13-qwen3-0.6b-foundry` and update `web/studio/foundry-assets.lock.json`, deploy `--typed`, and re-verify live Studio on `gh-pages`.

## Stop rule

- **Stop on regression or budget:** Download all partial logs/diagnostics, release lease & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel `L-004` watcher, and hand back via `enter_plan_mode` if `compute_float_ops != 0`, if `lora_eval.json` or `foundry_v13_eval.json` fails Bar #2 (`>= 92%`) or Bar #3 (`>= 95%`), if `foundry_v13_receipt.json` fails Bar #4a, if `foundry_v13_parity.json` fails Bar #4b (`parity_bar_passed != true`), if `foundry_pack_verify.json` fails XOR/48 MiB recovery, or if Run 7 cost approaches `$0.80` / `1.5h` (`~$2.65` combined `<= $3.00` D90 cap).
- **What pass unlocks:** Hand back with verified local `results_v2/` digests and full Bar #1–#5 receipts to approve publishing `weights-v13-qwen3-0.6b-foundry` and shipping Studio v2 lockfile updates.
- **What fail unlocks:** Live `gh-pages` and `weights-v13-qwen3-0.6b-foundry` remain untouched at PR #4213, with full Run 7 diagnostics preserved locally for diagnosis.

## Cost

- **Target hardware:** `1× RTX 5090` (`~$0.521/h` cheapest verified datacenter offer observed on Vast.ai, capped `<= $1.05/h`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling `1.5h` / `<= $0.80` for Run 7 (expected `~35–45 min`, `~$0.30–$0.45`; combined with `~$1.85` across Runs 1–6 keeps total D90 GPU spend well under `$2.65 <= $3.00` cap). Immediate teardown right after SCP.

## Minimum test power

- Held-out `val.jsonl` tests `440` distinct unseen episodes (`340` tool/build: `95% CI` half-width `<= 2.9 pp` around `92%`; `100` Chat across `20` disjoint topics × `5` phrasings: `95% CI` half-width `<= 4.3 pp` around `95%`).
- Parity probes `576 >= 512` token positions across `16` stratified `val.jsonl` trajectories plus end-to-end `HoloV13Model::generate_chat_turn_with_history` across all `440` validation episodes and 12 general OOD prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at a57f0a8a64f2c07330f0c7d0bb1fcdc5d439099f by fresh-context research subagent 532b3052-77b9-49f1-a8a8-4d4b304627d6
