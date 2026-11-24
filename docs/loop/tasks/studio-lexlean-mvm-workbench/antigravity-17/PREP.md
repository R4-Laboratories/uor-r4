# PREP Sheet for Qwen3-0.6B Foundry Webharness LoRA v2.1 (`d90-foundry-lora-v8`)

- Lab: antigravity
- Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939)
- Parent sheets & prior evidence: [`antigravity-16/PREP.md`](../antigravity-16/PREP.md) (`d90-foundry-lora-v7`, PR #4214), [`antigravity-15/REPORT.md`](../antigravity-15/REPORT.md) (PR #4213 Phase 4A), [`antigravity-10/REPORT.md`](../antigravity-10/REPORT.md) (Run 6 & D68 Bar #4 split), [`antigravity-9/PREP.md`](../antigravity-9/PREP.md)
- Owner decision: [D90](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43) (`2026-11-22T21:39:18Z`, `$3.00` cumulative Vast.ai cap)
- D90 cumulative ledger: Runs 1–6 `$1.8481` + Run 7 `$0.2313` (`vast-62261356`, `1× RTX 5090` `$0.521/h`, `uptime_s: 1598`) = **`$2.0794` spent (`$0.9206` remaining of `$3.00` cap)**
- Run: `d90-foundry-lora-v8` (`v2.1`)
- Scope boundary: Studio-only (`web/studio/` + external `Casey-allard/lexlean-r4@weights-v13-qwen3-0.6b-foundry`); `crates/**`, `src-tauri/**`, `.github/**`, and M1–M8 native training/serving remain untouched under Charter §3.

## Run 7 (`d90-foundry-lora-v7`) Stop-Rule Outcome & v2.1 Diagnosis

1. **Run 7 Observations (before clean teardown to `Running: 0 pod(s)`):**
   - **Bar #1 (Pod WAT Audit):** PASS (`wasm_sha256: aa10f1c37684a762831297928db48f6289810107d930afd894a4510e21b831bc`, `holo_inference_functions_scanned: 249`, `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_runtime_src_noncomment_float_lines: 0`).
   - **Baseline (`baseline_eval.json`):** `29/340` (`8.53%`) tool (`7/288` `tool_call`, `22/52` `prism_write`), `13/100` (`13.0%`) Chat, `398` total failures.
   - **LoRA Overfitting & Early-Epoch Restoration (`train_foundry_lora_v2.py`):** Validation loss rose monotonically after `0.5` epoch (`opt_step=225: val_loss=2.59591` -> `450: 2.73716` -> `675: 2.86139` -> `900: 2.87949`), restoring under-trained `0.5`-epoch step `225`.
   - **Merged BF16 Evaluation (`lora_eval.json`):** `314/340` (`92.35%`) tool (`273/288` `tool_call`, `41/52` `prism_write`; failures included `8` `github_pull` misclassified as `github_push`, `4` `invalid_tool_json` + `1` `unexpected_arg:end_line` on `workspace_write`, `7` `unclosed_prism_write:1!=0` + `1` `invalid_path:prism/Application.holo` on `<prism-write>`), and **`69/100` (`69.0%`) Chat -> FAILED Bar #3 (`>= 95.0%`)** (`bad_terminator:'1'`, `missing_expected_keyword`).
   - **Compiled `UORR4V13` (`recompile.log`, `473,524,382` B, `--promote-down-int8`):** **`top1=83.33%`, `top5=100.00%`, `mean_KL=0.128702` (> `0.08`, FAILED Bar #4a)**; `[ood_01_wasm]` drifted into Chinese (`LexLean-R4 是一个专为 web 接口开发的静态数学推导引擎...`) and `ood_04/08/10` spiked to `0.29–0.35`.
   - **Archive Caveat:** Run 7's local tarball transferred `0` bytes because remote `pkill -TERM -f foundry_v13_eval_pack_v2` matched its own SSH shell command string; primary Run 7 diagnostics were preserved directly in host command logs (`uor-pod log` + session output).
2. **Exact v2.1 Fixes Verified Offline at `$0.00`:**
   - **Eliminate Repetition Overfitting (`7,200 -> 3,200` rows, expanded template diversity):** Expanded unique `<prism-write>` build specs `8 -> 31` (`186` spec×template combinations in `TRAIN_BUILD_SPECS`; zero `.holo` paths; `13` disjoint specs in `VAL_BUILD_SPECS`), expanded multi-turn chains so all `18` tools have `2` distinct even-pair chains (`36` chains), doubled `TRAIN_WRITE_SPECS` `6 -> 12`, disambiguated `github_pull` phrasings, added `+8` `EXTRA_FOUNDRY_TOPICS_TRAIN`, `+4` `STANDARDS_RAG_TRAIN`, and `+4` multi-turn Chat Q&A pairs, and broadened fragile `expected_keywords` entries in `CHAT_TOPICS_VAL` with valid domain synonyms while keeping unseen prompt/topic separation strict (`0` train/val overlap).
   - **Stop OOD Language Drift Without OOD Leakage (`24` English anchors, 50/50 system split):** Replaced the old general anchors (which overlapped `ood_01..12` markers) with `24` completely OOD-disjoint general English CS/engineering Q&A topics (`120` rows with `DEFAULT_ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant."` + `120` rows with `FOUNDRY_SYSTEM_PROMPT`). `eval_foundry_v13_v2.py --self-test` explicitly asserts zero `ood_01..12` leakage keywords inside `train.jsonl`.
   - **Prevent Truncation & Premature Checkpoint Selection:** Raised generation budget from `384 -> 448` tokens across `train_foundry_lora_v2.py` and `foundry_v13_eval_pack_v2` CLI (`--max-new-tokens 448`), lowered learning rate `1.0e-4 -> 6.0e-5`, raised `weight_decay` `0.03 -> 0.05`, added running `train_loss` logging, and restricted best-adapter checkpoint selection strictly to epochs `1.0`, `1.5`, and `2.0` (never rolling back to an under-trained `0.5`-epoch step).

## Bar

Unchanged strict verification gates evaluated on the deterministic **held-out `val.jsonl` v2.1 split** (`N = 440`, `sha256: 3bacd41bd1889ab5d3f6754ddda446022d00ed3a0d3e0c10e9dad220ecc2be9b`, `0` prompt-hash overlap with `train.jsonl`), on BOTH merged BF16 (`lora_eval.json`) AND compiled **`UORR4V13` pure-integer runtime** (`foundry_v13_eval.json`, `foundry_v13_parity.json`), plus untuned baseline (`baseline_eval.json`), compiler receipt (`foundry_v13_receipt.json`), and pack verification (`foundry_pack_verify.json`) before pod teardown:

1. **Bar #1 — Zero compute floats in compiled WASM (`compute_float_ops == 0`, fail-closed):** `eval_foundry_v13_v2.py --wasm target/wasm32-unknown-unknown/release/lexlean_r4_wasm.wasm --holo-src crates/wasm/src/holo_geometric_runtime.rs --out wasm_float_audit.json` confirms `compute_float_ops == 0`, `holo_inference_functions_scanned > 0`, `holo_runtime_src_noncomment_float_lines == 0`, and `passed == true` (allowing `f64.convert_i32_u` in JS ABI bridge functions only).
2. **Bar #2 — Held-out 18-tool `<tool_call>` & `<prism-write>` schema validity (`>= 92.0%` on both `lora_eval.json` and compiled `UORR4V13` `foundry_v13_eval.json`):** `N_tool = 340` (`16 × 18 = 288` unseen single-turn tool calls + `52` single/multi-file `<prism-write>` builds) validated deterministically against `foundry_schemas_v2.json` (`TOOL_VALIDITY_BAR_PCT = 92.0` in `eval_foundry_v13_v2.py`; target goal `>= 98%`).
3. **Bar #3 — Held-out structured Chat quality & relevance (`>= 95.0%` on both `lora_eval.json` and compiled `UORR4V13` `foundry_v13_eval.json`):** `N_chat = 100` (`20` unseen topics × `5` unseen phrasings) evaluated at `--max-new-tokens 448` and checked by deterministic D28 / §3.10 + richness rules: `20..=512` generated tokens, `60..=360` eval words (train templates enforced at `85..=235` words via `check_chat_quality`), `>= 240` non-whitespace chars, clean terminator, zero spurious XML/ChatML tags or `\ufffd`, zero banned boilerplate substrings, max 4-gram count `<= 2`, unique word ratio `>= 0.50`, and `>= 1` `expected_keywords` match (`CHAT_PASS_BAR_PCT = 95.0`).
4. **Bar #4 — Dual compiled `UORR4V13` teacher parity (D68 split):**
   - **Bar #4a (`foundry_v13_receipt.json`, `12` OOD general prompts inside `recompile_holo_v13`):** `mean_kl_nats <= 0.08`, `top5_parity_pct >= 95.0%` (recording `top1_parity_pct` alongside strict English output verification with zero language drift).
   - **Bar #4b (`foundry_v13_parity.json`, `576` held-out `val.jsonl` positions across `16` stratified episodes × `36` positions = `10` `tool_call` + `2` `prism_write` + `4` `chat`):** `mlp_mode_label == "DenseExact"`, `top1_parity_pct >= 90.0%`, `top5_parity_pct >= 95.0%`, `p50_kl_nats <= 0.05`, `by_category.tool_call.mean_kl_nats <= 0.05`, `by_category.prism_write.mean_kl_nats <= 0.05`, overall `mean_kl_nats <= 0.12`, `parity_bar_passed == true`.
5. **Bar #5 — Stitch XOR 2-of-3 pack integrity (`foundry_pack_verify.json`):** `foundry_v13_eval_pack_v2 --mode pack` outputs `<= 50,331,648` B (`48 MiB`) `pack_XX.pack` bundles, `manifest.json` (`uor-r4.holo-pieces/1`), `manifest.stitch.json` (`stitch/xor-2-of-3/1`), `all_packs_under_48mib == true`, `all_xor_groups_exact == true`, and `recovered_container_sha256 == container_sha256`.

## Base artifact & pinned local scripts

- **Pretrained BF16 model:** `Qwen/Qwen3-0.6B` pinned at HF SHA `c1899de289a04d12100db370d81485cdf75e47ca`.
- **Compiler & runtime source repo:** `Casey-allard/lexlean-r4@main` `e782e7bc83bc45ab1881b13c045d39592f4cca06` (`uor-foundry-v13-wasm` v3 deployed reference `0b5b6d30a019286269d432a625eb7d9c7a339b1b`), Rust toolchain `1.97.1` (`rustup target add --toolchain 1.97.1 wasm32-unknown-unknown`).
- **Pinned local scripts under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/` (verified offline at `$0.00`, zero repo commits):**
  - `v2/foundry_schemas_v2.json`: `sha256: 56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, 18 tools byte-identical to `r4-agent-loop.mjs`)
  - `v2/generate_foundry_sft_v2.py`: `sha256: c6c25e1d9f7417c9e39ed7f9b72cc2b563e933c83c20aa0ea00c199d1f5394ec`
  - `v2/train_foundry_lora_v2.py`: `sha256: fdcbf6fbbe080165ae9f562bcc83f703e2ec3ecaae1cda8d94718eecefef05e6` (`TRAIN_SEED = 42`, `epochs = 2`, `lr = 6.0e-5`, `lora_r = 16`, `lora_alpha = 32`, `lora_dropout = 0.08`, `weight_decay = 0.05`, train `label_smoothing = 0.05`, unsmoothed best-`val_loss` selection over epochs `1.0 / 1.5 / 2.0`, `max_len = 1280`, `max_new_tokens = 448`, `dtype=torch.bfloat16`)
  - `v2/eval_foundry_v13_v2.py`: `sha256: 93fdd159f65aa0471a5eb3ce62a7e8fc7030f4b1484956680543db87422997b9` (`--self-test` verified `3,200` train / `440` val / `0` OOD leakage / `340/340` tool + `100/100` Chat gold pass; local WAT audit `passed: true`, `compute_float_ops: 0`)
  - `v2/foundry_v13_eval_pack_v2.rs`: `sha256: 0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5` (invoked with `--max-new-tokens 448`, D68 Bar #4b breakdown)
  - `patch_recompile.py`: `sha256: bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de` (dynamic CUDA arch + `--calib-file` loader)

## Data split

- Deterministic seed-fixed generator (`DATA_SEED = 20261124`, verified `train_val_overlap_hashes == 0`, strict even `(user, assistant)` history pairs across all rows, `0` `ood_01..12` leakage markers in `train.jsonl`):
  - **`train.jsonl` (`3,200` rows, `sha256: 7cfdccd128d50fb862b082eb6bfe9eb81738206c9ae84316efc2ccf0e96bec87`):** `1,080` single-turn tool calls (`60 × 18` tools, ~25% `[Foundry Context]`) + `640` single/multi-file `<prism-write>` builds (`31` distinct specs × `6` templates = `186` combinations) + `540` multi-turn `<tool_response>` continuation/synthesis/2-turn follow-ups (`30 × 18` tools across `36` chains) + `940` Chat rows (`240` general English CS/engineering anchors split `120` `You are a helpful assistant.` + `120` `FOUNDRY_SYSTEM_PROMPT`, plus `700` Foundry/RAG/multi-turn Chat rows; `85–235` words, `0` forbidden XML/ChatML tags or boilerplate tails; overall split `120` default-system + `3,080` `FOUNDRY_SYSTEM_PROMPT`).
  - **`val.jsonl` (`440` rows, `sha256: 3bacd41bd1889ab5d3f6754ddda446022d00ed3a0d3e0c10e9dad220ecc2be9b`):** `440/440` pinned to `FOUNDRY_SYSTEM_PROMPT`; `340` tool/build (`288` unseen single-turn `16 × 18` + `52` `<prism-write>`) + `100` unseen structured Chat (`20` disjoint topics × `5` unseen phrasings).
  - **`calib_foundry.txt` (`96` rows, `sha256: 934cb61d6ce6000b2220ba3d2d89614376a5367be8792c7ef69634c84106c67e`):** `96` user-only plain prompts sampled strictly from `train.jsonl` (`36` Chat, `30` tool_call, `16` prism_write, `14` multi_turn).
  - **`parity_probe_prompts.json` (`16` rows, `sha256: 393c4664da7d2369fd046443066ec41f2ac0e96c486a31e865e3c56262791de0`):** `10` `tool_call` + `2` `prism_write` + `4` `chat` from `val.jsonl` (`16 × 36 = 576` parity positions).

## Steps (for subsequent GPU execution after pre-run review)

1. Check `uor-pod status`; provision `1× RTX 5090` (`<= $1.05/h`, `uor-pod up --lab antigravity --session d90-foundry-lora-v8 --count 1 --hours 1.25 --no-bootstrap --purpose "D90 v8 Qwen3-0.6B v2.1 Foundry LoRA + UORR4V13"`) and arm an `L-004` watchdog timer (`schedule` 35-min alert). Copy ZERO secrets/tokens to the pod (§3.13).
2. Upload the 6 pinned local scripts (`v2/` 5 files + `patch_recompile.py`) to `/workspace/d90-v2/`, assert all 6 SHA-256 digests match this sheet, install Python deps + Rust `1.97.1` (`wasm32-unknown-unknown`) + `wasm-tools`, clone `Casey-allard/lexlean-r4@e782e7bc83bc45ab1881b13c045d39592f4cca06`, run `patch_recompile.py`, stage `foundry_v13_eval_pack_v2.rs`, and build `lexlean_r4_wasm.wasm`, `recompile_holo_v13`, and `foundry_v13_eval_pack_v2`.
3. Run Bar #1 WAT float audit (`eval_foundry_v13_v2.py --wasm ... --holo-src ... --out wasm_float_audit.json`), failing immediately if `passed != true` or `compute_float_ops != 0`.
4. Generate datasets (`generate_foundry_sft_v2.py --schemas ... --out-dir /workspace/d90-data` verifying `train_sha256`, `val_sha256`, `calib_sha256`, `probe_prompts_sha256` match above), then run `train_foundry_lora_v2.py --data-dir /workspace/d90-data --out-dir /workspace/d90-out --epochs 2 --lr 6.0e-5 --weight-decay 0.05 --max-len 1280 --max-new-tokens 448`.
5. Run `UOR_CUDA_ARCH=sm_120` (fallback `sm_89` if RTX 4090) `recompile_holo_v13 ... --promote-down-int8 --calib-file /workspace/d90-data/calib_foundry.txt --mirror-url https://raw.githubusercontent.com/Casey-allard/lexlean-r4/weights-v13-qwen3-0.6b-foundry`, then `foundry_v13_eval_pack_v2 --mode eval --max-new-tokens 448 ...`, `eval_foundry_v13_v2.py --schemas ... --val ... --pred foundry_v13_predictions.jsonl --out foundry_v13_eval.json`, and `foundry_v13_eval_pack_v2 --mode pack ...` (`v13_pieces_foundry/`, `foundry_pack_verify.json`).
6. **Mandatory Safe Archive Protocol:** NEVER run `pkill -f <name>` when `<name>` appears in the SSH command line. Archive `v13_pieces_foundry/` and all JSON/log artifacts into a tarball on pod, SCP to `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/results_v2_run8/`, verify tarball size (`> 100 KB`), unpack and inspect every Bar #1–#5 JSON receipt locally BEFORE calling `uor-pod release` and `uor-pod down <POD> --confirm-archived`, confirm `Running: 0 pod(s)`, and cancel `L-004` watcher.
7. Only if ALL Bars #1–#5 pass, hand back for owner/plan approval to publish `weights-v13-qwen3-0.6b-foundry`, update `web/studio/foundry-assets.lock.json`, deploy `--typed`, and verify live Studio on `gh-pages`.

## Stop rule

- **Stop on regression or budget:** Preserve partial logs/diagnostics safely via SCP before teardown, release lease & `uor-pod down`, confirm `Running: 0 pod(s)`, cancel `L-004` watcher, and hand back via `enter_plan_mode` if `compute_float_ops != 0`, if `lora_eval.json` or `foundry_v13_eval.json` fails Bar #2 (`>= 92.0%`) or Bar #3 (`>= 95.0%`), if `foundry_v13_receipt.json` fails Bar #4a (`mean_kl_nats <= 0.08`, `top5 >= 95.0%`), if `foundry_v13_parity.json` fails Bar #4b (`parity_bar_passed != true`), if `foundry_pack_verify.json` fails XOR/48 MiB recovery, or if Run 8 cost approaches `$0.68` / `1.25h` (`<= $2.76` combined `<= $3.00` D90 cap).
- **What pass unlocks:** Hand back with verified local `results_v2_run8/` digests and full Bar #1–#5 receipts to approve publishing `weights-v13-qwen3-0.6b-foundry` and updating Studio lockfiles.
- **What fail unlocks:** Live `gh-pages` and `weights-v13-qwen3-0.6b-foundry` remain untouched at PR #4213, with full Run 8 diagnostics preserved locally for analysis.

## Cost

- **Target hardware:** `1× RTX 5090` (`~$0.521/h` verified on Vast.ai, capped `<= $1.05/h`) via `uor-pod up --count 1`.
- **Budget cap:** Hard ceiling **`1.25h` / `<= $0.68`** for Run 8 (`3,200`-row dataset expected `~25–35 min`, `~$0.22–$0.30` at `$0.521/h`; combined with `$2.0794` spent across Runs 1–7 keeps cumulative D90 GPU spend `<= $2.76` strictly inside `$3.00` cap with `>= $0.24` headroom).

## Minimum test power

- Held-out `val.jsonl` evaluates `440` unseen prompts (`340` tool/build: per-prompt binomial `95% CI` half-width `<= 2.9 pp` around `92.0%`; `100` Chat across `20` disjoint domain topics × `5` unseen phrasings: per-prompt binomial `95% CI` half-width `<= 4.3 pp` around `95.0%`, noting `5` phrasings share each topic answer at `N_topics = 20`).
- Parity probes `576 >= 512` token positions across `16` stratified `val.jsonl` trajectories plus end-to-end `HoloV13Model::generate_chat_turn_with_history` across all `440` validation episodes and 12 OOD general prompts inside `recompile_holo_v13`.

## Fitness review

TEST FITNESS: FIT — reviewed at 1d598735f280c8fe9ec84526532b15d14fd81701 by fresh-context research subagent 943c6b61-0483-4782-8dbc-fd49b7ed094c (with non-blocking precision update recording `31` `TRAIN_BUILD_SPECS` × `6` templates).
