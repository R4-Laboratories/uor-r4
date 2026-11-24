# REPORT for Qwen3-0.6B Foundry Webharness LoRA v2.2 Pre-Registration (`antigravity-18`)

Lab: `antigravity`. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md) (#3939). Base: `472bcf77c` (PR #4217, `antigravity-17`; `ORDER.md` committed alone first at `e7022063d`).

## What did not work

- **Run 8 (`d90-foundry-lora-v8`, `vast-62270346`, `1× RTX 5090`, `uptime_s: 1237`, `$0.1790`) stopped under Charter Stop Rule before `recompile_holo_v13`:**
  - Bar #1 WAT (`wasm_float_audit.json`: `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_inference_functions_scanned: 248`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`) and baseline (`31/340` tool = `9.12%`, `18/100` Chat = `18.0%`, `391` failures) passed as expected, and merged BF16 (`lora_eval.json`, `--max-new-tokens 448`) reached **`335/340` (`98.53%`)** on held-out tool/build (`285/288` `tool_call`, `50/52` `prism_write`; `5` failures: `wrong_tool:workspace_list!=workspace_read`, unescaped newline `invalid_tool_json`, `unexpected_prism_write`, `invalid_path:prism/lex.tex`, `unexpected_tool_call`).
  - However, **`74/100` (`74.0%`) Chat failed Bar #3 (`>= 95.0%`)**: across `100` unseen prompts, `26` Chat episodes failed (`11` `missing_expected_keyword` across `val_ch_025/027/028/043/059/068/088/094/096/098/100` with `val_ch_100` truncated by `sample_failures[:30]`, `6` `forbidden_tag:<prism-write` leaks in Chat prose, `1` `forbidden_tag:\ufffd` in `val_ch_040`, `3` `repeated_4gram:3` failures from dotted version strings `Linux 6.18.7` [`val_ch_008/044`] and `LexLean 0.3.0` [`val_ch_070`, also emitting Chinese `流水线`], `3` `too_few_words` [`val_ch_029`: `47`w, `val_ch_049`: `56`w, `val_ch_073`: `35`w], and `2` `low_unique_ratio` loops [`val_ch_005`: `0.022`, `val_ch_041`: `0.250`]).
  - In `train.log`, under `--epochs 2 --lr 6.0e-5`, validation loss reached `2.07818` at Epoch `1.0` (`opt_step=200/400`) before rising monotonically in Epoch `2` (`2.39821` -> `2.44343`), restoring an uncooled mid-cosine checkpoint (`~57%` peak LR = `3.41e-5`–`3.43e-5`, cumulative `∫lr ≈ 0.00946–0.00950` vs Run 5's `0.01600–0.01605`) over heavily repeated Chat answers (`68` unique answers repeated `4..20` times).
- **Initial `v2.2` template/4-gram catches fixed offline at `$0.00`:**
  - Expanding `EXTRA_FOUNDRY_TOPICS_TRAIN`, `STANDARDS_RAG_TRAIN`, and `mt_chat` initially left `15` `EXTRA`, `3` `RAG`, and `mt_chat[10]` entries at `76–84 < 85` words (`_words` counts `[a-zA-Z0-9_'-]+`), plus a `js_bindgen` typo in `EXTRA[58]` and `>= 2×` repeated 4-grams inside `CHAT_TOPICS_TRAIN[3]` (`Linux 6.18.7` ×2) and `EXTRA[79]` (`or prism/spec.json` ×2). Expanded every entry cleanly to `85..133` words (`<= 220`), fixed `wasm-bindgen`, and removed all repeated 4-grams (`0` across all `1,200` train Chat rows).
- **Independent non-author FIT review #1 (`0704330d-c0af-4a5c-a648-9349f7311e2b` at `dffce15f8`) returned `TEST FITNESS: NOT_FIT` and caught three blocking issues fixed before final signoff:**
  1. Adding `prism/Admission.lex.tex` (`\section{Admission}`, `CapsuleAdmission`) as `TRAIN_BUILD_SPECS[31]` overlapped held-out `VAL_BUILD_SPECS[2]` (`val_pw_009..012`), `VAL_LEX_PATHS`, `VAL_PRISM_TARGETS`, and `VAL_READ_FILES`. Replaced it with held-out-disjoint `prism/CapsulePolicy.lex.tex` (`\section{PolicyRules}`, `CapsulePolicy`), verified `0` hits for `Admission.lex.tex` / `CapsuleAdmission` in `train.jsonl` and `{'src/App.tsx'}` as the only cross-split path (`0` `.lex.tex` overlap).
  2. Corrected swapped Chat failure counts in `ORDER.md` (`3` `too_few_words`, `2` `low_unique_ratio`, `3` `repeated_4gram:3` incl. `val_ch_070`) and overcounted `missing_expected_keyword` topic totals in `PREP.md` (`Topic 08`: `1/5` keyword fail `val_ch_043` / `2/5` pass; `Topic 13`: `1/5` keyword fail `val_ch_068` / `3/5` pass; `Topic 19`: `3/5` keyword fail `val_ch_096/098/100` / `2/5` pass).
  3. Corrected Run 8 historical figures in `PREP.md`: Run 8 Chat repeats were `4..20` (`max_target_repeat = 20`, not `<= 17`), Run 8 Epoch `1.0` LR was `~57%` peak (`∫lr ≈ 0.00946–0.00950`, `+30.3%–30.8%` lift to Run 9's `0.01238`, not `0.0087` / `+42%`), and Run 5 Epoch `1.0` `∫lr ≈ 0.01600–0.01605` (`-22.7%` to `-22.9%` vs Run 9).
- **No GPU pod launched in this PR:** `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`; actual `d90-foundry-lora-v9` GPU execution awaits pre-run approval after `antigravity-18` merges into `main`.

## What changed

- **Charter §4.5 Pre-Registration (`docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-18/`):**
  - [`ORDER.md`](ORDER.md): Task order & cancel criteria strictly bounded to `antigravity-18/**` (committed alone first at `e7022063d`).
  - [`PREP.md`](PREP.md): Pinned `d90-foundry-lora-v9` (`v2.2`) pre-registration documenting Run 8 exact failure counts, cumulative D90 ledger (`$2.2584` spent across Runs 1–8, `$0.7416` remaining of `$3.00` cap, Run 9 hard cap `1.1h / <= $0.58`), `3,600`-row diversified `v2.2` split (`DATA_SEED = 20261125`, `152` unique Chat targets with `7..10` repeats and `0` repeated 4-grams, `33` disjoint `.lex.tex`/UI build specs × `6` = `198` combos -> `708` rows, `120` `"You are a helpful assistant."` + `3,480` `FOUNDRY_SYSTEM_PROMPT`), single-epoch cosine schedule (`--epochs 1 --lr 1.1e-4`, `8%` warmup `18/225`, cooling to `0.0`, `0.50` `telemetry_only` + `0.75/1.00` eligible selection tracking `val_loss_tool` and `val_loss_chat`), `--max-new-tokens 448` passed explicitly, unchanged Bars #1–#5, safe archive protocol, and non-author `TEST FITNESS: FIT` signoff (`6ed06e086` by `0704330d-c0af-4a5c-a648-9349f7311e2b`).
  - Untracked local toolchain updated & verified offline at `$0.00` under `/Users/casey.allard/uor-r4/local/antigravity/d90-foundry-lora/v2/` (0 repo scripts or binaries committed).

## Measurements

- **Pinned local script SHA-256 digests (`local/antigravity/d90-foundry-lora/`):**
  - `v2/foundry_schemas_v2.json`: `56fe4459d45f0e29dd1ffceba4e34b704e7c9e3fafab0e76f3f67177ea52cc9a` (`system_prompt` sha256 `758c77e7b2be4006a2418f4bc87cdcceca8f864b1fa15b3e309a5e07489967f2`, `18` tools byte-identical to [`web/studio/foundry/studio/shell/r4-agent-loop.mjs`](../../../../../web/studio/foundry/studio/shell/r4-agent-loop.mjs))
  - `v2/generate_foundry_sft_v2.py`: `f6c368eba09ab825095b62d0e6d759c7ac53a150b34a72c6c43235f9eb7ab8ca`
  - `v2/train_foundry_lora_v2.py`: `5ca0aad611387ab3e7a9d1b45d32b570e7321ca26dd92b7bb6e623fd3c548c03`
  - `v2/eval_foundry_v13_v2.py`: `b71d9c7bfd5fea31725d846b7e89b4ab45852452c21af44890ce36e38f0433a6`
  - `v2/foundry_v13_eval_pack_v2.rs`: `0feb9b07d821b3a816700757f6c63b32b0e7ae835b501f2d01bca004c474c1b5` (unchanged from PR #4214)
  - `patch_recompile.py`: `bb5da659f37d7590791c723a230e21bc4dc6a4752003a94297ac0ccf775460de` (unchanged from Run 5 / PR #4214)
- **Deterministic dataset verification (`DATA_SEED = 20261125`):**
  - `train.jsonl` (`3,600` rows: `1,080` `tool_call`, `708` `prism_write`, `612` `multi_turn`, `1,200` `chat`; `152` unique Chat targets repeated `7..10`× [`max_target_repeat = 10`]; `0` repeated 4-grams; `120` `"You are a helpful assistant."` + `3,480` `FOUNDRY_SYSTEM_PROMPT`; `0` `ood_01..12` leakage; `0` `.lex.tex` path overlap): `21591aff6dc6193325e0a5b3d96aea85df43fa31ca747f70529167a85892bf5d`
  - `val.jsonl` (`440` rows: `288` `tool_call`, `52` `prism_write`, `100` `chat`; `440/440` `FOUNDRY_SYSTEM_PROMPT`; `train_val_overlap_hashes: 0`): `2c73655c81fd41f659f2917296d9dff9e9cec5f2fb611918d22191e38a939702`
  - `calib_foundry.txt` (`96` user-only rows from `train.jsonl`): `b70eb642450a2ca1b975f20b53b4fd496097fbd6b0f1d37139f0b3c6042faae2`
  - `parity_probe_prompts.json` (`16` stratified `val.jsonl` episodes -> `576` positions): `593b6114b79eb7ad10186eb9528fec3bd3e85c57b6cba559e71a7f391b3faba5`
- **Pod ledger & balance:** `uor-pod status` confirms `Running: 0 pod(s) ($0/h of $8.00/h)`; `uor-pod log` confirms Runs 1–7 `$2.0794` + Run 8 `vast-62270346` (`uptime_s: 1237` @ `$0.521/h` = `$0.1790`) = `$2.2584` cumulative spend, leaving `$0.7416` inside D90's `$3.00` cap (`<= $0.58` allocated for Run 9).

## Checks run at the head

- `python3 -m py_compile local/antigravity/d90-foundry-lora/v2/*.py` — PASS (`0` errors on final `v2.2` scripts).
- `python3 local/antigravity/d90-foundry-lora/v2/eval_foundry_v13_v2.py --self-test` — PASS (`self_test: "PASS"`, `agent_loop_verified: true`, `tool_count: 18`, `train_count: 3600`, `val_count: 440`, `calib_foundry_count: 96`, `train_val_overlap_hashes: 0`, `gold_eval: 340/340` tool + `100/100` Chat = `100.0%`, `bars_passed: true`, plus independent verification inside `0704330d-c0af-4a5c-a648-9349f7311e2b`).
- `python3 local/antigravity/d90-foundry-lora/v2/eval_foundry_v13_v2.py --wasm local/antigravity/d90-foundry-lora/foundry-v13-wasm/pkg/uor_foundry_v13_wasm_bg.wasm --holo-src /Users/casey.allard/lexlean-r4/crates/wasm/src/holo_geometric_runtime.rs` — PASS (`holo_inference_functions_scanned: 307`, `compute_float_ops: 0`, `js_abi_bridge_casts: 1`, `holo_runtime_src_noncomment_float_lines: 0`, `passed: true`).
- `0704330d-c0af-4a5c-a648-9349f7311e2b` (`pro` non-author fitness review on `6ed06e086c9825e5a54a80543530f4d7a3a3f2f0`) — `TEST FITNESS: FIT`.
- `uor-harness check --staged`, `uor-harness check`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --base origin/main --head HEAD --local` — verified cleanly in `.worktrees/antigravity-studio-phase4d-prep-v9` prior to push.

## Skills used

- Standard git, `python3 -m py_compile`, `wasm-tools print`, `uor-pod`, and `uor-harness` loop/ratchet checks.

## Still open

- Push `antigravity/studio-phase4d-prep-v9`, open PR, pass fresh-context `pro` exact-head adversarial review (`VERDICT: APPROVE`) + compliant `[antigravity] uor-review v1` comment, confirm `uor-harness loop pr-check --pr <N>` & GitHub Actions CI green, merge into `main`, clean worktree/branch, and hand back via `enter_plan_mode` for Run 9 pre-run approval (`d90-foundry-lora-v9`).

## Plain summary

Diagnosed Run 8's Bar #3 failure (`335/340` tool = `98.53%`, `74/100` Chat = `74.0%`, driven by uncooled Epoch `1.0` restoration, `4..20`× Chat target repetition, and dotted-version 4-gram loops), resolved a non-author FIT review catch that removed `prism/Admission.lex.tex` leakage via `prism/CapsulePolicy.lex.tex` (`0` `.lex.tex` train/val overlap), verified the `3,600`-row `v2.2` split (`152` unique Chat answers repeated `7..10`×, `0` repeated 4-grams) and `1`-epoch full-cosine schedule (`--lr 1.1e-4`, `∫lr = 0.01238`) offline at `$0.00`, obtained `TEST FITNESS: FIT` (`0704330d-c0af-4a5c-a648-9349f7311e2b` at `6ed06e086`), and prepared pre-registration `antigravity-18` (`<= $0.58` inside `$0.7416` remaining D90 headroom) for merge ahead of GPU launch review.
