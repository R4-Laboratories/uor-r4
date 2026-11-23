# ORDER for Qwen3-0.6B Foundry Webharness LoRA v2 Pre-Registration (`d90-foundry-lora-v7`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `8849fd267` (PR #4213, Phase 4A). Pre-registers Charter §4.5 GPU run `d90-foundry-lora-v7` under D90 (`docs/integration/DECISIONS.md` line 294, `$3.00` cumulative Vast.ai cap; prior spend `~$1.85` across Runs 1–6) and D68 (`Bar #4b`). Builds and locally verifies the `v2` Studio SFT/eval/pack toolchain (kept untracked under `local/antigravity/d90-foundry-lora/v2/`) aligned byte-for-byte with `web/studio/foundry/studio/shell/r4-agent-loop.mjs` (`FOUNDRY_SYSTEM_PROMPT` & 18 `FOUNDRY_TOOL_NAMES`), 100–235 word multi-paragraph structured Chat answers without boilerplate tails, single/multi-file `<prism-write>` apps, Run 5's `96` user-only GPTQ calibration regime, and `--max-new-tokens 384` `UORR4V13` evaluation before renting any Vast.ai pod.

## Cancel probe

Verify `uor-harness loop pr-check` (`PASS order`, `PASS commit-order`, `PASS report`), `uor-harness check --staged`, `uor-r4-claim-wording`, ratchets `d23`/`d25`/`d32` (`0` added), zero `crates/**` / `src-tauri/**` / `web/studio/**` / `.github/**` edits, zero binaries (§3.13), `$0.00` local `eval_foundry_v13_v2.py --self-test` + `cargo check -p lexlean-r4-wasm --bin foundry_v13_eval_pack_v2` + WAT float audit (`compute_float_ops == 0`), and non-author `TEST FITNESS: FIT` + fresh-context exact-head `VERDICT: APPROVE` on `antigravity-16/PREP.md` before handback for pre-run review.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-16/**` only (zero `owner-paths`, `crates/**`, `web/studio/**`, or `local/**` committed):
- `ORDER.md`: task order & cancel criteria for the Phase 4D pre-registration sheet (`antigravity-16`).
- `PREP.md`: pre-registration sheet (`d90-foundry-lora-v7`) recording exact pinned HF/Git/tool SHAs, all 6 SHA-256 script digests, deterministic `DATA_SEED = 20261123` `train.jsonl` (`7,200` rows) / `val.jsonl` (`440` rows) / `calib_foundry.txt` (`96` rows) / `parity_probe_prompts.json` (`16` rows) digests (`0` train/val prompt overlap), Bars #1–#5 (`#2 >= 92.0%` over `340`, `#3 >= 95.0%` over `100` multi-paragraph quality bar, `#4a` 12-OOD `KL <= 0.08`, `#4b` D68 `576`-position telemetry), hard GPU cap (`1x RTX 5090 <= $1.05/h`, max `1.5h / <= $0.80`), and stop/unlock rules.
- `REPORT.md`: negatives-first pre-registration verification report recording `$0.00` local self-test, WAT float check, Rust `cargo check`, and deterministic seed outputs.

## Done check

- `uor-harness check --staged`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `antigravity-16/PREP.md`, exact-head `VERDICT: APPROVE`, `uor-harness loop pr-check --pr <N>`, and GitHub CI all pass cleanly prior to merging into `main`.

## Stop and decide if

Stop and hand back to plan mode if non-author fitness review returns `NOT_FIT`, if `eval_foundry_v13_v2.py --self-test` or `cargo check` fails against `r4-agent-loop.mjs` or `lexlean-r4-wasm`, or if any harness/ratchet check blocks `antigravity-16/**`.
