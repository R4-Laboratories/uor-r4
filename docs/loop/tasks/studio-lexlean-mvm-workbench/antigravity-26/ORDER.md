# ORDER for D91 Run 5 Stop-Rule Negative & Qwen3-1.7B `v3.5` Pre-Registration (`antigravity-26`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `04cd332c7` (PR #4231, `antigravity-25`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records Run 5 (`d91-foundry-17b-v5`, `v3.4`, `vast-62322100`, `uptime_s: 3062` @ `$0.508/h` = `$0.4321`; cumulative D91 `$1.2280 / $3.00`, `$1.7720` remaining, `Running: 0 pod(s)`) passing Bar #1 (`compute_float_ops: 0`), BF16 Bar #2 (`97.35%` = `331/340`), and BF16 Bar #3 (`98.0%` = `98/100`) before halting cleanly at Stop Gate #2 on Bar #4a (`0.197846` with `--calib-file` and `0.243437` without `--calib-file` > `0.08`), and pre-registers **Charter §4.5 GPU run `d91-foundry-17b-v6` (`v3.5`)** ahead of non-author FIT review and Run 6 GPU execution.

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `scope`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` commits, zero binaries (§3.13), `uor-pod status` confirming `Running: 0 pod(s)`, all `v3.5` offline host checks passing (`$0.00`), non-author `TEST FITNESS: FIT` on `PREP.md`, and fresh-context exact-head `VERDICT: APPROVE` prior to merge or any `uor-pod up`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-26/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-26` (`<= 4,096 B`, committed alone first).
- `PREP.md`: Charter §4.5 pre-registration for `d91-foundry-17b-v6` (`v3.5` zero-leakage self-distillation KL anchor `--kl-anchor-weight 1.0 --kl-anchor-batch 2 --kl-anchor-tokens 32` over `64` disjoint train + `16` disjoint val `"You are a helpful assistant."` prompts alongside unchanged `v3.4` LoRA hyperparams & `val.jsonl` `06814bd9...` in `v3/train_foundry_lora_v3_5.py` [`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`]; primary no-`--calib-file` compile + bounded `--calib-file` fallback; `RUSTFLAGS="-C target-cpu=native"` untouched-source native eval binary; `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `2.0h / <= $1.10`) plus non-author `TEST FITNESS: FIT` signoff.
- `REPORT.md`: negatives-first closeout report recording Run 5 (`vast-62322100`, `$0.4321` / cumulative `$1.2280`, Bar #1 PASS, BF16 Bars #2/#3 PASS [`97.35%`/`98.0%`], Stop Gate #2 Bar #4a FAIL `0.197846`/`0.243437 > 0.08` + empirical base-vs-LoRA OOD drift & `7800X3D` eval timing diagnosis), `v3.5` offline verification (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-26/**`, if any `v3.5` offline check or `--self-test` fails, or if non-author FIT review returns `NOT_FIT`.
