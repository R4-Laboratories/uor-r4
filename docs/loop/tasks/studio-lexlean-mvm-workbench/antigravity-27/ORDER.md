# ORDER for D91 Run 6 Stop-Rule Negative & Qwen3-1.7B `v3.5` Tuned Pre-Registration (`antigravity-27`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `890dd10be` (PR #4232, `antigravity-26`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records Run 6 (`d91-foundry-17b-v6`, `v3.5`, `vast-62334503`, `uptime_s: 1603` @ `$0.508/h` = `$0.2262`; cumulative D91 `$1.4542 / $3.00`, `$1.5458` remaining, `Running: 0 pod(s)`) passing Bar #1 (`compute_float_ops: 0`, `248` funcs) and BF16 Bar #2 (`97.35%` = `331/340`: `285/288` tool calls, `46/52` builds) while cutting `anchor_val_kl_first` to `0.021992`, then halting cleanly at Stop Gate #1 on BF16 Bar #3 (`91.0%` = `91/100 < 95.0%`) due to over-strong `kl_anchor_weight=1.0` (`train_loss=1.22854` vs Run 5's `0.77225`), and pre-registers **Charter §4.5 GPU run `d91-foundry-17b-v7`** ahead of non-author FIT review and Run 7 GPU execution.

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `scope`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` commits, zero binaries (§3.13), `uor-pod status` confirming `Running: 0 pod(s)`, all `v3.5` offline host checks passing (`$0.00`), non-author `TEST FITNESS: FIT` on `PREP.md`, and fresh-context exact-head `VERDICT: APPROVE` prior to merge or any `uor-pod up`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-27/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-27` (`<= 4,096 B`, committed alone first).
- `PREP.md`: Charter §4.5 pre-registration for `d91-foundry-17b-v7` on untouched `v3/train_foundry_lora_v3_5.py` (`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`) with controlled 4-knob CLI tuning (`--kl-anchor-weight 0.25 --label-smoothing 0.05 --weight-decay 0.025 --lr 1.4e-4 --skip-baseline`), unchanged Bars #1–#5 (`--max-new-tokens 448` on `--mode eval`, `--mirror-url` explicit on `--mode pack`), primary no-`--calib-file` compile + bounded `--calib-file` fallback, `RUSTFLAGS="-C target-cpu=native"` untouched-source native eval binary, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `2.0h / <= $1.00`, plus non-author `TEST FITNESS: FIT` signoff.
- `REPORT.md`: negatives-first closeout report recording Run 6 (`vast-62334503`, `$0.2262` / cumulative `$1.4542`, `$1.5458` remaining, Bar #1 PASS, BF16 Bar #2 `97.35%`, Stop Gate #1 Bar #3 `91.0% < 95.0%` + empirical Chat underfitting diagnosis vs `antigravity-10/DECISION.md` & Run 5), `v3.5` offline verification (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-27/**`, if any `v3.5` / `eval_foundry_v13_v3.py` host check or pinned SHA-256 verification fails, or if non-author FIT review returns a design objection (`NOT_FIT`).
