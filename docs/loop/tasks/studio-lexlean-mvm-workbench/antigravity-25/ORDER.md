# ORDER for D91 Run 4 Stop-Rule Negative & Qwen3-1.7B `v3.4` Pre-Registration (`antigravity-25`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `9af20178c` (PR #4230, `antigravity-24`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records Run 4 (`d91-foundry-17b-v4`, `v3.3`, `vast-62316259`, `uptime_s: 1275` @ `$0.508/h` = `$0.1799`; cumulative D91 `$0.7959 / $3.00`, `$2.2041` remaining, `Running: 0 pod(s)`) passing Bar #1 (`compute_float_ops: 0`) and BF16 Bar #2 (`96.47%`) while halting cleanly at Stop Rule Gate #1 on Chat (`93.0% < 95.0%`), and pre-registers **Charter §4.5 GPU run `d91-foundry-17b-v5` (`v3.4`)** ahead of non-author FIT review and Run 5 GPU execution.

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `scope`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` code commits, zero binaries (§3.13), `uor-pod status` confirming `Running: 0 pod(s)`, all `v3.4` offline host checks passing (`$0.00`), non-author `TEST FITNESS: FIT` on `PREP.md`, and fresh-context exact-head `VERDICT: APPROVE` prior to merge or any `uor-pod up`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-25/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-25` (`< 4,096 B`, committed alone first).
- `PREP.md`: Charter §4.5 pre-registration for `d91-foundry-17b-v5` (`v3.4` gentle cooled `r=32, alpha=64, dropout=0.05, wd=0.015, ls=0.02, epochs=1.5, lr=1.3e-4, W=18, ∫lr=0.014625000` in `v3/train_foundry_lora_v3_4.py` [`sha256: eed988d5eafd9c7b59be174085d06c115ae14e2af4b49eb9ce091e2ca4457d35`], fractional-epoch `global_batch`/`accum_count` loop logging `{0.50, 0.75, 1.00, 1.25}` as `telemetry_only` and restoring ONLY cooled `1.50 ep` [`lr=0.0`], unchanged `val.jsonl` `06814bd9...`, Bars #1–#5 + Charter §4.1/§4.2 Bar #4a no-`--calib-file` fallback clause, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `1.5h / <= $1.00`) plus non-author `TEST FITNESS: FIT` signoff.
- `REPORT.md`: negatives-first closeout report recording Run 4 (`vast-62316259`, `$0.1799` / cumulative `$0.7959`, Bar #1 PASS `compute_float_ops: 0`, BF16 Bar #2 PASS `96.47%`, Stop Rule Gate #1 Chat FAIL `93.0% < 95.0%` + empirical Run 2 vs Run 4 early-LR diagnosis), `v3.4` offline unit verification (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-25/**`, if any `v3.4` offline unit test or `--self-test` fails, or if non-author FIT review returns `NOT_FIT`.
