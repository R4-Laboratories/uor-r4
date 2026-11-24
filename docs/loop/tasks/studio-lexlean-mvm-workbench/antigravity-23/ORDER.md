# ORDER for D91 Run 2 Bar #4a Negative & Qwen3-1.7B `v3.2` Pre-Registration (`antigravity-23`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `41586a64c` (PR #4226, `antigravity-22`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records Run 2 (`d91-foundry-17b-v2`, `v3.1`) passing BF16 Bars #2 (`97.35%`) and #3 (`96.0%`) while halting at Stop Rule Gate #2 on Bar #4a (`mean_kl_nats = 0.191195 > 0.08`, `$0.2440` spent on `vast-62303346`, cumulative D91 `$0.4112 / $3.00`, `$2.5888` remaining, `0` pods active) and pre-registers **Charter §4.5 GPU run `d91-foundry-17b-v3` (`v3.2`)** ahead of non-author FIT review and Run 3 GPU execution.

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `scope`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` code commits, zero binaries (§3.13), `uor-pod status` confirming `Running: 0 pod(s)`, all `v3.2` offline host checks passing (`$0.00`), non-author `TEST FITNESS: FIT` on `PREP.md`, and fresh-context exact-head `VERDICT: APPROVE` prior to merge or any `uor-pod up`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-23/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-23` (`< 4,096 B`, committed alone first).
- `PREP.md`: Charter §4.5 pre-registration for `d91-foundry-17b-v3` (`v3.2` cooled OOD-safe `r=16, alpha=32, dropout=0.08, wd=0.03, epochs=1, lr=1.6e-4, ∫lr=0.012000` in `v3/train_foundry_lora_v3_2.py` [`sha256: 73b58dd242c55477756ba16b4583dfdba4b2f990a772fe09ca50e521c3c7639d`], `{0.75, 0.875, 1.00}` `select_best_checkpoint`, unchanged `val.jsonl` `06814bd9...`, Bars #1–#5 + Stop Rule diagnostic clause for Bar #4a, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `1.5h / <= $1.00`) plus non-author `TEST FITNESS: FIT` signoff.
- `REPORT.md`: negatives-first closeout report recording Run 2 (`vast-62303346`, `$0.2440` / cumulative `$0.4112`, Bar #1 PASS `compute_float_ops: 0`, BF16 Bar #2 PASS `97.35%`, BF16 Bar #3 PASS `96.0%`, Stop Rule Gate #2 Bar #4a FAIL `mean_kl_nats = 0.191195 > 0.08` despite `100%` top-1/top-5 parity and root cause), `v3.2` offline unit verification (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-23/**`, if any `v3.2` offline unit test or `--self-test` fails, or if non-author FIT review returns `NOT_FIT`.
