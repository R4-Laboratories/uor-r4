# ORDER for D90 Run 10 (`v2.3`) Stop-Rule & Phase 4D Closeout (`antigravity-20`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `ce35f93db` (PR #4221, `antigravity-19`). Records the executed Run 10 (`d90-foundry-lora-v10`, `v2.3`) Stop-Rule closeout outcome under D90 (`$3.00` cap) and closes out Phase 4D (`Qwen3-0.6B` v2 Foundry LoRA) with zero GPU pods active and zero Run 11 launches inside D90's `$3.00` cap, preserving all verified Run 10 receipts ahead of owner/plan transition to Phase 5 (`Qwen3-1.7B`).

## Cancel probe

Verify `uor-harness loop pr-check` (`PASS order`, `commit-order`, `report`), `uor-harness check --staged`, ratchets `d23`/`d25`/`d32` (`0` added), zero `crates/**` / `src-tauri/**` / `web/studio/**` / `.github/**` / `local/**` edits committed, zero binaries (§3.13), `uor-pod status` confirming `Running: 0 pod(s)`, and fresh-context exact-head `VERDICT: APPROVE` on `antigravity-20/REPORT.md`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-20/**` only (zero `owner-paths`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` committed):
- `ORDER.md`: task order & cancel criteria for D90 Run 10 & Phase 4D closeout (`antigravity-20`).
- `REPORT.md`: negatives-first closeout report recording the aborted SSH-stalled Czechia pod (`vast-62288296`, `$0.1044`) and `UOR_POD_MAX_RATE=0.60` fix, Run 10 (`vast-62289287`, `$0.1077`) Bar #1 WAT audit (`PASS`), baseline vs BF16 eval (`317/340` tool = `93.24%` PASS, `78/100` Chat = `78.0%` Bar #3 FAIL), monotonic `val_loss_chat` overfitting & qualitative confabulation taxonomy, exact cumulative D90 ledger (`$2.7427` spent / `$0.2573` remaining), live Studio continuity on PR #4213 Phase 4A, and Phase 5 (`Qwen3-1.7B`) existing `lexlean-r4` baseline receipt.

## Done check

- `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` comment (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-20/**` or if any cited Run 10 receipt or ledger number cannot be verified against archived ground truth.
