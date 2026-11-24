# ORDER for D91 Run 1 Negative & Qwen3-1.7B `v3.1` Pre-Registration (`antigravity-22`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `9d3771b73` (PR #4224, `antigravity-21`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records the Run 1 (`d91-foundry-17b-v1`, `v3`) Stop-Rule outcome (`$0.1672` spent on `vast-62298484`, `$2.8328` remaining of D91's fresh `+$3.00` cap, `0` pods active) and pre-registers **Charter §4.5 GPU run `d91-foundry-17b-v2` (`v3.1`)** ahead of non-author FIT review and Run 2 GPU execution.

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` code commits, zero binaries (§3.13), `uor-pod status` confirming `Running: 0 pod(s)`, all `v3.1` offline host checks passing (`$0.00`), non-author `TEST FITNESS: FIT` on `PREP.md`, and fresh-context exact-head `VERDICT: APPROVE` prior to merge or any `uor-pod up`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-22/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-22` (`< 4,096 B`, committed alone first).
- `PREP.md`: Charter §4.5 pre-registration for `d91-foundry-17b-v2` (`v3.1` controlled ablation in untouched `v3/` pipeline + `v3/train_foundry_lora_v3_1.py` [`sha256: bf241ee50580a46b5facfec20e599031f3c4feff3afb016f0bd42ff1245a6566`], unchanged `val.jsonl` `06814bd9dcfece630998cc33699150da46afa7fae2102c046491e1300a8289a1`, Bars #1–#5, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, hard ceiling `<= $1.00 / 1.5h` inside D91's `$2.8328` remaining cap) plus non-author `TEST FITNESS: FIT` signoff.
- `REPORT.md`: negatives-first closeout report recording Run 1 (`vast-62298484`, `$0.1672`, Bar #1 PASS `compute_float_ops: 0`, baseline `49.41%`/`43.0%`, BF16 Bar #2 PASS `334/340` = `98.24%`, Stop Rule Gate #1 BF16 Bar #3 FAIL `76/100` = `76.0%`, exact 24-failure breakdown, monotonic under-trained `val_loss_chat: 2.74570 -> 2.65500 -> 2.62987`), `v3.1` offline unit verification (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-22/**`, if any `v3.1` offline unit test or `--self-test` fails, or if non-author FIT review returns `NOT_FIT`.
