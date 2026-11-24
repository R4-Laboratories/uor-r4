# ORDER for D91 Run 7 Stop-Rule Negative & Qwen3-1.7B `v3.5` Run 8 Pre-Registration (`antigravity-28`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `0abc5a85d` (PR #4233, `antigravity-27`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records Run 7 (`d91-foundry-17b-v7`, `vast-62341307`, `uptime_s: 1302` @ `$0.508/h` = `$0.1837`; cumulative D91 `$1.6379 / $3.00`, `$1.3621` remaining, `Running: 0 pod(s)`) passing Bar #1 (`compute_float_ops: 0`, `248` funcs) and reaching a Run 1–7 best BF16 Bar #2 (**`98.82%`** = `336/340`: `284/288` `<tool_call>`, `52/52` = `100.0%` `<prism-write>`) and lowest `total_failures: 10`, while halting cleanly at Stop Gate #1 on BF16 Bar #3 (`94.0%` = `94/100 < 95.0%`, 1 item short) after `--label-smoothing 0.05` introduced `2` `\ufffd` partial UTF-8 CJK (`验证`) glitches (`val_ch_016`, `val_ch_070`) vs `0/2,200` `\ufffd` across Runs 2–6 (`0.02`; `0/440` in Run 1 at `0.05`, `0/2,640` Runs 1–6), and pre-registers **Charter §4.5 GPU run `d91-foundry-17b-v8`** ahead of non-author FIT review and Run 8 GPU execution.

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `scope`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` commits, zero binaries (§3.13), `./scripts/pod/uor-pod status` from main checkout confirming `Running: 0 pod(s)`, all offline host checks passing (`$0.00`), non-author `TEST FITNESS: FIT` on `PREP.md`, and fresh-context exact-head `VERDICT: APPROVE` prior to merge or any `uor-pod up`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-28/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-28` (`<= 4,096 B`, committed alone first).
- `PREP.md`: Charter §4.5 pre-registration for `d91-foundry-17b-v8` on untouched `v3/train_foundry_lora_v3_5.py` (`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`) with surgical CLI tuning (`--kl-anchor-weight 0.30 --label-smoothing 0.02 --weight-decay 0.025 --lr 1.35e-4 --skip-baseline`, `∫lr = 0.015187500`), unchanged Bars #1–#5 (explicit `--mode eval` & `--logits /workspace/d91-out/teacher_parity_logits.bin` on Bar #4b, `--max-new-tokens 448` on `--mode eval`, `--mirror-url` explicit on `--mode pack`), primary no-`--calib-file` + bounded `--calib-file` fallback, `RUSTFLAGS="-C target-cpu=native"` untouched-source native eval binary, `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `2.0h / <= $1.00`, plus non-author `TEST FITNESS: FIT` signoff.
- `REPORT.md`: negatives-first closeout report recording Run 7 (`vast-62341307`, `$0.1837` / cumulative `$1.6379`, `$1.3621` remaining, Bar #1 PASS, BF16 Bar #2 `98.82%`, Stop Gate #1 Bar #3 `94.0% < 95.0%` + cross-run `\ufffd` audit `0/2,200` at `0.02` / `0/2,640` Runs 1–6 vs `2/440` Run 7), offline host checks (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-28/**`, if any `v3.5` / `eval_foundry_v13_v3.py` host check or pinned SHA-256 verification fails, or if non-author FIT review raises a substantive design objection (`NOT_FIT`).
