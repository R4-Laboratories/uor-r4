# ORDER for D91 Run 8 Stop-Rule Negative & Final `v3.5` Run 9 Pre-Registration (`antigravity-29`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `76dcf0f1e` (PR #4234, `antigravity-28`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records Run 8 (`d91-foundry-17b-v8`, `vast-62347334`, `uptime_s: 1229` @ `$0.508/h` = `$0.1734`; cumulative D91 `$1.8113 / +$3.00`, `$1.1887` remaining, `Running: 0 pod(s)`) passing Bar #1 (`compute_float_ops: 0`, `248` funcs) and BF16 Bar #2 (`97.65%` = `332/340`: `284/288` `<tool_call>`, `48/52` `<prism-write>`), while halting cleanly at Stop Gate #1 on BF16 Bar #3 (`91.0%` = `91/100 < 95.0%`, `total_failures: 17`: `3` `<tool_call>` prose leaks, `1` `\ufffd` `can\ufffd\ufffd证` glitch on `val_ch_016` at `--label-smoothing 0.02` falsifying the `0.05`-only hypothesis, `5` missing keywords; `cjk = 10`), and pre-registers final D91 **Charter §4.5 GPU run `d91-foundry-17b-v9`** restoring strong KL anchoring (`1.0`) with extended SFT step budget (`2.5` epochs).

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `scope`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` commits, zero binaries (§3.13), `./scripts/pod/uor-pod status` confirming `Running: 0 pod(s)`, all offline host checks passing (`$0.00`), non-author `TEST FITNESS: FIT` on `PREP.md`, and fresh-context exact-head `VERDICT: APPROVE` prior to merge or any `uor-pod up`.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-29/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-29` (`<= 4,096 B`, committed alone first).
- `PREP.md`: Charter §4.5 pre-registration for final D91 configuration **`d91-foundry-17b-v9`** on untouched `v3/train_foundry_lora_v3_5.py` (`sha256: 5e14f11b9f25e46ff16ebd68138be62a09767047f82d615a74bc3d34dd7fa76f`): restore Run 6's strong `--kl-anchor-weight 1.0 --weight-decay 0.015 --label-smoothing 0.02` (`0` CJK, `0` `\ufffd`, `anchor_val_kl_first = 0.021992`, `delta_w_rel_fro = 0.008689`, `val_loss_chat = 2.78726`) and raise step budget to `--epochs 2.5 --lr 1.4e-4 --skip-baseline` (`T = 375`, `W = 30`, `∫lr = 0.026250000` = `1.7949×` Run 6), unchanged Bars #1–#5, primary no-`--calib-file` + bounded `--calib-file` fallback (`< $0.55`), `UOR_POD_VAST_DISK_GB=80 UOR_POD_MAX_RATE=0.65`, `2.0h / <= $1.00` inside `$1.1887` remaining, explicit D91 finality rule (if Run 9 fails any Stop Gate, close D91 via Charter §5.1 / D68 decision review across Runs 1–9 and return to core native M2+), and non-author `TEST FITNESS: FIT` signoff.
- `REPORT.md`: negatives-first closeout report recording Run 8 (`vast-62347334`, `$0.1734` / cumulative `$1.8113`, `$1.1887` remaining, Bar #1 PASS, BF16 Bar #2 `97.65%`, Stop Gate #1 Bar #3 `91.0%`), Runs 1–8 CJK/`\ufffd` audit, offline checks (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, `claim-wording`, non-author `TEST FITNESS: FIT`, exact-head `VERDICT: APPROVE`, `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and CI all pass prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-29/**`, any `v3.5` / `eval_foundry_v13_v3.py` host check or pinned SHA-256 verification fails, or non-author FIT review returns `NOT_FIT`.
