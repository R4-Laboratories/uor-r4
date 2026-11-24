# ORDER for D91 Run 9 Closeout, Charter §5.1 / D68 Decision Review, & Return to M2+ (`antigravity-30`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `22245bb8a` (PR #4235, `antigravity-29`, Owner Decision [D91](../../../../integration/DECISIONS.md#decisions-after-the-charter-from-d43)). Records final D91 Run 9 (`d91-foundry-17b-v9`, `vast-62353488`, `uptime_s: 1411` @ `$0.508/h` = `$0.1991`; cumulative D91 `$2.0104 / +$3.00`, `$0.9896` unspent, `Running: 0 pod(s)`) passing Bar #1 (`compute_float_ops: 0`, `248` funcs) and BF16 Bar #2 (`97.65%` = `332/340`: `284/288` `<tool_call>`, `48/52` `<prism-write>`), while halting cleanly at Stop Gate #1 on BF16 Bar #3 (`92.0%` = `92/100 < 95.0%`, `total_failures: 16`, `val_loss_chat = 3.78158`, `30/440` CJK rows including `val_pw_010` `prism/spec潭.tex`, `val_ch_085` `\ufffd`, `val_ch_080` `<tool_call>`), executes the pre-registered finality & Charter §5.1 / D68 decision review across Runs 1–9 (`DECISION.md`) closing D91 `Qwen/Qwen3-1.7B` `v3`–`v3.5` LoRA training with `0` unverified weights published, preserves live `gh-pages` Studio untouched on verified `0.6B` (`PR #4213`), and returns focus to core native UOR-R4 M2+ milestones per D91.

## Cancel probe

Verify `uor-harness loop pr-check --local` (`PASS order`, `scope`, `commit-order`, `report`, `banned-presence`), `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, ratchets `d23`/`d25`/`d32` (`0` added), zero `owner-paths` / `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` commits, zero binaries (§3.13), `./scripts/pod/uor-pod status` confirming `Running: 0 pod(s)`, fresh-context `pro` `D68 DECISION REVIEW: APPROVE` on `DECISION.md`, and fresh-context `pro` exact-head `VERDICT: APPROVE` prior to merge (`$0.00` GPU spend).

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-30/**` ONLY (zero `DECISIONS.md`, `changes.md`, `PLAN.md`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` edits committed):
- `ORDER.md`: task order and cancel criteria for `antigravity-30` (`<= 4,096 B`, committed alone first).
- `DECISION.md`: Charter §5.1 / D68 decision card synthesizing complete Runs 1–9 (`v1..v9`) empirical evidence (`results_v3_run1..9/d91-out/` + `~/.local/share/uor-r4/compute/ledger.jsonl`), distinguishing the closed `v3`–`v3.5` `1.7B` LoRA configuration sweep from the `UORR4V13` runtime (§4.1–§4.2), closing D91 `1.7B` training at `$2.0104 / +$3.00` (`$0.9896` unspent, `0` unverified weights published), preserving live `gh-pages` Studio (`PR #4213`), and authorizing immediate transition back to core native UOR-R4 M2+ milestones (logged for Casey, reversible), plus fresh-context `pro` `D68 DECISION REVIEW: APPROVE`.
- `REPORT.md`: negatives-first closeout report recording Run 9 (`vast-62353488`, `$0.1991` / cumulative `$2.0104`, `results_v3_run9.tar.gz`, Bar #1 PASS, BF16 Bar #2 `97.65%`, Stop Gate #1 Bar #3 `92.0%`), `DECISION.md` closeout summary, offline host checks (`$0.00`), and `Running: 0 pod(s)`.

## Done check

- `uor-harness check --staged`, `uor-harness check --only prep arena briefing budget harness mistakes prlint receipts`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, `claim-wording`, fresh-context `pro` `D68 DECISION REVIEW: APPROVE`, exact-head `VERDICT: APPROVE`, `[antigravity] uor-review v1` (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass prior to merge.

## Stop and decide if

Stop and hand back to plan mode if any harness/ratchet check blocks `antigravity-30/**` or if the fresh-context `pro` D68 decision reviewer issues `REJECT`.
