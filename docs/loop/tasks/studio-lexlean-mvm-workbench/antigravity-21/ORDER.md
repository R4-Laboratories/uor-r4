# ORDER for Decision D91 (`Qwen3-1.7B`) & Phase 5 Pre-Registration (`antigravity-21`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `f9bd3b425` (PR #4222, `antigravity-20`). Records Owner Decision **D91** in `docs/integration/DECISIONS.md`, `docs/rules/changes.md`, and `PLAN.md`, and pre-registers **Charter §4.5 GPU run `d91-foundry-17b-v1` (`v3`)** on `Qwen/Qwen3-1.7B` ahead of non-author FIT and GPU execution.

## Owner approval

Approved by owner Casey Allard in Antigravity chat at `2026-11-24T04:32:05Z` (authorizing updates to `docs/integration/DECISIONS.md` and `docs/rules/changes.md` plus `PLAN.md`):

- Owner verbatim answer selecting Option #1 followed by Option #3: `"Ok I want you to do #1 and then I want you to do #3, which is returning to the core native UOR-R4 M2+ milestones"`
- Meaning of approved options from the prompt presented at `2026-11-24T04:32:05Z`:
  - Option #1 (`D91`): Extend the temporary Studio-only (`web/studio/`) exemption of `D90` to `Qwen/Qwen3-1.7B` (`HF SHA 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, pure-integer `UORR4V13`, `compute_float_ops == 0`, `dense_matmul_ops == 0`, published to `Casey-allard/lexlean-r4@weights-v13-qwen3-1.7b-foundry`) with a fresh **`+$3.00` Vast.ai GPU budget cap** (tracked separately from D90's closed `$2.7427 / $3.00` `0.6B` ledger), keeping fast `Qwen3-0.6B` as the default Studio boot model and adding `Qwen3-1.7B (UORR4V13)` as an opt-in selector.
  - Option #3: After Phase 5 (`Qwen3-1.7B`) ships cleanly inside `web/studio/`, return focus to core native UOR-R4 M2+ milestones (`crates/**` and all native M1–M8 training/serving paths remain strictly §3-quarantined).

## Cancel probe

Verify `owner-paths` pass in `uor-harness loop pr-check --local` (`PASS scope`, `PASS order`, `PASS report`), `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, ratchets `d23`/`d25`/`d32` (`0` added), zero `crates/**` / `src-tauri/**` / `.github/**` / `web/studio/**` / `local/**` code commits, zero binaries (§3.13), `eval_foundry_v13_v3.py --self-test` (`2,400` train / `440` val / `0` overlap / `100%` gold), and non-author `TEST FITNESS: FIT` + exact-head `VERDICT: APPROVE` on `antigravity-21/PREP.md` before launching `d91-foundry-17b-v1`.

## Scope

Inside `docs/integration/DECISIONS.md`, `docs/rules/changes.md`, `PLAN.md`, and `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-21/**`:
- `docs/integration/DECISIONS.md`: Record D91 (`2026-11-24`) extending D90's Studio-only exemption to `Qwen/Qwen3-1.7B` with a fresh `+$3.00` Vast.ai cap and post-Phase-5 return to native M2+.
- `docs/rules/changes.md`: Record entry `#22` (`24 November 2026 — D91`) updating `Entries in this log` (`21` -> `22`).
- `PLAN.md`: Update the Studio owner-direction row with D90 closeout (`$2.7427 / $3.00`, PR #4222) and D91 Phase 5 approval.
- `antigravity-21/ORDER.md`, `antigravity-21/PREP.md`, `antigravity-21/REPORT.md`: Charter §4.5 pre-registration (`d91-foundry-17b-v1`, `v3`), offline `$0.00` verification, and non-author `TEST FITNESS: FIT` signoff (`0` pods active).

## Done check

- `uor-harness check --staged`, `uor-harness check --only charter,docs,loop,ratchet,claim,hygiene,prep`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local` (`PASS owner-paths`), non-author `TEST FITNESS: FIT` on `PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` (`loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if `owner-paths` or `scope` fails in `loop pr-check`, if non-author FIT review returns `NOT_FIT`, or if `eval_foundry_v13_v3.py --self-test` fails any invariant.
