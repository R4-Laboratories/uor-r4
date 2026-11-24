# ORDER for Qwen3-0.6B Foundry Webharness LoRA v2.1 Pre-Registration (`d90-foundry-lora-v8`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `596fd86d3` (PR #4214, `antigravity-16`). Pre-registers Charter §4.5 GPU run `d90-foundry-lora-v8` (`v2.1`) under D90 (`docs/integration/DECISIONS.md`, `$3.00` cumulative Vast.ai cap; prior Runs 1–7 spend `$2.0794`, remaining `$0.9206`) and D68 (`Bar #4b`). Documents Run 7 (`d90-foundry-lora-v7`) Stop-Rule outcome (`314/340` tool = `92.35%`, `69/100` Chat = `69.0%` Bar #3 FAIL; `UORR4V13` `mean_KL = 0.128702` Bar #4a FAIL; `val_loss` monotonic rise `2.59591 -> 2.87949` from heavy template repetition) and verifies the upgraded local `v2.1` SFT/eval/pack toolchain (`local/antigravity/d90-foundry-lora/v2/`, zero commits to repo) featuring `3,200` diversified rows (`32` `<prism-write>` specs, `36` multi-turn chains, `24` OOD-disjoint English CS/engineering anchors split `120` `"You are a helpful assistant."` + `120` `FOUNDRY_SYSTEM_PROMPT`), `lr = 6.0e-5`, `weight_decay = 0.05`, `--max-new-tokens 448`, and `1.0 / 1.5 / 2.0`-epoch checkpoint selection at `$0.00` before renting any Vast.ai pod.

## Cancel probe

Verify `uor-harness loop pr-check` (`PASS order`, `PASS commit-order`, `PASS report`), `uor-harness check --staged`, `uor-r4-claim-wording`, ratchets `d23`/`d25`/`d32` (`0` added), zero `crates/**` / `src-tauri/**` / `web/studio/**` / `.github/**` / `local/**` edits committed, zero binaries (§3.13), `$0.00` local `eval_foundry_v13_v2.py --self-test` (`3,200` train / `440` val / `0` overlap / `0` OOD leakage / `100%` gold eval), and non-author `TEST FITNESS: FIT` + fresh-context exact-head `VERDICT: APPROVE` on `antigravity-17/PREP.md` before handback for Run 8 pre-run review.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-17/**` only (zero `owner-paths`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` committed):
- `ORDER.md`: task order & cancel criteria for Phase 4D Run 8 pre-registration (`antigravity-17`).
- `PREP.md`: pre-registration sheet (`d90-foundry-lora-v8`) recording Run 7 failure diagnosis, cumulative D90 ledger (`$2.0794` spent / `$0.9206` remaining), pinned HF/Git revisions, all 10 SHA-256 script & dataset digests (`DATA_SEED = 20261124`), unchanged strict Bars #1–#5 (evaluating at `--max-new-tokens 448`), hard GPU cap (`1× RTX 5090 <= $1.05/h`, max `1.25h / <= $0.68`), stop/unlock rules, and safe archive protocol (`pkill -f` self-kill prevention + verified non-empty local tar before teardown).
- `REPORT.md`: negatives-first pre-registration report recording Run 7 failure + empty-tar archive caveat, `v2.1` offline quality/leakage verification at `$0.00`, and non-author fitness signoff.

## Done check

- `uor-harness check --staged`, `uor-harness check`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `antigravity-17/PREP.md`, fresh-context exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` PR comment (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merging into `main`.

## Stop and decide if

Stop and hand back to plan mode if non-author fitness review returns `NOT_FIT`, if `eval_foundry_v13_v2.py --self-test` fails any invariant (`train/val` overlap, OOD leakage, schema match, or gold eval), or if any harness/ratchet check blocks `antigravity-17/**`.
