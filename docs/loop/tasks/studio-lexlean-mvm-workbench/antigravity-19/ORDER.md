# ORDER for Qwen3-0.6B Foundry LoRA v2.3 Final D90 Pre-Registration (`d90-foundry-lora-v10`)

Lab: antigravity. Item: [studio-lexlean-mvm-workbench](../../../../roadmap/items/studio-lexlean-mvm-workbench.md), #3939. Base: `2fab62985` (PR #4218, `antigravity-18`). Pre-registers final Charter §4.5 GPU run `d90-foundry-lora-v10` (`v2.3`) under D90 (`$3.00` cap; Runs 1–9 spend `$2.5306`, `$0.4694` remaining) and D68 (`Bar #4b`). Documents Run 9 (`vast-62279565`, `$0.2722`) Stop-Rule outcome (`333/340` tool = `97.94%`, `77/100` Chat = `77.0%` Bar #3 FAIL: `8` `low_unique_ratio` BF16 loops lacking `HoloV13Model`'s cycle guard, `7` `missing_expected_keyword`, `3` `repeated_4gram:3`, `3` `too_few_words`, `2` `forbidden_tag`; `val_loss_tool 0.531->0.550->0.566` vs under-trained `val_loss_chat 2.927->2.898->2.905` restoring uncooled step `168`) and verifies `v2.3` (`local/antigravity/d90-foundry-lora/v2/`, zero repo commits) offline at `$0.00`: `4,000` train rows (`720` `tool_call`, `448` `prism_write`, `432` `multi_turn`, `2,400` `chat` across `158` unique targets repeated `15..16`× with `2,400/2,400` unique Chat prompts, `192` default + `3,808` Foundry system rows, `0` `.lex.tex` leakage, `0` repeated 4-grams), `lora_r=32/alpha=64/dropout=0.05`, `weight_decay=0.03`, `--epochs 1 --lr 1.3e-4` (`∫lr = 0.01625`, cooled `1.00`-epoch selection with `> 0.10` `val_loss_chat` overfit guard), and `HoloCycleGuardLogitsProcessor` matching `HoloV13Model::pick_next_holo_token`.

## Cancel probe

Verify `uor-harness loop pr-check` (`PASS order`, `commit-order`, `report`), `uor-harness check --staged`, ratchets `d23`/`d25`/`d32` (`0` added), zero `crates/**` / `src-tauri/**` / `web/studio/**` / `.github/**` / `local/**` edits committed, zero binaries (§3.13), `$0.00` unit check on `HoloCycleGuardLogitsProcessor` + `eval_foundry_v13_v2.py --self-test` (`4,000` train / `440` val / `0` overlap / `0` OOD leakage / `0` repeated 4-grams / `100%` gold eval), and non-author `TEST FITNESS: FIT` + fresh-context exact-head `VERDICT: APPROVE` on `antigravity-19/PREP.md` before Run 10 approval.

## Scope

Inside `docs/loop/tasks/studio-lexlean-mvm-workbench/antigravity-19/**` only (zero `owner-paths`, `crates/**`, `src-tauri/**`, `.github/**`, `web/studio/**`, or `local/**` committed):
- `ORDER.md`: task order & cancel criteria for final D90 Phase 4D pre-registration (`antigravity-19`).
- `PREP.md`: pre-registration sheet (`d90-foundry-lora-v10`) recording Run 9 observed metrics, cumulative D90 ledger (`$2.5306` spent / `$0.4694` remaining), pinned revisions, all 10 SHA-256 digests (`DATA_SEED = 20261126`), unchanged Bars #1–#5 (`--max-new-tokens 448`), hard GPU cap (`1× RTX 5090 <= $1.05/h`, max `0.75h / <= $0.40`), stop/no-run-11 boundary, and safe archive protocol.
- `REPORT.md`: negatives-first pre-registration report recording Run 9 failure taxonomy, initial `v2.3` template word-count catches fixed at `$0.00`, and FIT signoff.

## Done check

- `uor-harness check --staged`, `uor-harness check`, `uor-harness ratchet d23/d25/d32`, `uor-harness loop report --check`, `uor-harness loop pr-check --local`, non-author `TEST FITNESS: FIT` on `antigravity-19/PREP.md`, exact-head `VERDICT: APPROVE`, compliant `[antigravity] uor-review v1` comment (`uor-harness loop pr-check --pr <N>`), and GitHub CI all pass cleanly prior to merge.

## Stop and decide if

Stop and hand back to plan mode if non-author FIT review returns `NOT_FIT`, if `eval_foundry_v13_v2.py --self-test` fails any invariant, or if any harness/ratchet check blocks `antigravity-19/**`.
