# PRE-RUN REVIEW: `written-state-count-model`, `antigravity-4` (Stage B1a `UORKNC03`)

- **Reviewer (Top Session):** `antigravity`
- **Date:** 2026-11-25
- **Non-Author Test Fitness Verdict:** `TEST FITNESS: FIT` ([`FIT.md`](FIT.md), SHA-256 `7853715c6838c19f69d65f1a380990c05ceb5c5aba3bb55d6209750598a127e5`)

## Whole-Project Chain Check

1. **Goal (`ORDER.md`, SHA-256 `0324c406...`):** Close the Stage B1a deployment size vs accuracy bottleneck by packing full-`C` `UORKNC02` (`148,096,362,864` B) into lossless reversed-suffix trie `UORKNC03` inside `<= 3,912,936,331 B` (`<= 4,000,000,000 B` full `uor-r4.lut-discrete-count/1` deployment), re-capturing `R` and `DEV-tune` features with AOT CUDA (`cuda-v3`), re-fitting reserved-`R` S5 (`A0`/`A1`/`A1s`), and selecting on frozen `DEV-tune` strictly below `codex-37` (`1,581,444` micro-BPB).
2. **Inputs & Custody Present on Wick:**
   - `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/gpu-full-c/compact.counts` (`148,096,362,864` B, `1afd1c27...`)
   - `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/gpu-c128/compact.counts` (`16,459,406` B, `eea859b5...`)
   - `$HOME/uor-codex/20261119-rival-admission-29/count/cr-v1/c1.tokens.u16` (`41,508,666` B, `8d94e88a...`)
   - `$HOME/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` (`31,903,352` B, `5875b173...`)
   - `/mnt/plexpool/uor-r4-corpus/runs/20261120-pruned-r-fit-37/tune/c420/` (`dev-tune.json` `90fbc73a...`)
   - `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` (`86,536,132` B, `04652a3d...`), `inputs/tokenizer.json` (`265,393` B, `a9fb13f7...`), `inputs/c420/tokenizer.json` (`265,393` B, `d36d3e87...`)
   - `/mnt/plexpool` free space `40T` (`1%` used); `TMPDIR=/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/tmp`.
   - Sealed `DEV-test`: UNTOUCHED (`eval-v3` enforces `CENSUS_SHA = 90fbc73a...`).
3. **Host Gates Passed Before Dispatch:**
   - `uor-r4-discrete --lib kn_ceiling` (29/29 passed), `uor-r4-count-runtime`, `uor-r4-integer --lib` (222/222 passed), `discrete_sparse_table_parity` (6/6 passed, 3 ignored benchmarks), `discrete_chat_cli` (4/4 passed), `wasm32-unknown-unknown` build (`7/7` audit clean), `audit_zero_matmul_serving.py` (`60/60` FULL PASS).
   - `compress-v3` (1/1 passed), `cuda-v3` (1/1 passed), `eval-v3` (7/7 passed), `pack-v3` (1/1 passed).
4. **Compute & Cap:** `$0.00` pod spend (Wick local RTX 4090 under shared `wick-codex-gpu.lock` ONLY for active GPU steps).

**GO decision:** Proceed with Wick build + Gate 1 SASS audit, Gate 2 finite `C128`, Gate 3 full-`C` `histogram` + `write`, Gate 4 `cuda-v3` captures, Gate 5 `eval-v3` fit + `DEV-tune`, and Gate 6 streaming `uor-chat`.
