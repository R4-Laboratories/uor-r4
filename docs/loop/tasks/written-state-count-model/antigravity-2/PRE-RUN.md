# PRE-RUN: `written-state-count-model`, `antigravity-2`

- **Lab:** `antigravity`. **Order:** [ORDER.md](ORDER.md). **Sheet:** [SHEET.md](SHEET.md). **FIT:** [FIT.md](FIT.md) (`TEST FITNESS: FIT`).

## Whole-project chain (CHARTER §5.10)

1. **Goal & current milestone (`M2+` / `P0–P5`):** [`claude-409`](../claude-409/DECISION.md) puts the written-state model on one native D85 integer state inside `TrunkTables`. `P0` is complete (`claude-422`: canonical `R*` on sealed D87 `DEV-test` scored `1.418136` BPB all / `1.743746` novel). `P1(a)` is complete (`codex-33`/`codex-36`). `codex-37` fitted `A1s` (`UORS5W02`) and `A0` (`UORS5W01`) on reserved `R` (`15,947,584` targets) and selected `A1s` on frozen `DEV-tune` (`[−50, −31]` micro-BPB vs `A1`).
2. **This run (`P1(b)`):** One-shot sealed `DEV-test` evaluation of frozen `A1s` (`3,999,237,441` B deployed `UORLUT01` + tokenizer `<= 4,000,000,000` B) and no-S5 control `A0` paired against canonical `R*` (`1.03 × R*` gate).
3. **Pass / Fail unlocks:**
   - **PASS (`<= 1.03 × R*`):** Closes `P1(b)` PASS; `P2` geometric state rows (`S3` E8 snap + Starling miss read) unblock directly on the threshold-318 compact base.
   - **NEGATIVE (`> 1.03 × R*`):** Closes threshold-318 flat-triple pruning only as `NEGATIVE_PRUNING_COST` (§4.1), records exact Q12/micro-BPB pruning gap, and points to `DECISION.md` §3.5 compression (Elias-Fano trie / rank-quantized counts or optional order>6 suffix-array mmap).

## Verified readiness

- **Compute & caps:** Wick RTX 4090 (`GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, `24,080 MiB` VRAM free, zero foreign PIDs), `0` rented pods (`$0.00` Vast/Runpod spend), exclusive subshell `flock -n` on `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock`.
- **Finite controls passed:** `validate.json` (`5,321` records, `21,349,576` targets), `finite-scorer.json` (`49,158` words across `8,193` rows × 2 arms, `0` divergent + planted distance `2→3`), and `kn-1pm.json` (`--per-mille 1 --check-host --write-masses` bit-exact + overwrite refusal).
- **Non-author FIT verdict:** `TEST FITNESS: FIT` ([FIT.md](FIT.md)).
