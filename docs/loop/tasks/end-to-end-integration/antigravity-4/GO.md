# Pre-Run Review (`GO.md`) — `antigravity-4`

- **Lab:** `antigravity`
- **Date:** `2026-11-28T02:35Z`
- **Decision:** `GO` (own pre-run review per `CHARTER.md` §5.1 / §5.10; reversible by Casey)

## Whole-Project Pre-Run Checklist

1. **Whole-project chain:** Unlocked directly by `antigravity-3` (PR `#4331`, merged on `main` `e87ef582c`) verdict `GATE1_DIAGNOSTIC_NEEDED`. Weight X-ray isolated three root causes of the universal step-`2,000` turnaround (`2.6873 BPB` floor): sparse-row Sign-Lion idle coasting (`embeddings` / `pkm.values`), uncapped Q12 norm drift (`+54%`), and identity `vocab_perm` + frozen PKM keys + `k=8` posterior amplification.
2. **Tools & pinned inputs verified:**
   - All local builds/tests pass (`54/54` `uor-r4-native-vector`, `17/17` `butterfly-diag`, `0` clippy warnings, `wasm32-unknown-unknown` clean, `d23`/`d25`/`d32` ratchets `OK`).
   - All 11 SHA-256 frozen inputs exist on Wick ZFS (`/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/corpus/`, `chat-v0-p2/heldout.u16`, `tokenizer.json`) and laptop (`local/antigravity/m2-fresh256/`). Sealed `dev-test.*` will NOT be copied to the pod.
3. **What pass and fail each unlock:** Defined exhaustively in `SHEET.md` §4 rules `(a)–(d)` (`DIAG_INT_ADAMW_BREAKTHROUGH` -> `L=8`/`L=12` pure-int scaling; `DIAG_PRECISION_GAP_ISOLATED` -> Q23/Q15 integer activations; `DIAG_CAPACITY_OR_LOSS_BOTTLENECK` -> `4×` butterfly FFN + `D=512` + lawful non-square loss; `INSTRUMENT_FAILED` -> stop & fix instrument).
4. **Non-author FIT:** `TEST FITNESS: FIT` recorded in `FIT.md` with exact SHA-256 digests (`98ff6b8f`, Round 3 after exact `tangent_pullback` clamp-boundary derivative fix caught by Gate 2 finite check).
5. **Compute lease inside global caps:** `uor-pod` leased `vast-62953159` (`2 × RTX 5090`, `$1.957/h`, well within `<= 4 pods`, `<= $8.00/h`), dispatching into fresh `L-002` output dir `/workspace/20261128-antigravity-gate1-diag-r2`, renewing lease every `<= 25 min`, and archiving to Wick ZFS (`/mnt/plexpool/uor-r4-corpus/runs/20261128-antigravity-gate1-diag-r2/`) + immediate `release` + `down` upon `GATE1_DIAG_COMPLETE` (`L-004`).
