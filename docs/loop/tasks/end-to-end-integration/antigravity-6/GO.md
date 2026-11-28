# Pre-Run Review (`GO.md`) — `antigravity-6` (`UORBFY03` Zero-Alloc Serving, `Entmax15` & `L=8`)

- **Lab:** `antigravity`
- **Date:** 2026-11-28
- **Decision:** **GO** (logged for Casey per `CHARTER.md` §5.1 / §5.10).
- **Order Commit:** `524899a97` (`ORDER.md`, `3,946 B <= 4,096 B`)

## Whole-Project Chain Check

1. **Prior milestones & unlock:** Gate 0 (`antigravity-1`, PR `#4316`), Gate 1 PR 1 (`antigravity-2`, PR `#4317`), Gate 1 PR 2 (`antigravity-3`, PR `#4331`), Gate 1 `2×2` Diagnostic (`antigravity-4`, PR `#4339`), and Gate 1 `UORBFY03` Capacity/Selectivity (`antigravity-5`, PR `#4347`, merged on `main` `c789325fa`) are complete. `antigravity-5` proved `UORBFY03 v1` works across every new mechanism (`chat-v0-p2 = 2.2980 BPB`, `-0.2362 BPB`; `DEV-tune E1_all = 2.6166 BPB`, `-0.3843 BPB`; `shuf_delta = +8,693`), while classifying `V3_PLATEAU_AT_RESCALED_SQUARE` (Rule `(e)`) because it sat `>= 2.00 BPB` and at `696 tok/s` single-core M1. Forensic X-ray isolated: (i) serving overhead (solved locally with `ButterflyV3Prepared` + zero-alloc `fwht_inplace_q15_with_scratch`, lifting `antigravity-5` Arm 2 `696 -> 1,892 tok/s` single-core M1 with `0` diffs vs `step()` and RTX 5090 `sm_120`), (ii) `83.2%` PKM vs `2.6%` trunk imbalance (fixed via `sk=32`, `L=8` inside `13.73 MB`), (iii) Hui-Belkin cap saturation and tail-cell shrinkage (tested head-to-head against coupled shift-invariant pure-integer `LossKind::Entmax15`), and (iv) under-trained 1-epoch horizon without answer-only SFT.
2. **Tools & pinned inputs verified:**
   - `uor-r4-native-vector`: `63/63` tests pass (`--release`), `clippy --lib -- -D warnings` clean (`0` warnings), `wasm32-unknown-unknown` clean.
   - `butterfly-v3b`: `16/16` host tests pass, `clippy --all-targets -- -D warnings` clean (`0` warnings), release build clean, `check_no_softmax.py` OK, `check_no_matmul.py` OK, `bash -n dispatch.sh` OK.
   - Pinned inputs verified on laptop and Wick ZFS (`dev-test.*` NEVER copied or opened):
     - Gate 0 corpus (`train-100m.{tokens.u16,mask.u8}`, `dev-tune.{tokens.u16,mask.u8,novel.u8,absent.u8}`, `id-bytes.u8`) at `/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate0-d87-proto2/corpus/`
     - `chat-v0-p2/heldout.u16` at `/mnt/plexpool/uor-r4-corpus/inputs/chat-v0-p2/heldout.u16`
     - `tokenizer.json` (`sha256: a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377`)
     - `m2-fresh256/inputs.json` (`sha256: 1de2fc28607eff6bc412eee41a59d806ee1479b7c925059f00e9d3b54afa99a`) & `labels.json` (`sha256: 3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be`)
3. **Non-author `TEST FITNESS: FIT`:** Recorded in `FIT.md` (Round 2 `FIT` from fresh-context `pro` reviewer `20c43115-14b9-4796-bd9b-bc4affa9ccb3` over all `10/10` exact SHA-256 digests).
4. **What pass and fail each unlock (`SHEET.md` §4, zero goalpost shifting):**
   - `(a) INSTRUMENT_FAILURE`: stop and repair instrument.
   - `(b) GATE1_FULL_PASS`: immediate Gate 2 (`737.5M` `TRAIN` + sealed `DEV-test`).
   - `(c) GATE1_BPB_AND_SPEED_BREAKTHROUGH`: scale winning loss (`Entmax15` vs `HB64`) directly to full `737.5M` `TRAIN` + wider/deeper `O(D log D)` trunk (`DEV-test` sealed).
   - `(d) LOSS_OR_HORIZON_PROGRESS_BELOW_BAR`: lock winning loss and advance to `antigravity-7` targeted capacity/supervision upgrade (`ReGLU` bilinear butterfly FFN / `8`-tap shift-mix / `737.5M` `TRAIN`; `DEV-test` sealed).
   - `(e) V3B_PLATEAU`: pivot note (`CHARTER §4.4`) & forensic X-ray before further GPU spend.
5. **Compute caps & `L-004` guard:** `uor-pod status` confirmed `0/4` pods running (`$0/h` of `$8.00/h`). Launched `1` Vast.ai `2 × RTX 5090` pod (`vast-62989215`, Quebec, CA, `$1.775/h`, session `gate1-v3b-r1`, `4 h` lease, renewed every `<= 25 min` via `schedule`). Output root: fresh `/workspace/runs/20261128-antigravity-gate1-v3b-r1` (`L-002`). Immediate `rsync` to laptop + Wick ZFS (`receipts.sha256 OK`) + `uor-pod release` + `uor-pod down` upon `GATE1_V3B_COMPLETE`.
