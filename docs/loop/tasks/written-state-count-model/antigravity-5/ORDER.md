# ORDER: written-state-count-model, antigravity-5

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (#4063, M2+/P1(b)). Base: `de0a94ebd`.

## Goal

Close the sealed `DEV-test` paired `P1(b)` pruning-cost gate for Stage B1a's frozen `UORKNC03` candidate on Wick's RTX 4090 (`$0.00` pod spend) by:

1. Verifying sealed canonical `R*` (`59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de`, `170,796,608 B`, `251,672,950,854` Q12 all / `223,758,978,608` Q12 novel, `masses_at_floor == 4`) and frozen `artifact.d11` (`dd217c0d...`, `3,891,016,460 B` + `265,393 B` tokenizer `<= 4,000,000,000 B`).
2. Capturing and verifying `UORKNC03` `(T3*=2, T_hi*=6)` (`full-c.v3`, `75ccb234...`, `3,804,378,885 B`) `UORCBF01` tapes across sealed `DEV-test` (`5,321` records, `21,349,576` targets, `43,326,982` all B / `31,328,319` novel B) at widths `65,536` and `65,521` via audited AOT CUDA `cuda-v3` (`123df397...`, cubin `4c329f9f...`).
3. Scoring frozen `A1s` (`UORS5W02`, `e2fbf41d...`) and no-S5 control `A0` (`UORS5W01`, `9ec8168a...`) on `DEV-test` with `eval-v3-tune-device` (`9d717ca3...`), computing exact `100 * A1s <= 103 * R*` and 1,000-resample record-block bootstraps (seed `4390`).

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` (`BOS=0`, `EOS=1`, `V=4096`); `UORKNC03` counts (`u32[11]`, `u64[11]`, Q48 KN mass); `192`-token `IdentityControlV1` over `seed1.d11` (`04652a3d...`); lossless `UORS5W01`/`UORS5W02` Q32 weights -> Q48 mass and PRLL Q12 `bits_q12`. Zero floats, matmul, or softmax.
2. **Shared operators:** `cuda-v3` (`lookup_v3.cuh`) + `PickDevice` + `eval-v3-tune-device` on GPU; `CompactCountLookup` + `observe_identity_pick` + `CeilingTable::from_lookups_with_pick_classes` + `predict_mass` + `bits_q12` on CPU.
3. **Exact transition replay:** Phase A verifies `R*` `59f07609...`; Phase B checks `cmp 65536.features 65521.features` and `cuda-v3 verify` (`> 27,000` native probes, `0` diffs); Phase C checks `0` diffs across all `13,607` probes/arm plus `8,193`-row finite controls.
4. **Update rule:** Zero updates; single-shot readback and scoring of frozen `antigravity-4` artifacts.
5. **Lossless serialization:** `UORKNC03`, `UORS5W01`/`UORS5W02`, `UORP1A01`, `UORCBF01`, and LE `u64` `R*` masses.

## Cancel probe (run first)

1. Confirm clean worktree `/Users/casey.allard/uor-r4-b1b-devtest` from `main` (`de0a94ebd`), `$0.00` pod spend (`uor-pod status`), and Wick custody/binary SHA-256 verification.
2. Confirm Wick GPU `GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41` idle before acquiring `flock -n` on `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock`.

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-5/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Standalone `eval/` crate (`#![forbid(unsafe_code)]`), `SHEET.md`, `dispatch.sh`, non-author `FIT.md` (`TEST FITNESS: FIT`), `PRE-RUN.md`, and `GO.json` pass host checks (`rustfmt`, tests, `clippy`) and Wick finite controls.
2. Phases A, B, and C complete cleanly under `flock -n` with `0` word mismatches and compact receipts verified by `evidence/receipts.sha256`.
3. `REPORT.md`, `m4-serving.md`, and `written-state-count-model.md` record the exact `DEV-test` `P1(b)` result, passing all harness checks, `pro` exact-head review, CI, and squash merge.

## Stop and ask if

- Any pinned Wick SHA-256 mismatches, finite/replay probes diverge, or Wick GPU remains occupied past three scheduled waits.
