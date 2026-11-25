# ORDER: written-state-count-model, antigravity-7

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (#4063, M2+/P2). Base: `b870e2d7f`.

## Goal

Execute the pre-registered Stage B2b Wick GPU (`NVIDIA GeForce RTX 4090`, `$0.00` rented pod spend) `P2` calibration & `DEV-tune` equal-byte selection over the Stage B2a `UORS3T01` runtime (`DEV-test` untouched):

1. Build & audit `p2-s3` (`s3_kernels.sm_89.cubin`, `0` SASS float opcodes) implementing exact causal geometric transport (`e8` `E8Product` with Starling 7-neighbour miss + linear Pick) and all registered fair controls (`rand` `RandomTransport { seed: 20261125 }`, `tc_{1,2,3,4,6,8,10,12}` `TokenContext { k }`, `splitmix` `SplitMixPrefix { seed: 20261125 }`), passing finite `C128` bit-exactness against `uor_r4_count_runtime::s3_state`.
2. Materialize all 11 arms from frozen `C` (`c.tokens.u16`) at exact equal byte budget `<= 100,000,000 B` per `UORS3T01` table (leaving `>= 8.7 MB` inside `4,000,000,000 B` alongside Stage B1 `artifact.d11` + tokenizer), capture RTX 4090 `UORS3F01` tapes on `R` and `DEV-tune` at batches `65,536` and `65,521` with `0` diffs across `>= 12,500` native `S3StateTable::lookup_q48` probes per arm.
3. Fit 64-pass integer EM weights strictly on reserved `R` (`15,947,584` targets) for both standalone `100 MB` `S3` and combined `P1(A1s) + S3`, evaluate frozen on `DEV-tune` (`2,670` records, `10,807,417` targets) with 1,000-resample paired block bootstrap (`seed = 4390`), and report `U_geo` vs `U_count` on `DEV-tune`.

## Hard Lock on Sealed DEV-test

`dev-test.tokens.u16`, `dev-test.json`, and `rstar-devtest.masses.u64` MUST NOT be opened, read, or scored in `antigravity-7`. Winning arms advance to Stage B2c single-shot `DEV-test`; losing `e8` closes this `S3` configuration on `DEV-tune` and advances to `DECISION.md` fallback arm 1 (`VSA SplitMix`).

## Native-space cycle contract (D85)

1. **State & carried fields:** 12-bit `UORT` tokens (`0..4096`), 32-bit `seed1.d11` `Cl(0,6)` causal `F_b` / `S_b` transport blocks, `E8` 240-root codes (`0..240`), `57,600` product keys, `UORS3T01` counts/totals, `UORS3F01` (`hit_kind`, `predicted_root`, `output_pick_q16`, `mass_q48`), Q32 mixture weights, Q48 masses, and Q12 `bits_q12`. Zero floats, matmul, or softmax.
2. **Shared operators:** `uor_r4_count_runtime::s3_state` on host/verify and mirrored `s3_kernels.cu` on `sm_89` CUDA.
3. **Bit-identical transition replay:** Finite `C128` builder matches `build_equal_byte_s3_table` byte-for-byte; CUDA shifted batches `65,536 == 65,521` match `S3StateTable::lookup_q48` (`0` divergent probes).
4. **Pure-integer updates:** Deterministic ascending count cutoff pruning inside `<= 100,000,000 B` and 64-pass integer EM on `R`.
5. **Lossless serialization:** Self-describing `UORS3T01` + `UORS3F01` + sealed `GO.json` / `receipts.sha256`.

## Cancel probe (run first)

1. Confirm `uor-pod status` = `0` pods (`$0.00`), Wick RTX 4090 free, all pinned SHAs match, and `DEV-test` paths absent from `p2-s3` inputs.
2. Confirm `0` SASS floats and `0` finite `C128` builder/lookup divergences before running full-`C` `100 MB` builds.

## Files in scope

- `docs/loop/tasks/written-state-count-model/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `FIT.md` = `TEST FITNESS: FIT`, Wick Gates 1–5 pass with `0` post-FIT source drift and `0` SASS/parity errors.
2. `REPORT.md`, `uor-harness` gates, fresh `pro` exact-head review (`APPROVE`), and PR merge to `main` pass.

## Stop and ask if

- Any `s3_kernels.sm_89.cubin` SASS float opcode appears, `DEV-test` is touched, full-`C` per-arm runtime exceeds 3 minutes, or Wick GPU lock is contended.
