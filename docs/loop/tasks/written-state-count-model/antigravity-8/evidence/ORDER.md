# ORDER: written-state-count-model, antigravity-8

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (#4063, M2+/P2). Base: `6fdd03679`.

## Goal

Execute the pre-registered Stage B2-Fallback (`DEV-tune` `P2` equal-byte calibration `<= 100,000,000 B` per table) over all four pre-registered fallback address arms from [`DECISION.md`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L293-L298) in strict priority order (`1. vsa` -> `2. hopf` -> `3. h4_600cell` -> `4. golden_shell`) against all 10 equal-byte controls (`rand`, `tc_{1,2,3,4,6,8,10,12}`, `splitmix`, `14` arms total; `DEV-test` untouched):

1. Build & audit `p2-fallback` (`0` PTX/SASS float opcodes) implementing pure-integer `UORS3T02` tables + Starling 7-neighbour miss + affine Q16 output Pick across all 4 fallback arms:
   - **`vsa`**: VSA SplitMix64 64-bit XOR+rotate+vertical-majority binding (`window = 7`, `pk in 0..32768`).
   - **`hopf`**: Hopf $S^3 \to S^2 \times S^1$ 96-sector quantized bundle over `seed1.d11` `Cl(0,6)` transport (`pk in 0..9216`).
   - **`h4_600cell`**: Exact pure-integer 120-vertex 600-cell H4 cell snap over `Cl(0,6)` transport (`pk in 0..14400`).
   - **`golden_shell`**: Exact 600-cell 9-shell tuple index `(s1..s4)` over prior token H4 roots (`pk in 0..6561`).
2. Verify finite `C128` byte-for-byte builder/kernel parity and `0` divergent rows across all 14 arms (`0` SASS floats).
3. Materialize all 14 `UORS3T02` tables from frozen `C` (`c.tokens.u16`) at `<= 100,000,000 B`, capture GPU 3-word `UORS3F02` tapes on `R` and `DEV-tune` at batches `65,536` and `65,521` with `0` diffs across `>= 12,500` native probes/arm, fit 64-pass integer EM on `R`, evaluate frozen on `DEV-tune` (1,000-resample paired block bootstrap, `seed = 4390`), and report `oracle_u_geo_subset` vs `U_count` (`128,614,178,256 Q12` / `1,431,925` micro-BPB).

## Hard Lock on Sealed DEV-test

`dev-test.tokens.u16`, `dev-test.json`, and `rstar-devtest.masses.u64` MUST NOT be opened, read, or scored in `antigravity-8`. The first fallback arm clearing both standalone and combined `P2` gates advances to Stage B2c single-shot `DEV-test`; if all 4 fail, close them on `DEV-tune`, record the §3.1 owner-queue summary, and advance to Stage B3 (`P3`).

## Native-space cycle contract (D85)

1. **State & carried fields:** 12-bit `UORT` tokens (`0..4096`), 32-bit `seed1.d11` `Cl(0,6)` causal `F_b` / `S_b` transport blocks, integer 64-bit VSA hypervectors, Q30/i64 Hopf $S^2 \times S^1$ sectors (`0..96`), integer H4 120-root `PHI_Q30` codes (`0..120`), exact 600-cell shells (`0..=8`), `UORS3T02` count/total tables, 3-word `UORS3F02` capture tapes, Q32 weights, Q48 masses, and Q12 `bits_q12`. Zero floats, matmul, or softmax.
2. **Shared operators:** CPU `fallback.rs` reference and mirrored `s3_kernels.cu` CUDA kernels.
3. **Bit-identical transition replay:** Finite `C128` CPU/GPU builders byte-identical; shifted batches `65,536 == 65,521` match CPU lookup (`0` divergent probes across all 14 arms).
4. **Pure-integer updates:** Deterministic ascending count cutoff pruning inside `<= 100,000,000 B` and 64-pass integer EM on `R`.
5. **Lossless serialization:** Self-describing `UORS3T02` + `UORS3F02` + sealed `GO.json` / `receipts.sha256`.

## Cancel probe (run first)

1. Check Wick RTX 4090 idle state and `uor-pod status` (`<= 4` pods, `<= $8/h`), verify all pinned SHAs and absence of `DEV-test` paths.
2. Verify `0` SASS float opcodes and `0` finite `C128` divergences across all 14 arms before full-`C` `100 MB` builds.

## Files in scope

- `docs/loop/tasks/written-state-count-model/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `FIT.md` = `TEST FITNESS: FIT`, Gates 1–5 pass with `0` post-FIT source drift and `0` SASS/parity errors.
2. `REPORT.md`, `uor-harness` checks, fresh `pro` exact-head review (`APPROVE`), and PR merge to `main` pass (with any rented pod deleted).
