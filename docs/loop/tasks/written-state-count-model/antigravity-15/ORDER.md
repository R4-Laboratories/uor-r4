# ORDER: written-state-count-model, antigravity-15

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Base: `8447b71be`.

## Goal

Execute **Stage B5b: Single-Shot Sealed D87 `DEV-test` `P3` Qualification ([`DECISION.md §4`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320))** over `5,321` records (`21,349,576` non-BOS targets) combining Stage B4's causal session cache + disjoint-`C` `S2-skip` + `S3` (`p4_full_errwrite` [`antigravity-12`](../antigravity-12/REPORT.md)) with Stage B5a's frozen `C`-corpus PPMI `S4` (`c_ppmi_s4.uors4p01` [`antigravity-14`](../antigravity-14/REPORT.md)), without any tuning on `DEV-test`:

1. **Zero-Drift Frozen `R` Replay & Single-Shot `DEV-test` (`devtest_eval.rs`):**
   - Load frozen `s2_skip.uors2s01` (`85307b88...`), replay Stage B4's exact `3`-epoch `R` EM + Brier error-write loop (`p4_c_skip_static`, `p4_c_skip_errwrite`, `p4_cache_static`, `p4_full_errwrite`), and hard-assert byte-identical SHA-256 matches against Stage B4's four archived `UORS6B01` tables (`e0d5a657...`, `a0d50fdc...`, `fe7f5d06...`, `5783ba3b...`) plus monotone `R` Brier descent on every epoch.
   - Score `P1(A1s)` and all four frozen Stage B4 arms on sealed `DEV-test` (`dev-test.json` `ec048217...`, `dev-test.tokens.u16` `023784b7...`, `65536.features` `f0e3432d...`, `rstar-devtest.masses.u64` `59f07609...`), asserting `P1(A1s) == 246_099_676_775` Q12 all (`215_795_693_587` novel), `>= 25,000` GPU-vs-CPU parity checks (`0` diffs), paired bootstrap 95% CIs vs `P1(A1s)`, `0.95/0.90/0.80 × R*`, and controlled `errwrite - cache_static`.
2. **Cold-Loaded `S4` Verification (`reply_verify.rs`):**
   - Cold-load frozen Stage B5a `c_ppmi_s4.uors4p01` (`cd0e4385...`) and re-verify `m2-fresh256` (`R1`/`R2`) and `R4` GPU-voted scores (`0` diffs vs CPU).

## Native-space cycle contract (D85)

1. **State schema:** Original 12-bit `UORT` tokens (`V = 4,096`), `UORKNC03` (`full-c.v3`), `UORS2S01` (`s2_skip.uors2s01`), `UORS3T02` (`golden_shell.uors3t02`), `UORS6B01` (`22`-expert Q32 simplex), causal per-record `CausalSessionCache` (`0` B disk), `UORS4P01` (`c_ppmi_s4.uors4p01`), Q48 masses, Q12 bits. Zero floats, matmul, softmax, or runtime CUDA divides.
2. **Shared operators:** `uor_r4_count_runtime::{p3_state, s3_state}`, `uor_r4_discrete::kn_ceiling::FrozenPickMixture`, `PickDevice`, `lawful_likelihood::bits_q12`, and AOT `p5b_kernels.sm_89.cubin`.
3. **Bit-identical replay oracles:** Gate 1 PTX/SASS (`0` float/MMA), Gate 2 finite (`525` probes, `0` diffs), Stage B4 exact-hash `R` replay (`4/4`), `DEV-test` shifted-batch (`65536` vs `65521`), `P1(A1s)` reproduction, `>= 25,000` `DEV-test` + `S4` parity probes (`0` diffs).
4. **Update rule & lossless serialization:** Replay frozen Stage B4 `R` fits once; zero updates during `DEV-test` scoring; export post-epoch-3 `p5b_s3_err_{p,b}.i8` & `p5b_s2_skip_errs.i8` with `sha256sum --strict -c receipts.sha256`.

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-15/**` (standalone `p5b-eval`, `ORDER.md`, `SHEET.md`, `FIT.md`, `dispatch.sh`, `REPORT.md`, small `evidence/*`)
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `pro` FIT review passes (`TEST FITNESS: FIT`).
2. Gates 1–4 pass on Wick or `1x4090` Vast pod (`0` PTX/SASS floats, `0` divergences across `525` finite + `4/4` Stage B4 `UORS6B01` exact replay + `>= 25,000` `DEV-test` probes + `S4` `reply-verify`), archived on Wick (`receipts.sha256` `OK`, `0` pods left active).
3. `REPORT.md`, `m4-serving.md`, and `written-state-count-model.md` report full honest outcomes, passes `uor-harness` + fresh `pro` review, and PR merges to `main`.
