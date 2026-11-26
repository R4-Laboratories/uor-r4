# STAGE B5b TEST SHEET: written-state-count-model, antigravity-15

Pre-registered under [`CHARTER.md` §4.5](../../../../CHARTER.md), [`D85`](../../../../integration/DECISIONS.md), and [`DECISION.md` §4 (`P3`)](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320). No tuning or hyperparameter selection runs on `DEV-test`.

---

## 1. Purpose & Sealed Evaluation Scope

- Execute the single-shot sealed D87 `DEV-test` (`5,321` records, `21,354,897` tokens, `21,349,576` non-BOS targets) evaluation of Stage B4's frozen `P3` predictors (`antigravity-12`) paired with cold-loaded verification of Stage B5a's frozen `S4` pointer table (`antigravity-14`, `c_ppmi_s4.uors4p01`).
- Resolve **all** `DECISION.md` §4 `P3` bars:
  1. Error-written rows achieve `>= 1%` fewer `DEV-test` bits than count-only rows (`<= 99% P1(A1s)` with 95% paired record-block bootstrap CI strictly `< 0`), with `R` mean Brier falling every epoch (`1, 2, 3`).
  2. Candidly report both overall gain vs `P1(A1s)` **and** controlled error-write vs static delta (`p4_full_errwrite - p4_cache_static` and `p4_c_skip_errwrite - p4_c_skip_static`) alongside standing against `0.95 / 0.90 / 0.80 × R*` (`1,347,229` / `1,276,322` / `1,134,508` micro-BPB; `R* = 1,418,136`).
  3. Cold-loaded frozen `S4` re-verifies `R1 >= 52/128`, `R2 >= 200/256`, and `R4 >= 200/240` beating `NoMemory` and 3-candidate `ShuffledRole` (`0` GPU/CPU diffs).

---

## 2. Frozen Pinned Inputs (`GO.json` verified before Gate 1)

| Input | Pinned SHA-256 | Provenance |
|---|---|---|
| `r.tokens.u16` | `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a` | Reserved `R` (`15,947,584` targets) |
| `r-65536.tape` / `pick.rows` | Stage B1a Wick artifacts | Verified against `R` & `a1s.weights` |
| `dev-test.json` (`c420`) | `ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2` | Sealed `DEV-test` census (`5,321` / `21,349,576`) |
| `dev-test.tokens.u16` | `023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b` | Sealed `DEV-test` tokens |
| `65536.features` | `f0e3432d3ecede6b54ae1d4782dd6a9ea87da280d2a8d04f456f88b2fc6a9951` | Sealed Stage B1b `UORCBF01` (`DEV-test`) |
| `rstar-devtest.masses.u64` | `59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de` | Canonical `R*` (`251,672,950,854` Q12 all) |
| `full-c.v3` (`compact_v3`) | `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233` | Stage B1a `UORKNC03` (`c.tokens.u16`) |
| `seed1.d11` (`trunk`) | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` | Frozen `Pick` trunk |
| `a1s.weights` / `a1s.seal.json` | `e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293` / `b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071` | Frozen Stage B1a `P1(A1s)` |
| `golden_shell.uors3t02` | `e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c` | Frozen Stage B2-Fallback `S3` |
| `s2_skip.uors2s01` | `85307b88f6bbc43d57fe8fe896789f9c8ea97e8e25bc24b074a3932e9332c777` | Frozen Stage B4 `C` `S2-skip` |
| `c_ppmi_s4.uors4p01` | `cd0e43859bcd72acfa08342d7ff08a0827c582c4cd8e1592218e6ca4c43f8771` | Frozen Stage B5a `S4` (`1,952,698` B) |
| `tokenizer.json` | `d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89` | Frozen ByteBPE tokenizer |
| `m2-fresh256` panel (`4` JSONs) | `0ed1ab98...` / `1de2fc28...` / `3b5c5477...` / `7c33f3f7...` | Emitted panel (`256` / `128`) |

### Stage B4 Zero-Drift Replay Pins (hard-asserted before `DEV-test`)
- `p4_c_skip_static.uors6b01`: `e0d5a65774d38279b84a5ccc56aa2966098f0eb6e7706e57dbc2be560518b79b`
- `p4_c_skip_errwrite.uors6b01`: `a0d50fdcf3971d793de42159f4271975ec43cc8e2c0cb3bec743ef4f187c71fb`
- `p4_cache_static.uors6b01`: `fe7f5d06a9717aaee5eb3eeac31b35eb8ce319072810a83ded4e27701333d285`
- `p4_full_errwrite.uors6b01`: `5783ba3b8b03cb4e9c35ed335b89c8ff36797175e9e1b8d9db40ddfd64f46a2d`

---

## 3. Acceptance Gates & Pass/Fail Outcomes

- **Gate 1 (`sass-check.txt`):** `ptx_float_opcode_hits == 0`, `float_opcode_hits == 0` across all `5` kernels (`nvcc -ptx` + `cuobjdump --dump-sass`).
- **Gate 2 (`finite.json`):** `525` finite probes (`256` Brier + `256` EM + `5` S3 + `4` S2 + `4` S4 incl. `role_class = 3`), `divergent == 0`, embedded `cubin_sha256` match.
- **Gate 3 (`devtest_report.json`):** `4/4` Stage B4 `UORS6B01` replay SHA-256 matches, monotone `R` Brier descent every epoch, `P1(A1s) == 246_099_676_775` all / `215_795_693_587` novel, `shifted_batch_parity == true`, `pick_probes_checked >= 13,000`, `spot_check_probes >= 25,000` (`0` divergences), plus frozen post-epoch-3 `p5b_s3_err_{p,b}.i8` & `p5b_s2_skip_errs.i8`.
- **Gate 4 (`reply_verify.json`):** Cold-loaded `c_ppmi_s4.uors4p01` (`0` GPU/CPU diffs) reproduces `R1 >= 52/128`, `R2 >= 200/256`, `R4 >= 200/240` beating `NoMemory` & `ShuffledRole`.
- **PASS:** `p4_full_errwrite` clears `<= 99% P1(A1s)` (`gate_brier_1pct_fewer_bits == true`), beats `p4_cache_static` cleanly (`ci < 0`), `R` Brier fell all 3 epochs, AND Gate 4 clears all `S4` panel bars -> **P3 PASSED**, unlocking **Phase `P4` M1 serving integration**.
- **FAIL:** Any P3 bar fails on sealed `DEV-test` with valid instrument (`0` divergences); close the failing configuration honestly without tuning on `DEV-test`.
