# ORDER: written-state-count-model, antigravity-12

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Base: `aecd3675a`.

## Goal

Pre-register and execute **Stage B4 (`P3` disjoint-`C` skip-gram error-write + causal session cache + 2-hop `S4` propagation, [`DECISION.md §4`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320))** while keeping sealed `DEV-test` strictly untouched:

1. **Part A (`DEV-tune` `>= 1%` fewer bits than `P1(A1s)` `125,750,699,315` Q12 / `1,400,045` micro-BPB, 95% CI upper bound `< 0`, Brier falling every epoch):**
   - Build disjoint-`C` non-contiguous skip-gram conjunction tables (`S2-skip`: `skip_lag2 <= 20,000,000` B, `skip_3_1 <= 30,000,000` B, `skip_4_1 <= 30,000,000` B; total `<= 80,000,000` B built from `c.tokens.u16`) + `golden_shell.uors3t02` (`S3`), comparing count-only (`p4_c_skip_static`) vs 3-epoch GPU Brier error-write `lr_shift = 7` + integer EM `S6` (`p4_c_skip_errwrite`).
   - Add causal within-record written-state session cache experts (`cache_uni_recency`, `cache_bi_local`, `cache_tri_local`, `cache_lcp_local`, `0` B on disk, reset at `BOS = 0`, 3-state cache-hit bucket conditioning `880 * 3 = 2,640` `S6` buckets), comparing static cache (`p4_cache_static`) vs full unified candidate (`p4_full_errwrite`) on `DEV-tune` (`1,000`-resample block bootstrap `seed = 4390` vs `P1(A1s)`, `0.95/0.90/0.80 × R*`, and controlled `errwrite - static` deltas).
2. **Part B (`R4 >= 200/240` while preserving `R1 >= 52/128` & `R2 >= 200/256` on `m2-fresh256`):**
   - Filter `S4` seed anchors to word-initial (`starts_with(' ')`, byte len `>= 4`) role-exclusive `TRAIN_MWORLD` tokens (excluding top-64 `D87 R` stopwords), build a `4096 × 512` medium-frequency context matrix (`±6` record-bounded window over all `15,947,584` `D87 R` tokens) to extract exact 512-bit PPMI context signatures (`[u64; 8]`), score word-initial candidates via 1-hop co-occurrence + 2-hop bitwise `AND`/`popcount` similarity under `>= 1.2x` (`5*a >= 6*b`) role contrast (top-`384` per role, tiered counts `1..=4`) paired against top-`192` frequent scaffold tokens + `shared_frame`, run `3` epochs of `s4_hinge_step`, and score `m2-fresh256` (`R1`/`R2`) + `R4` (`240` rows) via GPU `p3_s4_vote_batch` against `NoMemory` and `ShuffledRole`.

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` tokens (`V = 4096`), `UORKNC03` (`full-c.v3`), `A1s` `UORS5W02`, `golden_shell` `UORS3T02`, disjoint-`C` `S2-skip` `UORS2S01` (`<= 80,000,000` B), causal within-record cache state (`BOS`-reset), `UORS6B01`, `UORS4P01`, and `LawfulStopConfig`. Zero floats, zero matmul, zero softmax.
2. **Shared operators:** Standalone `p4-eval` depends on `uor_r4_count_runtime::p3_state` (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`), with mirrored pure-integer CUDA kernels verified via PTX/SASS 0-float audit and finite/runtime parity.
3. **Transition parity:** GPU vs CPU finite parity (`0` diffs across all 5 kernels), `65,536 == 65,521` batch parity (`880` and `2,640` buckets), and `>= 12,500` spot-check probes (`0` divergences).
4. **Update rule:** Pure-integer EM & 3-epoch GPU Brier `i8` error-writes on `R`; `s4_hinge_step` on balanced `TRAIN_MWORLD` (`total_hinge_updates > 0`).
5. **Lossless serialization:** Whole-payload SHA-256 + `c_corpus_sha256` / `train_sha256` binding on `UORS2S01`, `UORS4P01`, and `UORS6B01`, sealed `GO.json`, and `evidence/receipts.sha256`.

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-12/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `pro` `FIT.md` (`TEST FITNESS: FIT`), `GO.json`, `rustfmt --check`, `cargo test`, `clippy -D warnings`, PTX/SASS 0-float audit, and `receipts.sha256` pass on archived Wick evidence with `DEV-test` untouched and `0` pods owned by `antigravity`.
2. `REPORT.md` (all 7 canonical `## ` sections, negatives first), `m4-serving.md`, `written-state-count-model.md`, fresh `pro` exact-head review (`VERDICT: APPROVE`), and CI merge to `main` pass.
