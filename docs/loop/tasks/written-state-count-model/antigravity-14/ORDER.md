# ORDER: written-state-count-model, antigravity-14

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Base: `8e3282748`.

## Goal

Pre-register and execute **Stage B5a (`P3` `S4` negatives & limits polish over `c.tokens.u16`, [`DECISION.md §4`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L304-L320))** to resolve all three `S4` negatives identified in Stage B5 ([`antigravity-13`](../antigravity-13/REPORT.md)) on `m2-fresh256` and `R4`, leaving sealed **`DEV-test` strictly untouched**:

1. **Document-Frequency Crawl-Noise Filter + `top_words = 65,536` + Per-Anchor Quota Union (`c_ppmi.rs`):**
   - Extend Pass 1 (`PASS1_HASH_SLOTS = 4,194,304`, `PASS1_LOAD_LIMIT = 3,145,728`) with per-atom record spread (`doc_freq`, `last_rec`, incrementing `rec_id` across `0`/`1` delimiters), expand Pass 2 active table to `top_words = 65,536` (`PASS2_HASH_SLOTS = 262,144`, `u32` word IDs), require `min_ctx_docs = 256` for the `1,024` PPMI basis slots and `min_cand_docs = 32` + concentration guard `freq <= doc_freq * 32` (`max_freq_per_doc = 32`) on candidates to strip localized non-English crawl noise (`dailctic`/`tunisian`/`derja`).
   - Score 2-hop similarity via `top2_popcount` (`2 * h2 + h1`, `4 * win >= 5 * lose`) and build `prop_job` / `prop_home` via per-anchor quota union (`per_anchor_cap = 256` per individual anchor atom, deduplicated up to `prop_cap = 1,792`) so street scenery no longer evicts rare residential OOV synonyms (`"dwelling"`, `"quarters"`).
2. **Role-Neutral Abstention (`role_class = 3`) & Exact 3-Candidate Sattolo GPU/CPU Parity (`reply_eval.rs`):**
   - Add disjoint frame-matched `TRAIN_NEUTRAL_QUERIES` (`weather`/`season`/`color`/`music`/`meal`/`book`) mapped to `role_class = 3` (`72` balanced hinge examples across `24` `TRAIN_MWORLD` episodes) so unanchored/ambiguous prompts abstain cleanly (`chosen_span = None`) instead of mis-voting `job`, driving `ShuffledRole` toward `0/240` while preserving `0` GPU-vs-CPU divergences via exact `splitmix64` 3-candidate Sattolo derangement.

## Native-space cycle contract (D85)

1. **State & carried fields:** Original 12-bit `UORT` tokens (`V = 4096`), `UORKNC03` (`full-c.v3`), `A1s` `UORS5W02`, `seed1.d11`, `UORS4P01`, and `LawfulStopConfig`. Zero floats, zero matmul, zero softmax.
2. **Shared operators:** Standalone `p5a-eval` depends on `uor_r4_count_runtime::p3_state` (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`), with mirrored pure-integer `p5a_s4_kernels.cu` audited via PTX/SASS `0`-float/`0`-MMA checks.
3. **Transition parity:** GPU vs CPU finite `p3_s4_vote_batch` (`64` probes, `0` diffs) and runtime 3-candidate panel parity including `ShuffledRole` (`s4_gpu_vs_cpu_divergences == 0`).
4. **Update rule:** `72` balanced `TRAIN_MWORLD` examples (`job`/`home`/`neutral`), `3` epochs `s4_hinge_step(margin_q16 = 16 * P3_Q16_ONE, step_i8 = 8)`.
5. **Lossless serialization:** Whole-payload SHA-256 + `train_sha256` on `UORS4P01`, sealed `GO.json`, and Wick `receipts.sha256`.

## Files in scope

- `docs/loop/tasks/written-state-count-model/antigravity-14/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. Non-author `pro` `FIT.md` (`TEST FITNESS: FIT`), `GO.json`, `rustfmt --check`, `cargo test`, `clippy -D warnings`, PTX/SASS `0`-float audit, and Wick `receipts.sha256` pass with `DEV-test` untouched and `0` pods owned by `antigravity`.
2. `REPORT.md` (all 7 canonical `## ` headings, negatives first), `m4-serving.md`, `written-state-count-model.md`, fresh `pro` exact-head review (`VERDICT: APPROVE`), and CI merge to `main` pass.
