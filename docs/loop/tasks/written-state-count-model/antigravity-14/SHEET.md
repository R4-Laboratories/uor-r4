# Pre-Registration Sheet: written-state-count-model, antigravity-14 (Stage B5a)

- **Milestone / Issue:** `written-state-count-model` (`#4063`, `M2+/P3`)
- **Stage:** **Stage B5a** — `S4` Negatives & Limits Polish Over Broad-Domain `C` (`c.tokens.u16`) + Contrastive Abstention (`DECISION.md` §4)
- **Author lab:** `antigravity`
- **Base commit:** `8e3282748` (`origin/main`)
- **Charter authority:** `CHARTER.md` §3.1–§3.7, §4.1–§4.5, §5.1–§5.10; `D85` (`native-space-cycle`); `DECISION.md` §4 (`P1–P5`)
- **Non-author FIT reviewer:** Dedicated fresh-context `pro` subagent (recorded verbatim in [`FIT.md`](FIT.md))
- **Sealed boundary lock:** Sealed **`DEV-test` MUST remain strictly untouched**. Both `dispatch.sh` (exit `99` on any `dev-test|rstar-devtest|023784b7|ec048217` pattern in `GO.json`) and `p5a-eval` (`refuse_sealed_devtest_sha`) enforce hard refusal against sealed `DEV-test`.
- **Execution venue:** Wick workstation (`GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, lock `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock`, `compute_89`) if idle when launching; otherwise a single `1 × RTX 4090` (`compute_89`) Vast.ai pod provisioned and leased via `scripts/pod/uor-pod`, with `uor-pod status` confirming `0` active pods owned by `antigravity` after archival and teardown.

---

## 1. Exact Negatives Diagnosed in Stage B5 (`antigravity-13`)

In Stage B4 (`antigravity-12`), disjoint-`C` `S2-skip` + causal session cache + `S3` cleared `DEV-tune` BPB (`1,209,985` micro-BPB, `-13.58%` below `P1(A1s)`, `0.8447 × R*`), `R1 = 128/128`, and `R2 = 256/256`, while `R4` reached `160/240` (`120/120` in-stem, `40/120` OOV-synonym) on `15.95M`-token `D87 R`. In Stage B5 (`antigravity-13`), moving `S4` PPMI propagation onto broad-domain `c.tokens.u16` (`2,075,608,357` tokens) held `R1 = 128/128` (`NoMem 0`, `Shuf 0`) and `R2 = 256/256`, and cleared **`R4 = 200/240`** (`120/120` in-stem + `80/120` OOV-synonym), exposing three remaining `S4` limits documented in [`antigravity-13/REPORT.md`](../antigravity-13/REPORT.md#L7-L12):

1. **40/120 `R4` OOV-synonym miss (`"State my dwelling."` `0/20`, `"Clarify my quarters."` `0/20`):** Pass 1 tracked `279,602 << 3,145,728` unique atoms without hash saturation, so loss occurred in Pass 2 ranking where dense street-scenery tokens filled the single pooled `top_words = 32,768` / `top-1,024` `top3_popcount` quota ahead of residential nouns linked to `"residence"`/`"house"`/`"home"`.
2. **`ShuffledRole = 40/240` floor on `R4`:** Without an explicit role-neutral abstention basin (`role_class = 3`), unanchored OOV queries (`dwelling`/`quarters`) leaned slightly toward role `1` (`job`) under `Active`, causing `ShuffledRole` role inversion (`[(1, 0), (2, 0)] -> [(2, 0), (1, 0)]`) to flip those `40` wrong `job` rows into `40/240` `home` hits (`NoMemory = 0/240`).
3. **Non-English localized crawl noise in `top_prop_job_preview`:** Romanized clusters (`dailctic`, `arabic`, `tunisian`, `derja`) passed ASCII lowercase filtering and captured high-PPMI context bits due to high term frequency inside a narrow set of documents.

---

## 2. Frozen Inputs & Disjoint Data Guarantees

All input artifacts are pinned by SHA-256 and verified before Gate 1 in `dispatch.sh` and/or `p5a-eval`:

| Input | Path (Wick / Pod / Emitted) | SHA-256 | Role |
|---|---|---|---|
| `c_uort` | `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/input/c.tokens.u16` | `cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a` | Broad-domain training corpus (`2,075,608,357` tokens, `UORT` `u16` LE) |
| `compact_v3` | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate3-fullc/full-c.v3` | `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233` | `UORKNC03` continuation ranker over `C` |
| `r_uort` | `/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` | `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a` | `R`-corpus (`UORT`, background function-word frequencies) |
| `a1s_weights` | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.weights` | `e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293` | `A1s` (`UORS5W02`) carry-only integer scorer |
| `a1s_seal` | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.seal.json` | `b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071` | Seal binding `A1s` to frozen inputs |
| `trunk` | `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` | Frozen `seed1.d11` trunk |
| `tokenizer_json` | `.uor-models/research/issue-1019/tokenizer/tokenizer.json` | `d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89` | ByteBPE tokenizer (`V = 4096`) |
| `fresh256_dir/plan.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `0ed1ab98ab50bbad6c484c4e7e4ffeeb8bb7ad17d162251826302efe86d75715` | `m2-fresh256` eval plan (`128` pairs / `256` rows) |
| `fresh256_dir/inputs.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a` | `m2-fresh256` eval prompts (`256` rows) |
| `fresh256_dir/labels.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be` | `m2-fresh256` eval ground truth |
| `fresh256_dir/manifest.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `7c33f3f7e19edcb4b1a7aa5e5eab3bb0056db9351067d809f62f97b4a8880964` | `m2-fresh256` seal manifest |

### Strictly Disjoint Split Enforcement (`verify_disjoint_sets`)

- **Training pools (disjoint from all evaluation panels):**
  - Frame-matched `TRAIN_JOB_QUERIES` (`6` sentences covering `7` job anchors) and `TRAIN_HOME_QUERIES` (`6` sentences covering `7` home anchors).
  - Frame-matched `TRAIN_NEUTRAL_QUERIES` (`6` sentences covering neutral nouns `weather`/`season`/`color`/`music`/`meal`/`book`) mapped strictly to `role_class = 3` (not added to role anchors or shared role scaffold).
  - `TRAIN_MWORLD` (`24` episodes × `3` queries = `72` examples).
- **Held-out panels:**
  - `m2-fresh256`: `256` rows / `128` paired episodes (`R1` paired both complete & distinct of `128`, `R2` row hits of `256`).
  - `R4` held-out phrasing panel: `240` rows / `120` paired episodes (`120` in-stem + `120` OOV-synonym).
- `verify_disjoint_sets` asserts **0 exact lowercased string collisions** and **0 word 2-gram overlaps** between `TRAIN` (`TRAIN_MWORLD` literals + `TRAIN_{JOB,HOME,NEUTRAL}_QUERIES`) and all `ft/dev`, `m2-fresh256`, and `R4` strings.

---

## 3. Standalone Method (`p5a-eval` + `p5a_s4_kernels.cu`)

1. **Doc-Freq Crawl-Filtered Streaming + `top_words = 65,536` + Per-Anchor Quota (`c_ppmi.rs`):**
   - **Pass 1:** `PASS1_HASH_SLOTS = 4,194,304`, `PASS1_LOAD_LIMIT = 3,145,728`, tracking per-atom `counts`, `doc_freq`, and `last_rec` (`rec_id` starts at `1`, increments on `0`/`1` `rec_break`), plus streaming SHA-256 verification against `C_CORPUS_SHA256`.
   - **Pass 2 Selection:** `top_words = 65,536` (`PASS2_HASH_SLOTS = 262,144`, `u32` IDs/ring). Basis slots (`CTX_SLOTS = 1,024`) skip top `128` words and require `doc_freq >= min_ctx_docs` (`256`) and concentration guard `freq <= doc_freq * max_freq_per_doc` (`32`).
   - **2-Hop Scoring & Per-Anchor Union:** Filter candidate `w` by `!is_stop[w]`, `anchor_role[w] == 0`, `freq[w] >= 16`, `doc_freq[w] >= 32`, and `freq[w] <= doc_freq[w] * 32`. Compute `TOP_PPMI_BITS = 128` signatures `[u64; 16]` via `((cnt * tot_w) << 16) / (fw * cf).max(1) > (1u128 << 16)`. Score `h2 = top2_popcount << 9`, `h1 = ((co << 14) / (freq + 256)).min(65_536)`, `score = 2 * h2 + h1`, contrast `4 * win >= 5 * lose && h2 >= min_h2_scaled` (`18 << 9`). Rank contrast-passing candidates with `popcount_1024(sig[w], sig[a]) > 0` separately for each individual anchor atom lexicographically by `(popcount_1024(sig[w], sig[a]), margin, atom_key)` (`per_anchor_cap = 256` per anchor), deduplicate keeping max `margin`, and truncate at `prop_cap = 1,792` per class.
2. **Contrastive Multi-Token Stem Emission & 3-Candidate Abstention Hinge (`reply_eval.rs`):**
   - Multi-token atoms (`2..=5` subwords) emit deduplicated internal pairs `s4_pair_key(arr[a], arr[b])` (`0 <= a < b < len`) passing `4 * win >= 5 * lose` (`no t0 × scaffold leakage`).
   - Build `TRAIN_MWORLD` with `3` candidates `[(1, 0), (2, 0), (3, 0)]` across `24` episodes × `3` queries (`72` examples), run `3` epochs `s4_hinge_step(margin_q16 = 16 * P3_Q16_ONE, step_i8 = 8)` checking GPU/CPU parity on every epoch, and serialize lossless `c_ppmi_s4.uors4p01`.
3. **Pure-Integer GPU Scoring & 3-Candidate Sattolo Parity (`p5a_s4_kernels.cu` + `cuda.rs` + `reply_eval.rs`):**
   - `p3_s4_vote_batch` uses `bsearch_u32` and 128-by-64 `mul_div_u64` (zero floats/MMA/softmax/runtime divides).
   - For `shuffled_role`, `sattolo_permute_cands(&[(1,0),(2,0),(3,0)], 900 + idx)` matches `p3_state.rs` `S4ControlMode::ShuffledRole` `splitmix64` Sattolo derangement bit-for-bit across GPU and CPU (`s4_gpu_vs_cpu_divergences == 0`). Picking candidate `(3, 0)` abstains cleanly (`chosen_span = None`).

---

## 4. Acceptance Gates & Pass / Fail Unlocks

- **Gate 1 (PTX + SASS 0-Float/0-MMA):** `ptx_float_opcode_hits == 0` AND `float_opcode_hits == 0` (`nvcc -ptx` + `cuobjdump --dump-sass` non-vacuous entry checks).
- **Gate 2 (Finite Parity):** `s4_checked == 64 && s4_divergent == 0 && divergent == 0`, embedded `cubin_sha256` == `$CUBIN_SHA`.
- **Gate 3 (Full Pipeline + Parity):** `c_tokens_scanned == 2,075,608,357`, `s4_gpu_vs_cpu_divergences == 0`, `mworld_spans_extracted == 24`, `train_examples == 72`, `total_hinge_updates > 0`, `r1_r2_m2_fresh256.{rows == 256, pairs == 128}`, `r4_heldout_phrasing.{rows == 240, pairs == 120}`, `receipts.sha256` `sha256sum --strict -c` `OK`.
- **PASS:** Clear all Stage B5 bars (`R1 >= 52/128`, `R2 >= 200/256`, `R4 >= 200/240`, `beats_control == true` vs `NoMemory` & `ShuffledRole` under `paired_band_compare`) **and** confirm improvements against Stage B5 limits (hold `128/128` / `256/256`, lift `R4` / OOV recall above `200/240` [`80/120`], eliminate localized crawl noise in `top_prop_job_preview`, and drive `ShuffledRole` down via `(3, 0)` abstention). Unlocks **Stage B5b single-shot sealed `DEV-test` `P3` qualification** (`m2_devtest` + Stage B4 `p4_full_errwrite`).
- **FAIL / Partial Negative (Instrument Valid):** If any gate regresses below Stage B5 (`R4 < 200/240`, `R1 < 52/128`, `R2 < 200/256`) with `0` GPU/CPU divergences, record exact failure stratum in `REPORT.md` **without touching `DEV-test`** and decide whether Stage B5 (`c_ppmi_s4`) or tuned Stage B5a advances to Stage B5b.
