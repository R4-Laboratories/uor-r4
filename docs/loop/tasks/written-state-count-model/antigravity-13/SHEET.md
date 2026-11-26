# Pre-Registration Sheet: written-state-count-model, antigravity-13 (Stage B5)

- **Milestone / Issue:** `written-state-count-model` (`#4063`, `M2+/P3`)
- **Stage:** **Stage B5** — Broad-Domain `C`-Corpus ByteBPE Word-Atom 1024-Bit 2-Hop PPMI `S4` + Contrastive Stem-Pair Emission (`DECISION.md` §4)
- **Author lab:** `antigravity`
- **Base commit:** `bf4d9747a` (`origin/main`)
- **Charter authority:** `CHARTER.md` §3.1–§3.7, §4.1–§4.5, §5.1–§5.10; `D85` (`native-space-cycle`); `DECISION.md` §4 (`P1–P5`)
- **Non-author FIT reviewer:** Dedicated fresh-context `pro` subagent (recorded verbatim in [`FIT.md`](FIT.md))
- **Sealed boundary lock:** Sealed **`DEV-test` MUST remain strictly untouched**. Both `dispatch.sh` (exit `99` on any `dev-test|rstar-devtest|023784b7|ec048217` pattern in `GO.json`) and `p5-eval` (`refuse_sealed_devtest_sha`) enforce hard refusal against sealed `DEV-test`.
- **Execution venue:** Wick workstation (`GPU-7c813ad0-2151-bb1d-5efb-a3340b9a6a41`, lock `/home/callard/uor-codex/20261118-diagnostics/wick-codex-gpu.lock`, `compute_89`) if idle when launching; otherwise a single `1 × RTX 4090` (`compute_89`) Vast.ai pod provisioned and leased via `scripts/pod/uor-pod`, with `uor-pod status` confirming `0` active pods owned by `antigravity` after archival and teardown.

---

## 1. Exact Problem Diagnosed in Stage B4 (`antigravity-12`)

In Stage B4 (`antigravity-12`), pure-integer `R`-corpus `S4` + `s4_hinge_step` + `A1s` tiebreak cleared **`R1 = 109/128`** (`>= 52`), **`R2 = 228/256`** (`>= 200`), and **`R4_stem = 120/120`** (`100.0%` on held-out phrasings of seen stems), while **`R4_oov = 2/120`** (`R4_total = 122/240`, short of `200/240`) solely because `r.tokens.u16` (`2,420,971` tokens of coding/dialogue) lacked broad-domain lexical co-occurrence around unseen everyday words (`employment`, `career`, `profession`, `occupation`, `salary`, `wages`, `office`, `boss`, `residence`, `house`, `apartment`, `flat`, `street`, `city`, `town`, `village`, `neighborhood`, `rent`, `district`, `suburb`, etc.).

Stage B5 replaces tiny-corpus lexical mining with a single deterministic, integer-only streaming pass structure over **`c.tokens.u16` (`2,075,608,357` tokens)** plus contrastive multi-token stem-pair emission, **without touching any sealed held-out or `R4` text during training**.

---

## 2. Frozen Inputs & Disjoint Data Guarantees

All input artifacts are pinned by SHA-256 and verified before Gate 1 in `dispatch.sh` and/or `p5-eval`:

| Input | Path (Wick / Pod / Emitted) | SHA-256 | Role |
|---|---|---|---|
| `c_uort` | `/mnt/plexpool/uor-r4-corpus/runs/20261120-whole-context-36/input/c.tokens.u16` | `cd3adfe949ee34a62bc8d5e48592e091fcece796618d7910d1667d014a4bae2a` | Broad-domain training corpus (`2,075,608,357` tokens, `UORT` `u16` LE) |
| `compact_v3` | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate3-fullc/full-c.v3` | `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233` | `UORKNC03` continuation ranker over `C` |
| `r_uort` | `/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` | `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a` | `R`-corpus (`UORT`, verified non-contaminating function-word frequencies) |
| `a1s_weights` | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.weights` | `e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293` | `A1s` (`UORS5W02`) carry-only integer scorer |
| `a1s_seal` | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.seal.json` | `b6e9ac04006100922873d86562d6f525a9e5b73c713ca15d2dd329294d798071` | Seal binding `A1s` to frozen inputs |
| `trunk` | `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` | `04652a3de966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` | Frozen `seed1.d11` trunk |
| `tokenizer_json` | `.uor-models/research/issue-1019/tokenizer/tokenizer.json` | `d36d3e8700a123e620012df77de195f244fbdb4d05d9e1aa7e77fa9407590f89` | ByteBPE tokenizer (`V = 4096`) |
| `fresh256_dir/plan.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `0ed1ab98ab50bbad6c484c4e7e4ffeeb8bb7ad17d162251826302efe86d75715` | `m2-fresh256` eval plan (`128` pairs / `256` rows) |
| `fresh256_dir/inputs.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `1de2fc28607eff6bc412eee41a259d806ee1479b7c925059f00e9d3b54afa99a` | `m2-fresh256` eval prompts (`256` rows) |
| `fresh256_dir/labels.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `3b5c54771b8dc62e3c30aef2b30de28c09dfe79625f333854af046e6e354e2be` | `m2-fresh256` eval ground truth |
| `fresh256_dir/manifest.json` | `docs/labs/m2-fresh-heldout-panel/instrument` `--emit-dir` | `7c33f3f7e19edcb4b1a7aa5e5eab3bb0056db9351067d809f62f97b4a8880964` | `m2-fresh256` seal manifest |

### Strictly Disjoint Split Enforcement (`verify_disjoint_sets`)

- **Training pools (disjoint from all evaluation panels):**
  - Frame-matched `TRAIN_JOB_QUERIES` (`6` sentences covering `7` job anchor words: `job`, `employment`, `role`, `work`, `task`, `duty`, `status`) and `TRAIN_HOME_QUERIES` (`6` sentences covering `7` home anchor words: `home`, `street`, `address`, `house`, `residence`, `live`, `location`).
  - `TRAIN_MWORLD` (`24` balanced episodes using `TRAIN_MWORLD_JOB_LITERALS` and `TRAIN_MWORLD_HOME_LITERALS`).
- **Held-out panels:**
  - `m2-fresh256`: `256` rows / `128` paired episodes (`R1` paired both complete & distinct of `128`, `R2` row hits of `256`).
  - `R4` held-out phrasing panel: `240` rows / `120` paired episodes (`120` in-stem held-out phrasings + `120` OOV-stem phrasings).
- `verify_disjoint_sets` asserts **0 exact lowercased string collisions** and **0 word 2-gram overlaps** between `TRAIN` (`ft/dev` + `TRAIN_MWORLD` + `TRAIN_{JOB,HOME}_QUERIES`) and all `m2-fresh256` + `R4` queries and targets.

---

## 3. Standalone Method (`p5-eval` + `p5_s4_kernels.cu`)

1. **Pure-Integer Injective ByteBPE Word-Atom Streaming (`c_ppmi.rs`):**
   - Decode each `V = 4096` token once using `tokenizer.json`. Classify `is_word_start` (`' '` + `>=1` lowercase ASCII `'a'..='z'`) and `is_word_cont` (`>=1` lowercase `'a'..='z'`).
   - Pack sequences of `1..=5` BPE token IDs representing a word atom (`decoded trimmed len >= 3`) injectively into `u64`: `(len << 60) | (t0 << 48) | (t1 << 36) | (t2 << 24) | (t3 << 12) | t4`.
   - **Pass 1** over `c.tokens.u16`: record-bounded (`0`/`1` delimiters) open-addressed streaming frequency histogram (`PASS1_HASH_SLOTS = 4,194,304`, first-seen retention cap `PASS1_LOAD_LIMIT = 3,145,728`; once `filled == PASS1_LOAD_LIMIT`, unseen new atoms are not inserted) + exact SHA-256 verification against `C_CORPUS_SHA256`. Retain `top_words = 32,768` word atoms (`+` union with `TRAIN` anchor atoms -> `total_words <= 33,280`) and select `CTX_SLOTS = 1,024` medium-frequency context basis atoms (`ranks 128..1152`).
   - **Pass 2** over `c.tokens.u16`: build dense record-bounded `±6` word-window co-occurrence matrix (`active_words × 1,024` `u32`, ~`134 MiB`) + per-word totals (`u64`) and anchor-group (`job` vs `home`) 1-hop `anchor_idf`-weighted counts.
   - **Excess-PMI Signature & 2-Hop `S4` Propagation:** Compute `u128` fixed-point ratio `((cnt * tot_w) << 16) / (fw * cf).max(1) > (1u128 << 16)`, retain each active word's `TOP_PPMI_BITS = 128` highest-ratio basis indices packed into `[u64; 16]` (`1024`-bit signature), score every candidate word via `h2 = top3_popcount << 9` and `h1 = ((co << 14) / (freq + 256)).min(65_536)` with combined `score = 2 * h2 + h1`, and retain candidates clearing `>= 1.25x` contrast (`4 * win >= 5 * lose`, `prop_cap = 1,024` per role).
2. **Contrastive Multi-Token Stem-Pair Emission & Active `S4` Hinge (`reply_eval.rs`):**
   - For single-token propagated words (`tier` `8/6/4/2`): emit role-winning `S4` counts paired against shared frame scaffold tokens and top background function words.
   - For multi-token propagated words (`2..=5` subword tokens): accumulate all unique within-word internal subword pairs (`s4_pair_key(arr[a], arr[b])`, `0 <= a < b < len`) into `stem_pair_job` / `stem_pair_home` and emit only pairs clearing `4 * win >= 5 * lose` contrast (omitting short-prefix `t0 × scaffold` leakage so multi-token queries in `R4` trigger exact GPU `S4` conjunction votes cleanly).
   - Run `3` epochs of `s4_hinge_step` (`margin_q16 = 16 * P3_Q16_ONE`, `step_i8 = 8`, GPU vote parity checked on every epoch) over `TRAIN_MWORLD` (`48` role queries across `24` episodes) and serialize lossless `c_ppmi_s4.uors4p01`.
3. **Pure-Integer GPU Scoring (`p5_s4_kernels.cu` + `cuda.rs`):**
   - `p3_s4_vote_batch`: binary searches `u32` pair keys (`bsearch_u32`), scales integer count + `err_i8` via `mul_div_u64` (`__umul64hi` + restoring shift-subtract 128-by-64 division), zero floats, zero MMA, zero softmax, zero runtime divides.
   - Finite probe gate checks `64` queries against CPU `S4ConjunctionTable::vote_candidates` (`0` divergences allowed).
   - Runtime panel gate compares `GPU` vs `CPU` on **all** `m2-fresh256` (`256` rows × `3` control modes) + `R4` (`240` rows × `3` control modes) + training epochs (`s4_gpu_vs_cpu_divergences == 0`).

---

## 4. Acceptance Gates & Unambiguous Pass / Fail Unlocks

### Gate Requirements (Pre-Outcome Instrument Validity)

- **Gate 1 (PTX + SASS 0-Float Audit):** `ptx_float_opcode_hits == 0` AND `float_opcode_hits == 0` on compiled `p5_s4_kernels.cubin`.
- **Gate 2 (Finite GPU vs CPU Gate):** `s4_checked == 64 && s4_divergent == 0 && divergent == 0`.
- **Gate 3 (Full Pipeline + Parity):** `c_tokens_scanned == 2,075,608,357`, `s4_gpu_vs_cpu_divergences == 0`, `total_hinge_updates > 0`, `r1_r2_m2_fresh256.{rows == 256, pairs == 128}`, `r4_heldout_phrasing.{rows == 240, pairs == 120}`, valid `artifacts/c_ppmi_s4.uors4p01` and `receipts.sha256` (`sha256sum --strict -c receipts.sha256` `OK`).

### Pass / Fail Decision Criteria (`DECISION.md` §4)

- **PASS:**
  - `R1 >= 52/128` (`m2-fresh256` Active paired episodes both complete & distinct) AND `R2 >= 200/256` (`m2-fresh256` Active row hits), beating `NoMemory` and `ShuffledRole` paired controls; **AND**
  - `R4 >= 200/240` (`r4_heldout_phrasing` Active row hits), strictly beating `NoMemory` and `ShuffledRole` paired controls (`paired_band_compare` with `beats_control_by_band == true`, deterministic integer `net_gain > band_ceil`).
  - **Unlock on PASS:** Unlocks sealed Stage B5b **`DEV-test` qualification** (`m2_devtest`) under `DECISION.md` §4 and advances `written-state-count-model` (`#4063`) toward P3 completion.
- **FAIL / Partial Negative (Instrument Valid):**
  - If all Gates 1–3 pass cleanly (`s4_gpu_vs_cpu_divergences == 0`) AND **either** `R1 < 52`, `R2 < 200`, `R4 < 200`, or Active fails to beat `NoMemory` / `ShuffledRole`, classify the exact bottleneck (`R4_stem` vs `R4_oov`, missing stem coverage in `top_prop_{job,home}`, tiebreak dilution, or paired-control gap) in `REPORT.md` & `docs/research/notebook/m4-serving.md` **without touching sealed `DEV-test`** and design Stage B5-retry / next hypothesis per §4.1–§4.2.
  - **Negative closes only this configuration (`W = 32,768`, `CTX_SLOTS = 1,024`, `1024`-bit top-`3` PPMI, `>= 1.25x` contrast, 3-epoch hinge); never closes broad-domain `C`-corpus PPMI `S4` as a mechanism.**
