# Non-Author Test Fitness Review: written-state-count-model, antigravity-14 (Stage B5a)

- **Reviewer:** Non-Author Test Fitness Reviewer (fresh-context pre-registration FIT audit under `CHARTER.md` §4.5 & §5.10)
- **Target Package:** `/Users/casey.allard/uor-r4/.worktrees/antigravity-b5a-polish/docs/loop/tasks/written-state-count-model/antigravity-14/`
- **Milestone / Issue:** `written-state-count-model` (`#4063`, `M2+/P3`, Stage B5a under `DECISION.md` §4)
- **Authorities Checked:** `CHARTER.md` §3.1–§3.13, §4.1–§4.5, §5.1–§5.10; `DECISION.md` §4–§5; `crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs`; `crates/uor-r4-discrete/src/count_runtime.rs`; `crates/uor-r4-core/src/answer_oracle.rs`; `crates/uor-r4-tokenizer/src/lib.rs`; `docs/labs/m2-fresh-heldout-panel/instrument/src/lib.rs`; `docs/loop/tasks/written-state-count-model/antigravity-13/REPORT.md`

---

## Verified 11-File SHA-256 Table

Verified via `shasum -a 256` directly on disk in `docs/loop/tasks/written-state-count-model/antigravity-14/`:

| # | File | Verified SHA-256 | Matches Target |
|---|---|---|---|
| 1 | `ORDER.md` | `2e4945bc697f09a523bd6147d49ec427a4daec1b10d93397b546650b7d7e2141` | Exact match (R2 follow-up: `Lab: antigravity.` backticks removed) |
| 2 | `SHEET.md` | `dc5213a0528a5c8ade7173034a4c34c2af42177ceeddaf4b0627f19d65eff48b` | Exact match |
| 3 | `dispatch.sh` | `291153e943f0ba9d1ec67c519224a58970f1040405ccbc6040f2da07e275a77b` | Exact match |
| 4 | `p5a-eval/Cargo.toml` | `031008d80e83520af1f92492a281d056a2f64ff2df8b8ed3aa9a070a1ba1ff7e` | Exact match |
| 5 | `p5a-eval/Cargo.lock` | `d15f15b66ceb0b8527fd4bc6b9c30ce3d1b9e4c79796811ecd36e2232cec60f1` | Exact match |
| 6 | `p5a-eval/build.rs` | `5ffe82ab867bc4de90dacd9443e64a79c0e47ee492ee28c823c85841c227a1f9` | Exact match |
| 7 | `p5a-eval/src/main.rs` | `2ddb4767808376e8b307bc2b9939966c4c58bca167223f8d389399a9192d7224` | Exact match |
| 8 | `p5a-eval/src/cuda.rs` | `6a66c1423b66f9e6353303603100fb3a9b588a7cc4722ed339e8e83cc57ef21c` | Exact match |
| 9 | `p5a-eval/src/c_ppmi.rs` | `01bfba1cf4dcec767708d9d56bbcf0bdc51a8df2dd62a8160bc8ca30f979f432` | Exact match |
| 10 | `p5a-eval/src/reply_eval.rs` | `c6410cdd3a48849ad223abd3b8a67133c6105057ef5e3a8b1ad32f489bb5d1fb` | Exact match |
| 11 | `p5a-eval/src/p5a_s4_kernels.cu` | `90ad047b111d1b712e7af2c067ceb4b30f9ade98464639491bdf37aaa03bd232` | Exact match |

---

## Per-Area Audit Findings

### 1. CHARTER §3 Compliance (Pure Integer, Zero Float/Matmul/Squash/NVRTC/Runtime CUDA Divide)
- **CUDA Kernel (`p5a-eval/src/p5a_s4_kernels.cu:1-109`):**
  - Strictly uses integer types (`U8`, `I8`, `U16`, `U32`, `U64`, `I64`, `int`). Regex + manual inspection confirm zero `f16`/`f32`/`f64`/`bf16`/`float`/`double`, zero softmax/squashing, zero matmul/GEMM/MMA, and zero runtime `/` or `%` operators anywhere outside comments.
  - `mul_div_u64` (`p5a_s4_kernels.cu:14-41`) performs exact 128-by-64 restoring binary division using `__umul64hi` and bit shifts. At the call site (`p5a_s4_kernels.cu:101`), `e_counts[idx]` is `U32` (`<= 2^32 - 1`), `P3_Q16_ONE = 1ULL << 16`, and `total = s4_totals[ci]` is the row sum across `<= u16::MAX` entries (`< 2^48`). Since `rem < total < 2^48`, `rem << 1` cannot overflow `U64`, `q <= P3_Q16_ONE`, and `total >= 1` is guaranteed by `S4ConjunctionTable` validation (`p3_state.rs:281-283,450-452`).
  - `bsearch_u32` (`p5a_s4_kernels.cu:43-58`) uses `lo + ((hi - lo) >> 1)` (shift, no division).
  - Kernel signature (`p5a_s4_kernels.cu:60-78`, 17 parameters) matches `cuda.rs:166-183` in exact count, order, and types, and matches `p3_state.rs:487-523` vote arithmetic (`((e.count << 16) / total) + e.err_i8 * 512`).
- **AOT Compilation & `unsafe` Isolation:**
  - `p5a-eval/build.rs:1-41` enforces `#![forbid(unsafe_code)]` and compiles `src/p5a_s4_kernels.cu` ahead-of-time via `nvcc -cubin -arch=sm_${digits} -O3 -x cu` (zero NVRTC, `CHARTER.md` §3.4), exporting `P5A_EVAL_CUBIN`, `P5A_EVAL_CAP_MAJOR`, and `P5A_EVAL_CAP_MINOR` consumed by `p5a-eval/src/cuda.rs:12,38-39`.
  - `#![forbid(unsafe_code)]` is enforced on `p5a-eval/build.rs:1`, `p5a-eval/src/c_ppmi.rs:1`, and `p5a-eval/src/reply_eval.rs:1`; `p5a-eval/src/main.rs:1` enforces `#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`; the sole `unsafe` block is `p5a-eval/src/cuda.rs:185` inside `#[cfg(feature = "cuda")] mod gpu` for `launch.launch(cfg)`.
- **Non-Vacuous Gate 1 PTX + SASS Enforcement (`dispatch.sh:186-210`):**
  - `dispatch.sh:187-190` compiles `p5a_s4_kernels.ptx` (`nvcc -ptx -arch=sm_89 -O3 -x cu`), dumps `p5a_s4_kernels.sass` (`cuobjdump --dump-sass "$CUBIN_PATH"` without piping to `grep -q`, avoiding SIGPIPE `141` under `set -euo pipefail`), and verifies `>= 1` occurrence of `.entry p3_s4_vote_batch` in PTX and `p3_s4_vote_batch` in SASS via `grep -c`.
  - `dispatch.sh:192-209` audits `PTX_HITS` (`\.(f16|f32|f64|bf16)\b` excluding comments/directives) and `SASS_HITS` (`FADD|...|HMMA|IMMA|BMMA|WMMA|QMMA|DMMA|F2I|I2F|F2F|...|RRO`), writes `sass-check.txt`, and exits `1` if either count is nonzero.

### 2. D85 Native-Space & GPU-First Compliance
- **Shared Operator Path (`CHARTER.md` §3.5, §3.7; `D85`):**
  - `p5a-eval/Cargo.toml:19-23` consumes `uor-r4-count-runtime` (`p3_state` with `#![forbid(unsafe_code)]`), `uor-r4-core`, `uor-r4-discrete`, `uor-r4-integer`, and `uor-r4-tokenizer` directly. All paths resolve 6 directory levels up (`../../../../../../`) to the worktree root cleanly.
- **Lossless Serialization (`c_ppmi_s4.uors4p01`):**
  - `reply_eval.rs:1250,1292-1295` builds `S4ConjunctionTable` keyed to `c_corpus_sha`, serializes lossless `UORS4P01` bytes via `encode_bytes()`, verifies immediate fail-closed round-trip via `S4ConjunctionTable::from_bytes(&bytes, c_corpus_sha, sha)`, and writes `out_dir/c_ppmi_s4.uors4p01`.
  - `cuda.rs:74-125` parses the exact 56-byte `UORS4P01` header, 18-byte context records, and 8-byte pointer entries matching `p3_state.rs:21-23,317-349`.
- **Bit-Identical GPU vs CPU Parity:**
  - **Gate 2 (`main.rs:24-148`, `cuda.rs:201-219`, `dispatch.sh:212-223`):** Tests 64 finite queries over 4 sorted `S4` rows (including a `role_class: 3` entry at `main.rs:59-64`, saturated `65_535` / `err_i8: 127` / `-127` boundaries, and an intentionally missing key at `main.rs:106-108`) across both candidate orderings `[(1,0),(2,0)]` and `[(2,0),(1,0)]`, asserting `cubin_sha256 == $CUBIN_SHA`, `s4_checked == 64`, `s4_divergent == 0`, and `divergent == 0`.
  - **Gate 3 (`reply_eval.rs:1264-1275,960-980,1366-1371`, `dispatch.sh:245`):** Checks 3-candidate GPU `s4_vote_batch` against CPU `S4ConjunctionTable::vote_candidates` on every one of the `72` training examples across all `3` epochs (`216` checks) and on all `m2-fresh256` (`256` `Active` + `256` `ShuffledRole`) and `R4` (`240` `Active` + `240` `ShuffledRole`) rows (`1,208` 3-candidate parity checks total), requiring `s4_gpu_vs_cpu_divergences == 0`.
- **Frozen Input SHA-256 Pinning:**
  - `dispatch.sh:146-172`, `c_ppmi.rs:12-13,441-446`, and `reply_eval.rs:22-35,431-448,1065-1101` verify exact SHA-256 digests on `c.tokens.u16` (`cd3adfe9...` computed on the streaming Pass 1), `full-c.v3` (`75ccb234...`), `r.tokens.u16` (`5875b173...`), `a1s.weights` + `a1s.seal.json`, `seed1.d11` (`04652a3d...`), `tokenizer.json` (`d36d3e87...`, also verified on local `.uor-models/research/issue-1019/tokenizer/tokenizer.json`), and `fresh256/{plan,inputs,labels,manifest}.json` (`0ed1ab98...`, `1de2fc28...`, `3b5c5477...`, `7c33f3f7...`).

### 3. Control Integrity, Algorithm Correctness & Zero Data Leakage
- **Sealed `DEV-test` Exclusion (`SHEET.md:9`, `dispatch.sh:15-19`, `main.rs:158-193`, `reply_eval.rs:36-39,306-313,1057-1063,1095`):**
  - `dispatch.sh:16-19` hard-refuses any `GO.json` matching `dev-test|rstar-devtest|023784b7|ec048217` with `exit 99`.
  - `main.rs:158-193` accepts only `"finite"` or `"reply"` subcommands (any `--devtest` subcommand fails with `unknown subcommand --devtest`, non-zero exit), while `reply_eval.rs:306-313,1057-1063,1095` rejects `dev-test`/`rstar-devtest` filenames and `SEALED_DEV_TEST_{CENSUS,UORT}_SHA256` (`ec048217...`, `023784b7...`) in addition to exact SHA-256 pinning of every input.
- **Zero Data Leakage (`verify_disjoint_sets`, `reply_eval.rs:43-96,257-303,589-592,1104-1135,1535-1567`):**
  - `verify_disjoint_sets` enforces zero exact lowercased string collisions and zero word 2-gram overlaps between `TRAIN` (`TRAIN_MWORLD_{JOB,HOME}_LITERALS`, `TRAIN_{JOB,HOME,NEUTRAL}_QUERIES`) and `HISTORICAL_FT_DEV_POOL` (`86` strings), `m2-fresh256` (`256` rows), and `R4` (`240` rows) in both unit tests and `run_reply_eval` before corpus build.
  - Independent check against `docs/labs/m2-fresh-heldout-panel/instrument/src/lib.rs` and `HISTORICAL_FT_DEV_POOL` confirmed `0` exact collisions and `0` 2-gram overlaps across all three sets, and verified that `TRAIN_NEUTRAL_QUERIES` nouns (`weather`, `season`, `color`, `music`, `book`, `meal`) have zero overlap with `m2-fresh256` or `R4` queries/literals.
  - In `collect_r_background_and_anchors` (`reply_eval.rs:589-592`), `TRAIN_NEUTRAL_QUERIES` is encoded into `neutral_q_toks` without calling `extract_query_atoms`, so neutral domain nouns never enter `job_anchor_atoms`, `home_anchor_atoms`, `train_query_toks`, or `shared_frame`, while shared frame pairs appear symmetrically across all three classes (`job`, `home`, `neutral`) in `TRAIN_MWORLD` (`24 * 3 = 72` examples, `reply_eval.rs:1194-1228`).
- **3-Candidate `ShuffledRole` Sattolo Parity & Abstention Symmetry (`reply_eval.rs:99-121,852-864,942-983` vs `p3_state.rs:82-87,524-573`):**
  - `reply_eval.rs:99-121` (`splitmix64` + `sattolo_permute_cands`) is bit-for-bit identical to `p3_state.rs:82-87,524-550`.
  - Under `candidates = [(1, 0), (2, 0), (3, 0)]` and `seed = 900 + idx`, the Sattolo loop produces only the two length-3 cyclic derangements (`[(2,0), (3,0), (1,0)]` on 124/240 seeds; `[(3,0), (1,0), (2,0)]` on 116/240 seeds), with zero fixed points (`assert_ne!` checked in `reply_eval.rs:1570-1620`).
  - Behavioral check on `(3, 0)` in `ShuffledRole`:
    - Any query where `Active` picks the gold role (`(1, 0)` or `(2, 0)`) maps that score under `ShuffledRole` to either the wrong role or `(3, 0)` (which maps to `role_spans.get(&(3, 0)) == None` -> abstains), yielding `0` hits.
    - Any query where `Active` picks `(3, 0)` (abstains) maps `(3, 0)` to slot `0` (`(1, 0)`, 116/240 seeds) or slot `1` (`(2, 0)`, 124/240 seeds) rather than systematically inverting `job <-> home` 100% of the time.
    - All-zero rows (`scores <= 0`) return `None` in both `Active` and `ShuffledRole` (`p3_state.rs:567`, `reply_eval.rs:975`); positive tie-breaking (`val > bs`, lowest index wins) matches bit-for-bit between `p3_s4_vote_batch` + `reply_eval.rs:973-979` and `p3_state.rs:563-572`.
- **Pass 1 `doc_freq`, `top_words = 65_536`, `top2_popcount`, and `union_per_anchor` (`c_ppmi.rs:15-20,193-334,408-476,542-886`):**
  - `PASS1_HASH_SLOTS = 1 << 22` (`4,194,304`), `PASS1_LOAD_LIMIT = 3,145,728`; `pack_word_atom` sets `len in 1..=5` in bits `60..63`, so `key >= 1 << 60 != 0` never collides with the empty-slot `0` sentinel.
  - `rec_id` starts at `1u32` (`last_rec` initialized to `0`), increments on `0`/`1` `rec_break` after counting any word closed at the delimiter (`c_ppmi.rs:415,427-434`), matching Pass 2 ring reset boundaries (`c_ppmi.rs:678-740`). Since `c.tokens.u16` has `2,075,608,357` tokens, `rec_id <= 2,075,608,358 < u32::MAX` (cannot saturate or overflow).
  - `WordIdLookup` (`c_ppmi.rs:265-304`) uses `u32` word IDs (`u32::MAX` sentinel) and `PASS2_HASH_SLOTS = 1 << 18` (`262,144` slots for `<= 65,550` words = `~25%` load), preventing the `u16` overflow that `top_words = 65_536` would otherwise trigger.
  - Basis filtering (`skip_top = 128`, `min_ctx_docs = 256`, `freq <= doc_freq * 32`), candidate filtering (`!is_stop`, `anchor_role == 0`, `freq >= 16`, `doc_freq >= 32`, `freq <= doc_freq * 32`), `u128` PPMI (`((cnt * tot_w) << 16) / (fw * cf).max(1) > (1 << 16)`, bounded `< 2^86 << 2^128`), `top2_popcount` (`<< 9`), `h1` (`< 2^54 << 2^64`), `4 * sj >= 5 * sh && sj > sh && h2_j >= (18 << 9)`, and `union_per_anchor` (`ov > 0`, sorted `(ov DESC, margin DESC, atom_key ASC)`, `per_anchor_cap = 256`, deduplicated into `BTreeMap` by `atom_key`, truncated to `prop_cap = 1_792 = 7 * 256`) match `SHEET.md:57-60` verbatim with zero arithmetic overflows or vacuous filters.

### 4. Gate Non-Vacuity & Scope Hygiene
- **Gate & Receipt Completeness (`dispatch.sh:90-286`):**
  - Enforces `L-002` fresh `RUN_DIR` (`lines 91-94`), Wick lock (`lines 97-104`), `CUDA_DISABLE_PTX_JIT=1`, `sm_89` compute cap and VRAM readiness (`lines 131-143`), all 11 package SHAs + binary/cubin + 11 frozen input SHAs + `FIT.md` `'TEST FITNESS: FIT'` & 11-SHA embed check (`lines 145-184`), Gate 1 PTX/SASS (`lines 186-210`), Gate 2 finite parity (`lines 212-223`), Gate 3 parity/volume/split assertions (`c_tokens_scanned == 2075608357`, `mworld_spans_extracted == 24`, `train_examples == 72`, `total_hinge_updates > 0`, `s4_gpu_vs_cpu_divergences == 0`, `fresh256` `256/128`, `R4` `240/120`), and `sha256sum --strict -c receipts.sha256` (`lines 271-283`). Note: `dispatch.sh` validates instrument integrity before sealing `receipts.sha256` so even a valid partial negative (`SHEET.md:76`) is preserved with sealed provenance rather than unsealed failure.
- **Scope & Formatting Hygiene:**
  - `git diff --name-only main...HEAD` and `git status --short` confirm changes are confined solely to `docs/loop/tasks/written-state-count-model/antigravity-14/**` (`ORDER.md`, `SHEET.md`, `dispatch.sh`, `p5a-eval/**`); zero `crates/**` files modified and zero binaries committed.
  - Read-only local checks passed: `bash -n dispatch.sh` (exit `0`), `rustfmt --edition 2021 --check p5a-eval/build.rs p5a-eval/src/*.rs` (exit `0`), and in-memory Gate 1 `grep -c` / PTX+SASS filter probe under `set -euo pipefail` (`PTX_HITS=0 SASS_HITS=0`). (Local macOS lacks `nvcc` and `sm_89` GPU, so `--features cuda` compilation and Gate 1–3 runtime execution run on the `sm_89` GPU host via `dispatch.sh`.)

---

## Blocking Defects / Non-Blocking Observations

- **Blocking defects:** **None.** Zero blocking defects found across SHA-256 verification, `CHARTER.md` §3 constraints, `D85` native-space parity/serialization, `DEV-test` isolation, data disjointness, 3-candidate `ShuffledRole` Sattolo parity, `c_ppmi.rs` streaming/filtering/quota logic, `dispatch.sh` gates, and worktree scope.
- **Non-blocking observations:**
  1. `p5a-eval/src/main.rs:161` still logs `"usage: p5-eval <finite|reply> ..."` and `p5a-eval/src/reply_eval.rs:309,1062` log messages say `"Stage B5"`; cosmetic string literals only, zero functional impact.
  2. `DEV-test` `exit 99` refusal is implemented in `dispatch.sh:16-19` on `GO.json`, while `main.rs:158-193` rejects unknown subcommands (such as `--devtest`) with non-zero error and `reply_eval.rs:306-313,1057-1063,1095` enforces filename + SHA-256 refusal (`refuse_sealed_devtest_sha`) plus full SHA-256 input pinning — accurately documented in `SHEET.md:9`.

TEST FITNESS: FIT — all 11 package SHA-256 digests match verbatim, `p5a_s4_kernels.cu` + `dispatch.sh` Gate 1 enforce zero-float/zero-MMA/zero-runtime-divide PTX+SASS execution, `D85` `UORS4P01` serialization and finite/3-candidate GPU-vs-CPU parity checks are non-vacuous, `DEV-test` is hard-locked untouched and `TRAIN` is verified disjoint from `ft/dev`/`m2-fresh256`/`R4`, 3-candidate `ShuffledRole` Sattolo derangement matches `p3_state.rs` bit-for-bit, `c_ppmi.rs` `doc_freq`/`top_words=65_536`/`top2_popcount`/`union_per_anchor` match `SHEET.md` cleanly, and zero `crates/**` files or binaries are in scope.
