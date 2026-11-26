# Stage B5b Pre-Registration Test Fitness Review (FIT) — `antigravity-15`

Scope: Strict non-author, read-only FIT audit of `/Users/casey.allard/uor-r4/.worktrees/antigravity-b5b-devtest/docs/loop/tasks/written-state-count-model/antigravity-15/` against `CHARTER.md` (§3, §4, §5), `docs/integration/DECISIONS.md` (D85, D87), `docs/research/reviews/pi-review-2026-11-16/DECISION.md` (§2, §4 P3), Stage B4 (`antigravity-12`), Stage B5a (`antigravity-14`), and Stage B1b (`antigravity-5`).

---

## 1. Verified File Hashes & Local Hygiene Checks

All 13 files under `docs/loop/tasks/written-state-count-model/antigravity-15/` match the expected SHA-256 table byte-for-byte (`shasum -a 256` verified directly):

- `1123a55c0980a515aeeea4fb515a38f82ff95d2eae3349c950228d21f46242f0  ORDER.md`
- `d397bc67751f3a92df2b40e44e45803609c5a2e2e806acf85196da6e4493fc04  SHEET.md`
- `37a0c38c97c3773f96c5a4fe20855b29dcc0831a80e65de1bb0e84e2853f1e85  dispatch.sh`
- `7bafe7ef6c23cabbe766b4d3bbc0cddd8a2181e4d4522d6ae296b99420af69d9  p5b-eval/Cargo.toml`
- `2330ea3d48f601aecd5713038586d8a7648c666da28f9fd8f5debb473604fb52  p5b-eval/Cargo.lock`
- `de12f89c55313a41720cdb8bdd45ee63d9ef6cb05c323486b6056950aa1d0f4f  p5b-eval/build.rs`
- `8f7a15787af4f7451839ff14eb4867af1cff3710d5fb251515a68b939820288b  p5b-eval/src/p5b_kernels.cu`
- `e13944ff8c276d36cac913d38714d2f696aaf5590a92681d9bb0521c94ae9364  p5b-eval/src/cuda.rs`
- `b507eb2318c2ad18fc69df677c3d3b350e9415e67ead266d63b5a084bcf1458c  p5b-eval/src/cache_eval.rs`
- `32fd733f4c3158ac2d55add4a4f176e78246f81d9c0feba70532ea2f4c437560  p5b-eval/src/s3_decode.rs`
- `9026dcf12431aed618b957cf5441d22c49950a74dc39248326c23bc535975b6a  p5b-eval/src/devtest_eval.rs`
- `87e78c3de69e672a0ef9f1eee26ddeab715f06f27d379a68daeaf2998086f557  p5b-eval/src/reply_verify.rs`
- `265f9871ccac93feca7d14b645fe317a19f9fd8a1f131d568e20d1e8a1da9c05  p5b-eval/src/main.rs`

Local static checks:
- `bash -n dispatch.sh`: PASS.
- `rustfmt --edition 2021 --check p5b-eval/build.rs p5b-eval/src/*.rs`: PASS.
- Diffs against frozen Stage B4 / B5a references:
  - `p5b_kernels.cu` is byte-identical (`diff -u` empty) to `antigravity-12/p4-eval/src/p4_brier_kernels.cu`.
  - `cache_eval.rs` is byte-identical (`diff -u` empty) to `antigravity-12/p4-eval/src/cache_eval.rs`.
  - `cuda.rs`, `Cargo.toml`, `Cargo.lock`, `build.rs` differ from `antigravity-12` only by crate/env naming (`p4-eval` -> `p5b-eval`, `P4_EVAL_*` -> `P5B_EVAL_*`, `p4_brier_kernels.cu` -> `p5b_kernels.cu`).
  - `reply_verify.rs` preserves Stage B5a's `antigravity-14/p5a-eval/src/reply_eval.rs` logic for `build_r4_heldout_cases`, `extract_causal_role_spans`, `free_generate_case`, `evaluate_panel_mode`, and `sattolo_permute_cands`, while dropping `R`-side training/PPMI construction in favor of cold-loading `FROZEN_S4_SHA256` (`cd0e4385...`).

---

## 2. Audit Findings by Requirement

### A. Zero Tuning on `DEV-test` & Exact Stage B4 Replay (`devtest_eval.rs`)
- **Strict train/eval separation:** `run_devtest_eval` (`devtest_eval.rs:1343-1436`) loads `seed1.d11` (`TRUNK_SHA256`), `a1s.seal.json`/`a1s.weights`, `golden_shell.uors3t02` (`GOLDEN_SHELL_SHA256`), and `s2_skip.uors2s01` (`S2_SKIP_SHA256`), and executes `replay_stage_b4_on_r` (`devtest_eval.rs:1398-1413`) **before** `DevTestData::load_checked(devtest_dir)` (`devtest_eval.rs:1435`).
- **Frozen `R` replay & exact Stage B4 receipt pins:** `replay_stage_b4_on_r` (`devtest_eval.rs:702-1315`) verifies `R_UORT_SHA256` (`5875b173...` at `:717`), validates `b1a_r_tape` (`UORCBF01`) & `b1a_r_pick` (`R37PK001`) against `C_CORPUS_SHA256`, `COMPACT_V3_SHA256`, `R_UORT_SHA256`, `TRUNK_SHA256`, and `PickClassRule::Five` (`:242-340, 724`), asserts monotone `R` mean Brier descent every epoch (`:995`, `:1258`), and hard-asserts exact SHA-256 matches for all 4 archived Stage B4 `UORS6B01` tables against `antigravity-12/evidence/receipts.sha256`:
  - `p4_c_skip_static`: `e0d5a65774d38279b84a5ccc56aa2966098f0eb6e7706e57dbc2be560518b79b` (`:46-47`)
  - `p4_c_skip_errwrite`: `a0d50fdcf3971d793de42159f421975ec43cc8e2c0cb3bec743ef4f187c71fb` (`:48-49, 1024-1030`)
  - `p4_cache_static`: `fe7f5d06a9717aaee5eb3eeac31b35eb8ce319072810a83ded4e27701333d285` (`:50-51`)
  - `p4_full_errwrite`: `5783ba3b8b03cb4e9c35ed335b89c8ff36797175e9e1b8d9db40ddfd64f46a2d` (`:52-53, 1287-1293`)
- **Clean reset & exact `err_i8` plumbing:** Between Arm 2 and Arm 3/4 on `R`, `gs_table` (`primary_rows`, `backoff_rows`), `s2_skip`, and GPU `gpu_gs` / `gpu_s2s` error buffers are zeroed (`devtest_eval.rs:1032-1046`). Arm 2 (`arm2_p_errs`, `arm2_b_errs`, `arm2_s2_errs` at `:1016-1022`) and Arm 4 (`arm4_p_errs`, `arm4_b_errs`, `arm4_s2_errs` at `:1279-1285`) are saved and exported (`p5b_s3_err_p.i8`, `p5b_s3_err_b.i8`, `p5b_s2_skip_errs.i8` at `:1416-1432`) prior to `DEV-test`. During `DEV-test`:
  - `p4_c_skip_static` explicitly passes zeroed `err_i8` arrays (`:1690-1700`).
  - `p4_c_skip_errwrite` explicitly passes `replayed.arm2_*_errs` (`:1782-1792`).
  - `p4_cache_static` uses only `c_p1` + `c_cache_cols` (`:1823-1838`).
  - `p4_full_errwrite` explicitly passes `replayed.arm4_*_errs` (`:1850-1869`).
  - All `eval_s3_s2` calls pass `None` for `mixed_q48` (`:1649-1659`, verified in `cuda.rs:249-253` and `p5b_kernels.cu:254, 281, 373`), guaranteeing zero error writes or parameter updates on `DEV-test`.

### B. Causal Scoring, Cache Invariants, Baselines & Statistics (`devtest_eval.rs`, `SHEET.md`)
- **Sealed census & exact canonical baselines:** `DevTestData::load_checked` (`devtest_eval.rs:145-205`) verifies `DEV_TEST_CENSUS_SHA256` (`ec048217...`), `DEV_TEST_RECORDS = 5_321`, `DEV_TEST_TOKENS = 21_354_897`, `DEV_TEST_TARGETS = 21_349_576`, `DEV_TEST_UORT_SHA256` (`023784b7...`), `mask.u8`, `segment.u8`, `novel.u8`, and `id-bytes.u8` against the census digests. `aggregate_rstar_devtest` (`:1317-1340`) checks `RSTAR_MASSES_SHA256` (`59f07609...`) and asserts `RSTAR_ALL_BITS_Q12 = 251_672_950_854` and `RSTAR_NOVEL_BITS_Q12 = 223_758_978_608`. `65536.features` is length-verified (`112 + 140 * 21_349_576`), header-bound to `C_CORPUS_SHA256` + `COMPACT_V3_SHA256` + `DEV_TEST_UORT_SHA256`, and streamed with full-file `DEV_TEST_TAPE_SHA256` (`f0e3432d...`) verification (`:1518-1535, 1941-1945`). `P1(A1s)` reproduction asserts `246_099_676_775` all / `215_795_693_587` novel Q12 (`:58-59, 1946-1953`), matching `antigravity-5`.
- **Record-aligned chunking (`CHUNK_TOKENS = 2_097_152`):** `build_record_chunks` (`devtest_eval.rs:650-683`) requires `ids[0] == 0` (`BOS`) and emits `(tok_s, last_bos, ord_s, last_ord_at_bos)` only at `t == 0` (`BOS`) boundaries. Thus every chunk begins at `BOS = 0` and ends at the final token of complete records; `compute_causal_cache_features` (`cache_eval.rs`) and `stream_golden_and_skip_keys` (`s3_decode.rs`) reset all record state on every `id == 0`, so chunked evaluation is bit-identical to unchunked evaluation (tested by `record_chunked_cache_matches_unchunked_bit_for_bit` at `devtest_eval.rs:2056-2088`). `SplitAcc::accumulate_chunk` (`:601-647`) opens a new `rec_all`/`rec_novel` accumulator on each `id == 0`, consumes one target mass per `id != 0` with `chunk_novel[at] == 1` matching `antigravity-5`'s `CLASS_NOVEL = 1` (`antigravity-5/eval/src/main.rs:28`), and asserts `ord == masses.len()`.
- **Causal Pick parity (`devtest_eval.rs:1441-1493`):** Under `--features cuda`, `PickDevice::rows(&dev.ids, 192, ...)` runs over `windows(&dev.ids, 192)`, keeps the same seam/prefix/periodic probe set (`kept`) as `antigravity-5/eval/src/score.rs:208-225` (checked `pick_probes_checked >= 13_000` at `:1955`), and verifies `native.block_words == observed.blocks` against `trunk_model.observe_identity_pick` (`:1477-1480`) and `pr.hit == exp_hit` (`:1481-1486`; both `pick_read` in `discrete_d11.rs:1523` and `code_pick` in `code_pick.rs:17` iterate `1..t`, so `p == 0` (`BOS`) can never be picked).
- **Bootstrap CI, controlled delta, & R\* thresholds (`devtest_eval.rs:216-238, 1966-1995`):**
  - `paired_record_ci` (`1,000` resamples over all `5,321` record bit-differences, deterministic `splitmix64` seed `4390`, taking sorted indices `[24, 974]`) computes 95% paired record-block bootstrap CIs for `p4_c_skip_errwrite - p4_c_skip_static`, controlled `p4_full_errwrite - p4_cache_static`, and each arm vs `P1(A1s)` (`all` & `novel`) and vs `R*` (`all`).
  - `gate_brier_1pct_fewer_bits` requires `acc.all.bits_q12 * 100 <= P1_A1S_DEVTEST_ALL_Q12 * 99 && ci_vs_p1_all[1] < 0` (`:1977-1978`), and `SHEET.md:50` additionally requires `p4_full_errwrite` to beat `p4_cache_static` (`ci < 0`) so a cache-only gain cannot pass for error-write (`L-010`).
  - Integer `0.95 / 0.90 / 0.80 x R*` comparisons check `micro_all * 100 <= RSTAR_DEVTEST_BPB_MICRO * {95,90,80}` (`RSTAR_DEVTEST_BPB_MICRO = 1_418_136`, thresholds `1_347_229 / 1_276_322 / 1_134_508` matching `SHEET.md:12` & `devtest_eval.rs:2014-2017`).

### C. Charter §3 / D85 Compliance, CUDA Kernels & `dispatch.sh`
- **No banned ops/types:** `#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]` (`main.rs:1`); `#![forbid(unsafe_code)]` (`devtest_eval.rs:1`, `reply_verify.rs:1`, `cache_eval.rs:1`). Grep across `p5b-eval/src/` and `build.rs` confirms zero `f16`/`f32`/`f64`/`float`/`double`/`softmax`/`softplus`/`sigmoid`/`tanh`/`gelu`/`gemm`/`matmul`/`nvrtc`/`exp`/`log` math calls (only `leading_zeros` integer bit-scan at `devtest_eval.rs:306, 1584`).
- **CUDA kernels & AOT build (`p5b_kernels.cu`, `cuda.rs`, `build.rs`):** Zero floats, zero matmul, zero runtime division operators (only compile-time constants `4096ULL`, `6561u`, `9u` at `:16, 291-298`, and 128-by-64 shift-subtract `mul_div_u64` at `:42-59`). `build.rs` compiles AOT with `nvcc`; `dispatch.sh` sets `CUDA_DISABLE_PTX_JIT=1` (`:121`).
- **Gate 1 (`dispatch.sh:194-218`):** Compiles PTX (`nvcc -ptx -arch=sm_89`) and dumps SASS (`cuobjdump --dump-sass`), checks all 5 kernels (`p3_brier_step_batch`, `p3_em_step_batch`, `p3_s3_golden_lookup_and_accum`, `p3_s2_lookup_and_accum`, `p3_s4_vote_batch`) non-vacuously exist in both PTX and SASS, and asserts `ptx_float_opcode_hits == 0` and `float_opcode_hits == 0`.
- **Gate 2 (`main.rs:36-301`, `dispatch.sh:220-231`):** `finite` runs `525` GPU vs CPU probes (`256` Brier + `256` EM + `5` S3 + `4` S2 + `4` S4 including `role_class = 3` at `main.rs:155, 175` with 3 candidates `[(1,0),(2,0),(3,0)]` at `:191`), asserting `cubin_sha256` match, `rows_checked == 525`, and `divergent == 0`.
- **Gate 3 (`devtest_eval.rs:1719-1768, 1880-1931, 1954-1964`, `dispatch.sh:232-253`):** Verifies `shifted_batch_parity` (`CUDA_CHUNK = 65_536` vs prime `65_521`) on `MACRO_BATCH = 524_288` for both `880` (`E18`, EM + Brier) and `2640` (`E22`, EM) buckets, plus `spot_checked >= 25_000` (`ord < 256 || ord % 839 == 197 || ord + 16 >= 21_349_576`) comparing GPU S3, S2 (`0..3`), and S6 mixture outputs against CPU `GoldenShellStateTable`, `S2SkipTable`, and `brier_mix_q48` (`0` divergences).
- **Gate 4 (`reply_verify.rs:491-689`, `dispatch.sh:255-265`):** Cold-loads `c_ppmi_s4.uors4p01` verifying `FROZEN_S4_SHA256 = cd0e4385...`, re-evaluates `m2-fresh256` (`256` rows / `128` pairs) and `r4_cases` (`240` rows / `120` pairs) across `active`, `no_memory`, and 3-candidate `shuffled_role` (`sattolo_permute_cands`), checks `s4_gpu_vs_cpu_divergences == 0`, and emits `paired_band_compare` deltas for Stage B5a bars (`R1` / `R2` / `R4`).
- **L-002 & Sealed Receipt Discipline (`dispatch.sh:100-103, 267-286`):** Refuses overwrite if `$RUN_DIR` exists, locks `$WICK_LOCK` via `flock -n`, checks `FIT.md` contains `"TEST FITNESS: FIT"` and all 13 SHA-256 digests (`:183-191`), verifies frozen inputs (`:149-181`), and seals outputs via `sha256sum --strict -c receipts.sha256` (`:267-286`).

---

## 3. Blocking vs Non-Blocking Observations

**Blocking findings:** None.

**Non-blocking observations (safe as-is; logged for completeness):**
1. **`dispatch.sh` omits explicit `verify_sha` for `B1A_R_TAPE` & `B1A_R_PICK` (`dispatch.sh:69-71, 166-181`):** Both paths are required in `GO.json`, validated internally by `devtest_eval.rs:249-278` against `C_CORPUS_SHA256`, `COMPACT_V3_SHA256`, `R_UORT_SHA256`, and `TRUNK_SHA256`, and transitively pinned by the 4 exact Stage B4 `UORS6B01` SHA-256 assertions (`devtest_eval.rs:1025, 1288`).
2. **`dispatch.sh` `mkdir -p "$RUN_DIR"` precedes pre-Gate checks (`dispatch.sh:132` vs `:134-191`):** Runs after the `[[ -e "$RUN_DIR" ]]` (`L-002`) guard (`:100`), so it never overwrites a prior run; if a pre-check (`GPU_CAP`, `GPU_UTIL > 15`, `verify_sha`, `FIT.md`) aborts, a fresh `$RUN_DIR` path is needed on retry.
3. **`dispatch.sh` `sm_89` targeting & Gate 1 regex (`dispatch.sh:134-141, 195, 201-207`):** Hardcodes `GPU_CAP == "8.9"` and `-arch=sm_89` (and does not export `CUDA_VISIBLE_DEVICES`, matching single-GPU `sm_89` Wick execution; would cleanly refuse `exit 1` on another architecture). `PTX_HITS`/`SASS_HITS` omit `.ftz/.rn/.rz/.rm/.rp/.approx` and standalone `RCP/RSQ/SQRT/...` (covered by `.f*/.bf16` in PTX and `MUFU` in `sm_89` SASS on byte-identical `p5b_kernels.cu`).
4. **`reply_verify.rs` standalone `a1s_seal.json` SHA pinning (`reply_verify.rs:513-519`):** `reply_verify.rs` checks `FrozenPickMixture::from_bytes` cryptographic binding against `seal["manifest_sha256"]`, while `dispatch.sh:171-172` explicitly verifies the SHA-256 of both `a1s_weights` (`e2fbf41d...`) and `a1s_seal` (`b6e9ac04...`).
5. **`build_record_chunks` single-record `>= max_chunk_toks` edge case (`devtest_eval.rs:664-678`):** If a single record exceeded `2,097,152` tokens (`last_bos == tok_s`), one empty `(tok_s, tok_s, ord_s, ord_s)` entry would precede the full record's chunk; harmless (and inapplicable to `DEV-test`, whose `5,321` records total `21,354,897` tokens).
6. **`devtest_eval.rs:1476-1488` `Pick` parity check:** Verifies `native.block_words == observed.blocks` (`observe_identity_pick`, covering `[pick, distance, ties, gate+1, root]`) and `pr.hit == exp_hit` across `>= 13,000` probes, without additionally cross-checking `pr.pick`/`pr.distance` against `observed.blocks[0][0..2]` (`PickDevice::rows` derives both from the same device output).

---

**TEST FITNESS: FIT**
