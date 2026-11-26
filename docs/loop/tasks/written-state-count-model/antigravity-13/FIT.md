# Non-Author Test Fitness Review: written-state-count-model, antigravity-13 (Stage B5)

- **Reviewer:** Non-Author Test Fitness Reviewer (R4, fresh-context subagent under `CHARTER.md` §4.5 & §5.10)
- **Target Package:** `/Users/casey.allard/uor-r4/.worktrees/antigravity-b5-s4/docs/loop/tasks/written-state-count-model/antigravity-13/`
- **Milestone / Issue:** `written-state-count-model` (`#4063`, `M2+/P3`, Stage B5 under `DECISION.md` §4)
- **Authority Checked:** `CHARTER.md` §3.1–§3.13, §4.1–§4.5, §5.1–§5.10; `AGENTS.md`; `DECISION.md` §4; `crates/uor-r4-discrete/src/kn_ceiling/runtime/p3_state.rs`; `crates/uor-r4-tokenizer/src/lib.rs`

## Audit Summary (R4 Verification over R3)

An independent, read-only R4 audit verified all 11 package files directly on disk (`ORDER.md`, `SHEET.md`, `dispatch.sh`, `p5-eval/Cargo.toml`, `p5-eval/Cargo.lock`, `p5-eval/build.rs`, `p5-eval/src/{main,cuda,c_ppmi,reply_eval}.rs`, `p5-eval/src/p5_s4_kernels.cu`). Only `dispatch.sh` changed since R3 (new SHA-256 `607b95b9c7c18d65cdf600b9731881f6dd234521e2245558a4949b9af1a89c16`, superseding R3 `5a9ece9cf1bc5e75ed0d1ee281fbdbcfdf0fba32356552e9e1547c7ffa8d05c5`), while all other 10 package files match their R3 SHA-256 digests verbatim. In `dispatch.sh:187-200`, Gate 1 now dumps SASS directly to `$RUN_DIR/p5_s4_kernels.sass`, checks `>= 1` match for `.entry p3_s4_vote_batch` in PTX and `p3_s4_vote_batch` in SASS via `test "$(grep -c ...)" -ge 1` without `grep -q` SIGPIPE under `set -euo pipefail`, and preserves exact PTX/SASS zero-float/zero-MMA auditing (`PTX_HITS` / `SASS_HITS` and `sass-check.txt`). R2 blocking finding **B3-R2**, Gate 3 `timeout 2400` (`dispatch.sh:226`), `DEV-test` `exit 99` lock (`dispatch.sh:15-19`), L-002 guard (`dispatch.sh:91-94`), and `GO.json` bindings (`dispatch.sh:21-88,145-184`) remain intact.

## Audit Findings

1. **Resolution of Blocking Finding B3-R2 (`p5-eval/src/reply_eval.rs:1502-1541`, unchanged from R3):**
   - `contrastive_stem_pair_disambiguation_suppresses_ambiguous_stems` constructs a valid minimal `ByteBpeTokenizer` JSON (`version`, `pre_tokenizer` `{"type":"ByteLevel","add_prefix_space":false}`, `model.type = "BPE"`, `model.vocab = {"a":0,"b":1,"c":2,"d":3}`, `model.merges = []`) and exercises `emit_contrastive_s4_atom_votes` unconditionally without skipping assertions.
   - Exact tie (`ambig_tie = [220, 420]`, `4 vs 4`) and sub-threshold weak imbalance (`ambig_weak = [221, 421]`, `9 vs 8`, `36 < 40`) are suppressed; multi-token prefix-scaffold non-leakage (`210 x 20`) is enforced; 3-token internal pair emission (`j1 = [210, 310, 410]` -> `(310, 410)` with weight `8`) and overpowered `(210, 410)` (`12` for role `1`, absent for role `2`) / `h1 = [210, 411]` (`8` for role `2`) are asserted directly.

2. **`dispatch.sh` Verification (R4 Gate 1 SIGPIPE Fix, `timeout 2400`, Gates 1–3, L-002 Guard, `DEV-test` Lock, `GO.json` Bindings):**
   - **R4 Gate 1 SIGPIPE fix (`dispatch.sh:186-210`, specifically `lines 187-200`):**
     - `dispatch.sh:187` compiles `$P5_KERNELS_CU` to `$RUN_DIR/p5_s4_kernels.ptx` (`nvcc -ptx -arch=sm_89 -O3 -x cu`).
     - `dispatch.sh:188` dumps SASS directly to `$RUN_DIR/p5_s4_kernels.sass` (`cuobjdump --dump-sass "$CUBIN_PATH" > "$RUN_DIR/p5_s4_kernels.sass"`) without piping into `grep -q`, eliminating SIGPIPE (`141`) under `set -euo pipefail`.
     - `dispatch.sh:189-190` verify `test "$(grep -c '\.entry p3_s4_vote_batch' "$RUN_DIR/p5_s4_kernels.ptx")" -ge 1` and `test "$(grep -c 'p3_s4_vote_batch' "$RUN_DIR/p5_s4_kernels.sass")" -ge 1`, confirming `>= 1` entry match in both PTX and SASS with zero `grep -q` pipelines (and failing immediately under `set -e` if count is `0`).
     - `dispatch.sh:192-209` preserve exact PTX/SASS zero-float/zero-MMA auditing: `PTX_HITS` filters PTX header/comment directives (`grep -vE ... | grep -Eic '\.(f16|f32|f64|bf16)\b' || true`, where `-c` drains all upstream lines without SIGPIPE), `SASS_HITS` scans `$RUN_DIR/p5_s4_kernels.sass` directly with the exact float/MMA opcode list (`FADD|FADD32I|...|RRO`, `|| true`), writes `$RUN_DIR/sass-check.txt` (`lines 199-204`), and exits `1` if `PTX_HITS -ne 0 || SASS_HITS -ne 0`.
   - **Gate 3 `timeout 2400` (`dispatch.sh:226`):** `timeout 2400 /usr/bin/time -v -o "$RUN_DIR/reply.time" "$P5_EVAL_BIN" reply ...` accommodates two streaming passes over `c.tokens.u16` (`2,075,608,357` tokens) without altering mechanism logic, determinism, or any Gate 1–3 check.
   - **`DEV-test` `exit 99` hard lock (`dispatch.sh:15-19`):** Case-insensitive scan over `$GO_JSON` refuses any match to `dev-test|rstar-devtest|023784b7|ec048217` with exit code `99`, paired with `reply_eval.rs:270-277,1007-1013,1045` (`refuse_sealed_devtest_sha`).
   - **L-002 guard (`dispatch.sh:91-94`):** Refuses pre-existing `$RUN_DIR` (`exit 2`) before directory creation (`line 128`).
   - **`GO.json` bindings (`dispatch.sh:21-88,145-184`):** Validates all 48 string fields via `need_str` + `shlex.quote`, verifies GPU `compute_cap == 8.9`, `GPU_UTIL <= 15`, `GPU_FREE_MIB >= 4096`, Wick lock (`flock -n 9`, `exit 75`), executable `$P5_EVAL_BIN`, `$CUBIN_PATH`, all 11 package SHA-256s, all 11 frozen input SHA-256s (`c_uort`, `compact_v3`, `r_uort`, `a1s_weights`, `a1s_seal`, `trunk`, `tokenizer_json`, `fresh256/{plan,inputs,labels,manifest}.json`), and embedded `TEST FITNESS: FIT` + all 11 SHA-256 digests in `FIT.md`.
   - **Gates 1–3 (`dispatch.sh:186-281`):** Gate 1 enforces non-vacuous `p3_s4_vote_batch` presence and `ptx_float_opcode_hits == 0` / `float_opcode_hits == 0`; Gate 2 runs `p5-eval finite` (`timeout 60`) and asserts `cubin_sha256 == $CUBIN_SHA`, `s4_checked == 64`, `s4_divergent == 0`, and `divergent == 0`; Gate 3 verifies `artifacts/c_ppmi_s4.uors4p01`, `s4_gpu_vs_cpu_divergences == 0`, `c_tokens_scanned == 2075608357`, `mworld_spans_extracted == 24`, `total_hinge_updates > 0`, `r1_r2_m2_fresh256.{rows == 256, pairs == 128}`, and `r4_heldout_phrasing.{rows == 240, pairs == 120}`, then records and strictly checks `sha256sum --strict -c receipts.sha256`.

3. **CHARTER §3, D85 Native-Space Compliance, and Algorithm/Kernel Parity (10 source/config files unchanged from R3):**
   - **Zero banned ops (`CHARTER.md` §3.2–§3.4):** Static inspection and grep across `p5-eval/build.rs`, `p5-eval/src/*.rs`, and `p5-eval/src/p5_s4_kernels.cu` confirm zero `f16`/`f32`/`f64`/`bf16` types or ops, zero softmax/squashing, zero matmul/GEMM/MMA, and ahead-of-time `nvcc -cubin` in `build.rs` (zero NVRTC).
   - **Zero runtime CUDA divisions (`p5-eval/src/p5_s4_kernels.cu:14-41,101`):** `p3_s4_vote_batch` uses exact 128-by-64 restoring binary division `mul_div_u64` via `__umul64hi` and shifts; `cuda.rs:74-125` matches the `UORS4P01` layout, and the 17 kernel arguments (`cuda.rs:166-183`) match `p5_s4_kernels.cu:60-78` in count, order, and types.
   - **`unsafe` isolation:** `#![forbid(unsafe_code)]` is enforced on `build.rs`, `c_ppmi.rs`, and `reply_eval.rs`; `main.rs:1` enforces `#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`; the single `unsafe` block resides in `cuda.rs:185` inside `#[cfg(feature = "cuda")] mod gpu` for `launch.launch(cfg)`.
   - **Injective streaming, 2-hop PPMI, and `#[cfg(feature = "cuda")]` parity:** `c_ppmi.rs` (`lines 832-873`) and `reply_eval.rs` (`lines 908-935`, `verify_disjoint_sets`) remain byte-identical to R3.
   - **Local host note (non-blocking):** Local macOS lacks `nvcc` and an RTX 4090 (`sm_89`), so `--features cuda` compilation and runtime Gates 1–3 execute on the Wick/pod `sm_89` host under `dispatch.sh`.

## Read-Only Verification Executed on Disk

- `find . -type f | sort` in `docs/loop/tasks/written-state-count-model/antigravity-13`: **PASS** (11 package files + `FIT.md`, no stray files)
- `shasum -a 256` on all 11 package files: **PASS** (10/10 non-`dispatch.sh` files match R3 digests verbatim; `dispatch.sh` matches `607b95b9c7c18d65cdf600b9731881f6dd234521e2245558a4949b9af1a89c16`)
- `bash -n dispatch.sh` and `shellcheck dispatch.sh`: **PASS** (both exit `0`, zero warnings, `0` `grep -q` hits in `dispatch.sh`)
- In-memory `bash -c 'set -euo pipefail; ...'` probe of Gate 1 `grep -c` and `grep -vE | grep -Eic ... || true` over `>256KB` synthetic PTX/SASS inputs and missing-entry/float negative probes: **PASS** (`CLEAN_OK PTX_HITS=0 SASS_HITS=0`, negative probes rejected cleanly, zero SIGPIPE)
- `rustfmt --edition 2021 --check p5-eval/build.rs p5-eval/src/*.rs`: **PASS** (exit `0`)
- Banned-pattern & `unsafe` grep across `p5-eval/build.rs` and `p5-eval/src/`: **PASS** (`0` code hits for float/squash/matmul/NVRTC; `unsafe` confined to `cuda.rs:185` behind `#[cfg(feature = "cuda")]`)

## Verified File SHA-256 Table

| File | Verified SHA-256 |
|---|---|
| `ORDER.md` | `73fe95ef8457f24f538d8b3e29821207cc53786f3ad2f55feb78521da0411453` |
| `SHEET.md` | `ed9ca433048e0b9b8d6a421ec658dc4699c2eba789ba6d335ad105284df43ce4` |
| `dispatch.sh` | `607b95b9c7c18d65cdf600b9731881f6dd234521e2245558a4949b9af1a89c16` |
| `p5-eval/Cargo.toml` | `d21c4abc48e1d37bd1639c2c899a68b2cc15624e5765d3310a897fbc0879be69` |
| `p5-eval/Cargo.lock` | `4fa7b0456b87ce81bc798a81dc522851a6caf0d9611cfe5ef9ade38a15edc284` |
| `p5-eval/build.rs` | `174eeb12cda419867026ef0e8045e1bbb29b6d411538e9e55667c462a06a2260` |
| `p5-eval/src/main.rs` | `2ddb4767808376e8b307bc2b9939966c4c58bca167223f8d389399a9192d7224` |
| `p5-eval/src/cuda.rs` | `4e859803329c9957c422654807c4bb281589c26323de2c3c769abf6a3093f790` |
| `p5-eval/src/c_ppmi.rs` | `2cd47757fc40fc412eb0c7d11c7553e2beb49d76efe00ca5ced5575f1f280c90` |
| `p5-eval/src/reply_eval.rs` | `859715b2d724dcd31e5128ea6bed724f02741a08da7789c28f36580522919164` |
| `p5-eval/src/p5_s4_kernels.cu` | `90ad047b111d1b712e7af2c067ceb4b30f9ade98464639491bdf37aaa03bd232` |

TEST FITNESS: FIT
