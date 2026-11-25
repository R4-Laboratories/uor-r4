# Test Fitness Review: `written-state-count-model`, `antigravity-7` (Stage B2b, Round 3)

Reviewer: Non-Author Test Fitness Reviewer (Round 3, `research` subagent `3d77f1af-24f2-4171-9204-49a7277a96a1`, `Model: "pro"`).

Completed Round 3 non-author Test Fitness review of Stage B2b (`written-state-count-model`, `antigravity-7`) in `/Users/casey.allard/uor-r4-b2b-p2-wick`.

### Verification Findings

1. **`p2-s3/src/s3_kernels.cu` (`mul_div_q48`, lines 399–418):**
   - Verified exactness of the 48-step pure-integer binary restoring division against `s3_state.rs::lookup_q48`'s `((u128::from(count) * S3_Q48_ONE) / u128::from(total)).max(1).min(S3_Q48_ONE)` across all boundary cases:
     - `count == 0ULL`: returns `1ULL` immediately (matches `.max(1)` clamp, even if `total == 0`).
     - `count >= total` (`count == total` and `count > total`): returns `S3_Q48_ONE` (`1ULL << 48`), matching `.min(S3_Q48_ONE)`.
     - `1 <= count < total <= u64::MAX`: inductive loop invariant `count * 2^k = q_k * total + rem_k` with `0 <= rem_k < total < 2^64` holds at every step `0..48`. `carry = rem >> 63` captures bit 64 of `2 * rem`, so `(carry != 0ULL || doubled >= total)` is exact for `2 * rem >= total`. When true, `0 <= 2 * rem - total < total <= 2^64 - 1`, so unsigned `U64` subtraction `rem = doubled - total` computes the exact non-negative remainder modulo `2^64` without overflow/UB even for `total = u64::MAX`, `count = u64::MAX - 1` (producing `q = 2^48 - 1`, `rem_48 = 2^64 - 2^48 - 1`) or `count = 1, total = u64::MAX` (`q = 0` clamped at line 417 to `1ULL`).
   - Zero `unsigned __int128`, zero runtime division/modulo inside `mul_div_q48`, and zero floating-point types/intrinsics anywhere in `s3_kernels.cu`. Gate 1 (`dispatch.sh:118-121`) disassembles the sealed `$CUBIN` via `cuobjdump --dump-sass` and fails closed on `FLOAT_HITS != 0` before any GPU step runs.

2. **`p2-s3/src/main.rs` (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`, `uor_r4_discrete::Identity`):**
   - Line 1 matches `antigravity-4/cuda-v3/src/main.rs:2` (`#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`), allowing `cuda.rs`'s two `// SAFETY:`-annotated `launch.launch(cfg)` calls under `--features cuda` while enforcing `#![forbid(unsafe_code)]` on `main.rs`, `build.rs`, `builder.rs`, `eval.rs`, and `transport.rs`, and keeping non-CUDA `verify-tape`/`eval` builds strictly `unsafe`-free.
   - Lines 774–779 construct `uor_r4_discrete::Identity { token, occurrence, record, version }`, matching `crates/uor-r4-discrete/src/lib.rs:314-319` (`pub struct Identity`), with `DEV-tune` `PickDevice` probes verifying `>= 8,000` native `observe_identity_pick` matches.

3. **`p2-s3/build.rs` and `p2-s3/src/cuda.rs` (parameterized `UOR_NVCC_ARCH` & capability check):**
   - `build.rs` emits `cargo:rerun-if-env-changed=UOR_NVCC_ARCH`, validates `UOR_NVCC_ARCH` has form `compute_<digits>` with `digits.len() >= 2` and ASCII digits (`compute_89` -> `("8", "9")` / `s3_kernels.sm_89.cubin`; `compute_120` -> `("12", "0")` / `s3_kernels.sm_120.cubin`), compiles AOT with `nvcc -cubin -arch=sm_<digits> -O3`, and exports `P2_S3_CUBIN`, `P2_S3_CAP_MAJOR`, and `P2_S3_CAP_MINOR` (matching `crates/uor-r4-discrete/build.rs:426-433`, which itself requires explicit `UOR_NVCC_ARCH` for `PickDevice`).
   - `cuda.rs` embeds `CUBIN` via `include_bytes!(env!("P2_S3_CUBIN"))`, parses `P2_S3_CAP_MAJOR`/`P2_S3_CAP_MINOR` at runtime, and fails closed if `context.compute_capability()? != (cap_major, cap_minor)`.

4. **`SHEET.md`, `dispatch.sh`, `PRE-RUN.md` (Wick RTX 4090 or single-GPU Vast.ai RTX 5090/4090 pod):**
   - `SHEET.md:4` and `PRE-RUN.md:5` authorize either Wick RTX 4090 (`sm_89`, `$0.00`, `wick-codex-gpu.lock`) or one dedicated single-GPU Vast.ai RTX 5090 (`sm_120`) / RTX 4090 (`sm_89`) pod via `uor-pod up --count 1` (`<= $1/h` inside `$8/h` cap, torn down immediately after archiving `evidence/` + `receipts.sha256`).
   - `dispatch.sh` enforces `GO.json` pinning of `expected_host` (`hostname -s`), `expected_compute_cap` (`nvidia-smi -i "$gpu" --query-gpu=compute_cap`), `device_uuid`, `gpu_lock` (`flock -n 8` when `!= "none"`), `run_root`, `p2_s3_bin`, `cubin_path`, all source/binary/input SHAs, `DEV-test`/`rstar-devtest` refusal (`exit 99`), `L-002` `test ! -e "$ROOT"`, `CUDA_VISIBLE_DEVICES="$gpu"`, `CUDA_DISABLE_PTX_JIT=1`, and `admit_gpu` (`compute-apps` empty, `memory.free >= 8192 MiB`, `MemAvailable >= 10 GiB`) via `run_gpu_guarded` before every live GPU step (Gate 2, each Gate 3 arm under `timeout 180` / `elapsed <= 180`, Gate 4 `devtune-pick`, and each Gate 4 `capture`), followed by unlocked `cmp`, CPU `verify-tape` (`>= 12,500` probes, `0` divergent rows, `cubin_sha256` & `tape_sha256` parity), and Gate 5 CPU `eval`.

TEST FITNESS: FIT

---

# Test Fitness Review: Round 4 (Gate 1 PTX + SASS Audit Update)

Reviewer: Non-Author Test Fitness Reviewer (Round 4, `research` subagent `90d11f19-9a26-4cb8-aefc-74236d871fdb`, `Model: "pro"`).

Completed Round 4 non-author Test Fitness re-verification of the Gate 1 update in `SHEET.md` and `dispatch.sh`:
- **Scope integrity:** Zero changes to `ORDER.md`, `PRE-RUN.md`, or `p2-s3/` since Round 3 `FIT`; `bash -n dispatch.sh` exited `0`.
- **PTX 0-float static audit (`SHEET.md:32`, `dispatch.sh:34,121-122,134`):** Compiles `p2-s3/src/s3_kernels.cu` to PTX via `nvcc -ptx -arch="$NVCC_ARCH" -O3 -x cu` and asserts `PTX_FLOAT_HITS == 0`.
- **SASS 0-float / 0-MMA audit (`SHEET.md:32`, `dispatch.sh:123-135`):** Strict zero-exclusion SASS check when `cap != "12.0"` (`sm_89`); when `cap == "12.0"`, excludes ONLY exact `HFMA2 R[0-9]+, -RZ, RZ,` constant-load disassembly aliases (verified across valid, dirty, and empty synthetic SASS/PTX cases), records `sm120_const_mov_aliases` in `sass-check.txt`, and asserts `FLOAT_HITS == 0`.
- **`PATH` & `receipts.sha256` exclusions:** `export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH` set; `sass.asm` and `s3_kernels.ptx` excluded from `receipts.sha256`.

TEST FITNESS: FIT

---

# Test Fitness Review: Round 5 (`s3_kernels.cu` `mode == 1u/2u` Fix)

Reviewer: Non-Author Test Fitness Reviewer (Round 5, `research` subagent `37a19c61-6bd0-4477-937e-5c1e370edcb2`, `Model: "pro"`).

Verified cryptographically that ONLY the 4 tokens in `s3_kernels.cu` (`L201`, `L226`, `L239`, `L244`) changed (`8ee276600cc82a9525f87e34f7d17d6b9b7da0a834651f5617a7dc8a72d92d56`), aligning `s3_transport_docs` with `S3AddressArm::encode_parts` (`0=E8Product`, `1=TokenContext`, `2=RandomTransport`, `3=SplitMixPrefix`) and `s3_lookup_batch` across all 11 arms (`e8`, `rand`, `tc_{1,2,3,4,6,8,10,12}`, `splitmix`) bit-for-bit with zero floats, matmul, or softmax.

TEST FITNESS: FIT
