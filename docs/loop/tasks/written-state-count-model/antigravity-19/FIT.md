# Non-Author Test Fitness Review (`FIT R3`) — `antigravity-19` (`written-state-count-model`, Phase `P5` PR 2 of 2)

- **Reviewer:** Fresh-context `pro` non-author `research` subagent (`8b37573f-c89e-4c60-b6f9-6782790ff61f`)
- **Reviewed Worktree / Branch:** `/Users/casey.allard/uor-r4/.worktrees/antigravity-p5-qual` (`antigravity/p5-qual`)
- **Exact Reviewed Commit:** `62f216a96b5cdb2a9b8fa74020c260974b021517`
- **Prior FIT History:** FIT R1 (`feae0449`, `NOT FIT`, 3 findings fixed at `8ace40cc8`) -> FIT R2 (`997efcab`, `FIT` at `8ace40cc8`, Wick host lost power mid-Epoch 0 `DEV-tune` in retired `20261126-antigravity-p5-qual`) -> FIT R3 (`8b37573f`, `FIT` at `62f216a96b5cdb2a9b8fa74020c260974b021517` targeting fresh `L-002` `OUT_DIR=20261126-antigravity-p5-qual-r2`).
- **Pre-FIT Wick Build & Synthetic Audit (`62f216a96`):** `p5_kernels.sm_89.cubin` `sha256=96ce1a34257d2f7519d13af0fc2b4c2829cde77cbfe71cedb24ce541f65996bc`, `0` clippy warnings (`--features cuda`), `0` PTX/SASS float or MMA ops, `5/5` kernels present, Gate 2 `p5-train finite` `0` diffs across all checks in `0.90s`.

---

## Verbatim Reviewer Report (`8b37573f`)

# FIT R3 Report — Phase P5 PR 2 of 2 (`written-state-count-model`, `#4063`, `antigravity-19`)

- **Target worktree:** `/Users/casey.allard/uor-r4/.worktrees/antigravity-p5-qual`
- **Exact reviewed HEAD:** `62f216a96b5cdb2a9b8fa74020c260974b021517` (clean working tree; scope strictly confined to `docs/loop/tasks/written-state-count-model/antigravity-19/**` relative to base `9631cc2fd`)
- **Prior FIT R2:** `5d6d75b14` / `8ace40cc8` (`docs/loop/tasks/written-state-count-model/antigravity-19/FIT.md`)

## 1. Charter §3 & Build/Static Audits
- **Pure integer / zero banned ops:** Static scan across all `p5-train/src/{cuda.rs,driver_check.rs,finite.rs,main.rs,train_eval.rs,p5_kernels.cu}` found `0` float types/literals/ops, `0` softmax/squashing ops, `0` dense matmul/MMA ops.
- **Zero hardware variable division in CUDA (`kn_score_division.cuh`):** All 15 `/` or `%` tokens in `p5_kernels.cu` are compile-time constants (`P4_Q48_ONE / P4_VOCAB` at L27, and `6561u` / `9u` Starling-7 radix unpack at L625–632 and L961–968). All runtime divisions (`KN` gamma, `KN` term, `ML` mass, `S3`/`S2`/cache mass modulation) use `kn_score_division::divide`.
- **Repo ratchets & shell check:**
  - `python3 scripts/check_no_softmax.py`: `OK` (`3828 <= 4451`).
  - `python3 scripts/check_no_matmul.py`: `OK` (`93 <= 93`).
  - `bash -n docs/loop/tasks/written-state-count-model/antigravity-19/dispatch.sh`: exit `0`.
- **AOT & safety hygiene:** `#![forbid(unsafe_code)]` on `build.rs`, `driver_check.rs`, `finite.rs`, `train_eval.rs` (`cuda.rs` isolates `unsafe` to `#[cfg(feature = "cuda")] mod gpu`, and `main.rs` enables `#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]`). `build.rs` compiles AOT `sm_89` cubin (`nvcc -cubin -arch=sm_89 -O3`), `dispatch.sh:61-62` sets `UOR_NVCC_ARCH=compute_89` and `CUDA_DISABLE_PTX_JIT=1`, and `dispatch.sh:190-205` pins `cubin_sha256` across Gates 1–4.
- **Scope:** `git diff --name-only 9631cc2fd..HEAD` touches only `antigravity-19/` (`0` edits to `crates/`, root `Cargo.toml`, or root `Cargo.lock`).

## 2. Bit-Exact Equivalence vs `P4Session::step` & `p5_brier_core` <!-- claim-wording: allow -->
1. **Stateless 256-thread parallel `CausalSessionCache` (`p5_kernels.cu:515-558, 689-700, 1064-1288` vs `p4_serving.rs:1288-1600`):**
   - `rec_start` uses `atomicMax` over `i + 1` where `sh_hist[i] == 0u`, giving `rlen = len - rec_start`, `rec_tokens = sh_hist + rec_start`: handles `rlen == 0` (`len == 1`, `[0]`), `rlen == 1` (`len == 2`), `BOS`-less slid windows (`pos > 2048` -> `rec_start = 0`, `rlen = 2048`), and mid-window `BOS` identically to `P4Session::sync_prompt` + `CausalSessionCache::reset_from_tokens`.
   - Unigram halving (`p5_kernels.cu:1075-1089`) loops `base = 0; base < rlen; base += 64` and halves `sh_uni[0..4096] >>= 1` iff `base > 0`, matching `pos > 0 && (pos & 63 == 0)` at `rlen = 64` (0 halves) vs `rlen = 65` (1 half) and all multiples up to `2048` (`sh_uni` bounded `< 8192`, no `u32` overflow).
   - Bigram (`i in 0..rlen-1`), Trigram (`i in 1..rlen-1`, `TRI_TABLE_CAP = 16384` never saturates at `rlen <= 2048`), and LCP (`i in 2..rlen-1`, `pfp = i + 1 <= 2047`, `lcp_mask[64]` indexed at `pfp >> 5 <= 63`, scanned descending to pick exact 8 most-recent matches, `max_l = min(16, pfp, rlen)`, `ml >= 3`) are race-free and bit-identical to `CausalSessionCache::eval_sparse_all`. Note `rlen == 3` cannot set LCP flag (`i >= 2` and `i + 1 < 3` is empty).
2. **Host-precomputed Order-0 `UORKNC03` (`cuda.rs:320-435` vs `compact.rs` & `interpolation.rs`):**
   - Handles both inline (`deg0_raw < 255 -> deg0_raw + 1`) and escaped (`deg0_raw == 255 -> find_esc_deg_host(..., 0)`) Order-0 degrees, `shared` and `!shared` `esc16`/`esc32` decoding, exact `interpolate_order(P4_KN_BASE, ...)` without premature clamping (`st->kn_mass[v]` clamped to `>= 1` only after all orders `k = 1..active_orders-1` at `p5_kernels.cu:897`), and `d_o0_ml_mass`.
   - Pass 1 & Pass 2 over `k >= 1` (`p5_kernels.cu:703-891`) compute `esc_rank` from `esc_sb[blk]` and scan disjoint `[fs, fe)` blocks across threads (strictly ascending follower tokens guarantee zero write races on `st->kn_mass[tok]` and `st->acc_{lo,hi}[s][tok]`).
3. **`S3` / `S2-skip` / `UORS6B01` & `u96` accumulators (`p5_kernels.cu:358-445, 609-684, 801-1060`):**
   - `golden_shell_pkey_dev`, `S3` primary/backoff/Starling-7 (`st->kn_mass` scratch reused safely after L906 `__syncthreads()`), `S2-skip` 3-family `lo24` lookup, `highest_order`/`plain_b`/`b2640`, and `u96` accumulators (`sum <= 2^80 < 2^96`) match `p4_serving.rs` bit-for-bit.
   - Shared memory per block is `~29.1 KB` (Kernel 1) / `~29.5 KB` (Kernel 5), well inside the `48 KB` static limit; `P5WorkerScratch` (`229,376 B`) matches `P5_SLOT_SCRATCH_BYTES` with static asserts on all ABI structs (`RowStepCmd`, `EvalTileCmd`, `EvalArmRecordOut`, `EvalSpotOut`, `OrderV3`).
4. **`RowBatchScheduler`, `build_eval_tiles`, & `active_pc_mask` (`train_eval.rs:103-151`, `cuda.rs:540-577, 1004-1133`, `p5_kernels.cu:1684-1914`):**
   - `RowBatchScheduler` visits every non-BOS target across `R` in canonical order in uniform `<= 4,096`-row batches (`3,893 * 4,096 + 1,856 = 15,947,584`, `3,894` batches/epoch).
   - `build_eval_tiles` splits records (`rl >= 2`) into `<= 256`-target tiles, precomputes `first_spot_pos` on host (zero CUDA variable `%`/`/`), and preserves exact `rec_local` indexing; Kernel 5 evaluates `code_pick` across all `<= 5` arms first to compact distinct `pc in 1..=4` (or `pc = 0` when `len < 3`) via `sh_active_mask` / `pc_to_slot`, guards `out_spot_count < max_spots`, and reduces tiles deterministically on host.

## 3. Experimental Validity, Gate Rules & L-002 Compliance
- **Phase P5 gate (`DECISION.md:335-346` & `CHARTER.md §4.5`):** `P5_S1_COORDS = 16,384 <= 100,000`, `K = 64` at `4,096` rows, exact `15 * K` equal-trial re-seeded controls (`ctrl_reseed_canonical` `4390` & `ctrl_reseed_random` `4391`/`4392` asserted equal in `train_eval.rs:895` and `dispatch.sh:204`), `DEV-tune`-only stopping (`0` `DEV-test` peeks), non-extrapolating `evaluate_p5_gate` (`>= 1%` actual `DEV-test` Brier drop for `elapsed <= 86,400s`, 1,000-draw paired record bootstrap 95% CIs `< 0`, BPB non-increasing), `>= 512` (`640`) live CPU `P4Session` spot checks + `64`-record determinism repeat (`0` diffs required), and lossless `rebind_and_pack_p5_deployment` only if `P5_QUALIFIED`.
- **Gate 2 `finite.rs` coverage:** Exercises Kernels 1–5 including `rlen = 0, 1, 64, 65, 128, 129`, `pos = 2049..=2055` strict `> 2,048` slide across Kernels 1–4 and exhaustive 5-arm Kernel 5 + `>= 10` strict `> 2,048` spots, all 5 `BVG-K` branches, equal-trial random control, planted target shift, and live `4,096`-row `Driver` CUDA check.
- **L-002 & Pinned Inputs:** All 10 Wick input SHA-256s and byte/record/target counts match across `SHEET.md:28-39`, `dispatch.sh:14-20`, and `train_eval.rs:33-65`. `dispatch.sh:10,27-30` enforces fresh `OUT_DIR=/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5-qual-r2` with fail-closed `[[ -e "$OUT_DIR" ]]`, leaving aborted `20261126-antigravity-p5-qual` untouched.
- **Non-blocking doc note:** `ORDER.md:11` (from pre-optimization commit `050efbb8b`) states `11,679` batches (`11,211,840` trials) whereas `SHEET.md:63` accurately reflects `3,894` batches/epoch (`11,682` batches = `11,214,720` trials for 3 epochs) and `FIT.md` still holds R2 (`8ace40cc8`) until the main session records R3 (`62f216a9`). Neither affects runtime execution or gate validity.

```
TEST FITNESS: FIT
```
