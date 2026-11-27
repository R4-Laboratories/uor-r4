# Pre-Registration Sheet: Phase P5 Mixed-Mass Brier `S1` Training & Sealed `DEV-test` Gate (`antigravity-19`)

- **Lab:** `antigravity`
- **Item / Issue / Phase:** `written-state-count-model` (`#4063`), **Phase `P5` PR 2 of 2** (`DECISION.md:335-346`, `M2+/P5`)
- **Prerequisites:** Phase `P4` PR 2 (`antigravity-17`, PR `#4304`, merged at `e90a1429c`, `p4.d11` serving qualified; Bar A `chat-v0-p2` PRLL honestly failed at `1,467,631` micro-BPB vs `0.90` BPB) and Phase `P5` PR 1 (`antigravity-18`, PR `#4306`, merged at `9631cc2fd`, `DriverConfig` `1..=4,096` row cap lifted + `p5_brier_core` pure-integer core).
- **Scope & Constraints:** Pure-integer native-space execution (`D85`, `CHARTER.md` §3). Zero floats (`f32`/`f64`/PTX/SASS float ops), zero matrix multiply / MMA, zero squashing or normalized exponential, zero hardware variable division in CUDA (`kn_score_division.cuh`). Zero edits to `crates/` or root `Cargo.toml`/`Cargo.lock`. Zero `DEV-test` peeks during training or stopping decisions.

---

## 1. Instrument, Persistent State & Hardware (`D85`)

1. **Hardware:** Wick RTX 4090 (`sm_89`, `24 GB` VRAM, `$0.00` pod spend, `0` cloud pods).
2. **Persistent Native State Schema:**
   - **Trainable `S1`:** `P5S1State` (`16,384 <= 100,000` `Cl(0,6)` orbit root coordinates `0..16`, `V = 4,096`, `B = 4`, `epoch`, serialized losslessly as self-describing `UORP5S10` with trailing SHA-256).
   - **Frozen `S0`/`S2`/`S3`/`S4`/`S6`:** Embedded `UORP4A01` sections loaded verbatim from sealed `p4.d11` (`UORKNC03`, `UORS3C01`, `UORS2C01`, `UORS4P01`, `UORS6B01`, `CausalSessionCache` at `P4_MAX_CONTEXT = 2,048` — bit-identical to `uor-chat` serving).
3. **Shared Typed Operators & Exact Objective (`R1-M6`):**
   - Standalone offline crate `docs/loop/tasks/written-state-count-model/antigravity-19/p5-train/` compiling AOT `nvcc -cubin -arch=sm_89 -O3` (`CUDA_DISABLE_PTX_JIT=1`).
   - Objective is exact full-vocabulary (`1..4,096`) multi-class integer Q32 Brier of the `22`-expert mixed `P4` Q48 probability vector (`full_vocab_brier_q32`), decomposed losslessly per row over `(pick_class in 0..5, pick_pos in 1..L-1)` with zero approximation.
   - Contiguous deterministic causal row scheduler (`<= 4,096` causal rows/batch, `3,894` uniform batches/epoch across `R`'s `15,947,584` targets) evaluates non-Pick Q80 accumulators once per batch across `256` persistent worker blocks using stateless 256-thread-parallel `CausalSessionCache` in shared memory + host-precomputed Order-0 (`UORKNC03` `k = 0`) tables (`0` approximation, bit-identical to `P4Session::step`), then scores `K = 64` `S1` coordinates (`16` lanes/coord = `960` generator trials/batch) via `rows-K` (`p5_rowsk_score_kernel` + `p5_rowsk_reduce_kernel` checked `u64` sum with `status = 12` overflow guard) and commits via exact `BVG-K` (`|S_t| = 0`, `1`, or `>= 2` with GPU scratch joint verification `L(S_t) < L({best})`, `resulting_port` tie-break, `epoch += 1`).
   - Multi-arm evaluation (`p5_eval_records_multi_arm_kernel`) partitions every split into `256`-target tiles dispatched across `256` persistent worker blocks (eliminating long-record stragglers up to `55,479` tokens), evaluates `code_pick` across all `<= 5` arms first to compute non-Pick accumulators strictly for the distinct active `pick_class` bitmask used at each position, and deterministically reduces tile outputs per record on host.

---

## 2. Frozen Inputs & Data Split (Fail-Closed SHA-256 Pins)

Every input file is SHA-256 verified before computation; any mismatch halts with non-zero exit code:

| Role | Path on Wick | Exact Bytes | Exact SHA-256 |
|---|---|---:|---|
| Sealed `P4` artifact (`S0`–`S6`) | `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/artifacts/p4.d11` | `3,987,940,843` | `e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776` |
| Base `S1` trunk (`seed1.d11`) | `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` | `86,536,132` | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` |
| Bound tokenizer | `/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json` | `265,393` | `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377` |
| Training split `R` (`4,060` recs / `15,947,584` tgts) | `/home/callard/uor-codex/20261119-rival-admission-29/count/cr-v1/r.tokens.u16` | `31,903,352` | `5875b173ad19aefbdd38929af8d88c15344333e941f44128d5e0a67b50102a0a` |
| Checkpoint split `DEV-tune` (`2,670` recs / `10,807,417` tgts) | `/home/callard/uor-codex/20261118-diagnostics/inputs/c420/dev-tune.tokens.u16` | `21,620,238` | `75c6d1ac479bf120bac3aa60ac160536e8146f8052aa49c002209500799b7d0a` |
| Sealed `DEV-test` census | `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/inputs/c420/dev-test.json` | `2,118` | `ec048217ac64f05a6519d7e38dd2fabf6e244eca8009b24476b3870cbc0f73f2` |
| Sealed `DEV-test` tokens (`5,321` recs / `21,349,576` tgts) | `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/inputs/c420/dev-test.tokens.u16` | `42,709,858` | `023784b74c98a796401a064d50bf2f83f2f8eb95ab8bbead1cbe2aec6e7a176b` |
| Sealed `DEV-test` novel mask | `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/inputs/c420/dev-test.novel.u8` | `21,354,897` | `5bf3881bff8fb5abe5c15da2b6dfdfe60f4eaa709ee061d9d0405d618af10471` |
| Token UTF-8 byte lengths | `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/inputs/c420/id-bytes.u8` | `4,096` | `befb22f6908751ae522a80336afa584eb2c24e36a10b90ad3105e39271a48ac9` |
| Canonical `R*` masses | `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-p1b-devtest/rstar/run-v1/rstar-devtest.masses.u64` | `170,796,608` | `59f076093c51b15469b55b01821947b7c51d8038e68d834465e29be72d56f8de` |

---

## 3. Execution Steps (Gates 1–5)

### Gate 1 — AOT `sm_89` PTX/SASS Audit & `uor-r4-discrete` CUDA Self-Test
- Build `p5-train --release --offline --features cuda` (`UOR_NVCC_ARCH=compute_89`, `CUDA_DISABLE_PTX_JIT=1`).
- Run `cargo test --offline --locked -p uor-r4-discrete --features cuda --lib driver_config_admits_4096_rows_and_refuses_out_of_range` and `p5_brier_core` tests on Wick.
- Inspect `nvcc -ptx` and `cuobjdump --dump-sass` on the single compiled `p5_kernels.sm_89.cubin`: require **`0` float opcodes**, **`0` MMA opcodes**, and all `5` required kernel entry points (`p5_slot_step_brier_kernel`, `p5_batch_eval_roots_kernel`, `p5_rowsk_score_kernel`, `p5_rowsk_reduce_kernel`, `p5_eval_records_multi_arm_kernel`). Record `cubin_sha256`.

### Gate 2 — Synthetic Finite GPU-vs-CPU Exact Parity & Live `4,096`-Row `uor-r4-discrete` CUDA Check
- Run `p5-train finite`:
  1. Verify **`0` raw-word diffs** between `p5_kernels.cu` (`p5_slot_step_brier_kernel`, `p5_batch_eval_roots_kernel`, `p5_rowsk_score_kernel`, `p5_rowsk_reduce_kernel`, `bvgk_step_from_gpu_scores`, `p5_eval_records_multi_arm_kernel`) and CPU `P4Session::step` + `uor_r4_discrete::p5_brier_core` across `K = 1, 8, 16, 64`, `|S_t| = 0, 1, >= 2` (joint-accept & joint-reject, all 5 branches hard-asserted), `resulting_port` tie-break, planted target shift, equal-trial random-coordinate control parity, AND **strict `> 2,048`-token sliding-window parity** (`pos = 2,049..=2,055` across Kernels 1–4 `materialize_batch`/`batch_eval`/`rows-K`, plus exhaustive 5-arm replay of the full `2,056`-token record and `>= 10` strict `pos > 2,048` spot checks on Kernel 5 `eval_records_multi_arm`).
  2. Verify `DriverConfig::validate()` at `0` (refused), `4,096` (admitted), `4,097` (refused) and run live **`4,096`-row, `K = 64`** `uor_r4_discrete::device::Device::{carry_windows, score_rows_k, propose_block}` on Wick RTX 4090 confirming `carried.rows() == 4096`, `64/64` `status == 0` scores returned, and epoch `0` untouched (`UT` invariant).

### Gate 3 — `R` `rows-K` + `BVG-K` Training vs Equal-Trial Re-Seeded Controls (`DEV-tune` Only)
- Run `p5-train train --min-epochs 3 --max-epochs 6 --k 64`:
  - **5 Arms:**
    1. `baseline_seed1`: frozen geometric default `seed1.d11` (`0` updates).
    2. `s1_p5_from_seed1`: initialized from `seed1.d11`, trained with canonical wrapping `K = 64` `rows-K` + `BVG-K`.
    3. `ctrl_reseed_canonical`: re-seeded random `S1` orbit roots (`seed = 4390`), trained with identical canonical wrapping `K = 64` `rows-K` + `BVG-K` (**exact equal trials**).
    4. `ctrl_reseed_random`: re-seeded random `S1` orbit roots (`seed = 4391`), trained with `K = 64` distinct random coordinates/batch (`coord_rng = 4392`) under identical `rows-K` + `BVG-K` (**exact equal trials**, `DECISION.md:342`).
    5. `ctrl_reseed_static`: frozen re-seeded random `S1` (`seed = 4390`, `0` updates).
  - **Batching:** Contiguous `<= 4,096`-row batches (`3,893 × 4,096 + 1,856 = 15,947,584`, `3,894` batches/epoch = `15.2` sweeps of all `16,384` `S1` coordinates/epoch; `3` epochs = `45.6` sweeps, `> 11.2M` candidate trials/arm) visiting **100% of `R`'s `15,947,584` targets** every epoch in canonical record order.
  - **Pre-registered `DEV-tune`-only stopping / continuation rule:** Evaluate full `DEV-tune` (`2,670` records, `10,807,417` targets) at Epoch `0` and after every epoch `e = 1..=6`. Zero `DEV-test` peeks. After `e >= 3`:
    - If `DEV-tune` Brier drop from epoch `e - 1` to `e` is `< 200` ppm (`0.02%`) **OR** cumulative `DEV-tune` drop from Epoch `0` is `< 1,000` ppm (`0.10%`), stop (saturated / negligible slope);
    - Otherwise continue up to `max_epochs = 6` (`~91` sweeps of all `16,384` coordinates).
  - Serialize all `5` frozen `UORP5S10` checkpoints with lossless round-trip verification and SHA-256 pins.

### Gate 4 — Single-Shot Sealed `DEV-test` Qualification Gate (`DECISION.md:342`)
- Run `p5-train eval-devtest` ONCE on all `5,321` sealed `DEV-test` records (`21,349,576` non-BOS targets) across all `5` frozen `UORP5S10` checkpoints simultaneously in ONE GPU pass (`P4_MAX_CONTEXT = 2,048`):
  - Report exact `u128` total Q32 Brier, micro-Brier, `ppm` delta vs `baseline_seed1`, exact Q12 bits, micro-BPB, `% P1(A1s)`, `% R*` (all & novel), and `1,000`-draw paired record-level bootstrap 95% CIs (seeds `4393..=4398`).
  - **Instrument verification:**
    1. Live CPU spot-check of `>= 512` (`640`) stratified positions across all `5` arms through `DiscreteCountModel::bind_embedded` + `P4Session::step` + `full_vocab_brier_q32` + `bits_q12`: require **`0` diffs**.
    2. Determinism repeat over first `64` records × `5` arms: require **`0` diffs**.
  - **Exact non-extrapolating verdict rule (zero linear upward extrapolation from short runs):**
    - **`P5_QUALIFIED`** iff ALL hold:
      1. Actual observed `DEV-test` Brier reduction vs `baseline_seed1` clears `>= 1.00%` (for single-arm equiv wall time `<= 86,400 s`; or `>= 1.00% * elapsed / 86,400` if `> 86,400 s`), with paired 95% CI upper bound `< 0`;
      2. `s1_p5_from_seed1` beats `ctrl_reseed_canonical`, `ctrl_reseed_random` (paired 95% CI upper bound `< 0`), and `ctrl_reseed_static` on `DEV-test` Brier;
      3. `DEV-test` BPB does not rise (`all_bits_q12(s1_p5_from_seed1) <= all_bits_q12(baseline_seed1)`);
      4. `cpu_spot_diffs == 0` AND `det_diffs == 0`.
    - **`P5_S1_FROZEN_AT_DEFAULT`** iff instrument is clean (`0` diffs) and any of (1)–(3) fails: freeze `S1` at `seed1.d11` (`04652a3d...`) and close the Phase `P5` `S1`-Pick configuration (never closing the mechanism).
    - **`INSTRUMENT_FAILED`** iff `cpu_spot_diffs != 0 || det_diffs != 0` (halts with non-zero exit code).

### Gate 5 — Conditional `p5.d11` Pack (and Post-Qualification Follow-Up)
- Run `p5-train pack-if-pass`: ONLY if `verdict == "P5_QUALIFIED"`, rewrite `seed1.d11` -> `p5_s1.d11` via `rewrite_uorlut01_s1`, re-bind `fit_json`'s `prepared.trunk_sha256` and `UORS5W02` header digest via `rebind_and_pack_p5_deployment` (without altering any weight words or `S0`/`S2`/`S3`/`S4`/`S6` payloads), pack `p5.d11` (`<= 4,000,000,000 B` with `tokenizer.json`), and cold-reload verify `load_client_artifact` + `bind_embedded` + `identity_pick_codes()`. If `P5_QUALIFIED`, `96`-prompt M1 `uor-chat` parity and Bar A `chat-v0-p2` PRLL re-scoring on `p5.d11` are executed via the existing `antigravity-17/p4-qual` harness before PR closeout. Otherwise write `pack_status.json` recording `S1` frozen at `seed1.d11`.
- Seal all outputs in `receipts.sha256` and verify `sha256sum -c receipts.sha256`.

---

## 4. Cost & Stop Conditions

- **Compute cost:** `$0.00` pod spend (Wick RTX 4090 local host; `0` rented pods). Expected Wick wall time `~8–15` minutes for `3` epochs + `DEV-test` (`<= 30` minutes if `DEV-tune` slope stays `>= 200` ppm/epoch through `6` epochs). Note: an earlier attempt into `20261126-antigravity-p5-qual` was aborted by a Wick host power loss during `DEV-tune` Epoch `0` with zero checkpoints; per `L-002` that partial directory is never reused and this run writes strictly to fresh `OUT_DIR=20261126-antigravity-p5-qual-r2`.
- **Hard stop conditions:**
  1. Any input SHA-256 mismatch, census mismatch, or existing `OUT_DIR` (`L-002`).
  2. Any PTX/SASS float or MMA opcode, or missing kernel symbol in Gate 1.
  3. Any non-zero GPU-vs-CPU diff in Gate 2 (`finite`) or Gate 4 (`>= 512` CPU spot checks or `64`-record determinism repeat), or any equal-trial count mismatch.
  4. Negative on `DEV-test` (`P5_S1_FROZEN_AT_DEFAULT`) is recorded honestly with zero tuning or peeking on `DEV-test`.
