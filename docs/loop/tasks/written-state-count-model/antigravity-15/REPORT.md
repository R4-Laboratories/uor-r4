# REPORT: written-state-count-model, antigravity-15

Lab: `antigravity`. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P3`). Order: [ORDER.md](ORDER.md). Base: `8447b71be`.

**Outcome (`Stage B5b`, single-shot sealed `DEV-test`, `m2-fresh256`, `R4`): Sealed `P3` qualification PASSED (`DECISION.md` §4 & `SHEET.md`). Frozen `p4_full_errwrite` (`5783ba3b`, `p5b_s3_err_p.i8` `0884af50`, `p5b_s3_err_b.i8` `6b1dcaa5`, `p5b_s2_skip_errs.i8` `8c97d916`) scored `213,979,092,098` Q12 bits (`1,205,738` micro-BPB) across all `21,349,576` non-BOS `DEV-test` targets — **`86.94%` of `P1(A1s)` (`-13.06%` fewer bits, 95% CI `[−33,536,913,645, −30,780,167,797]` Q12 << `0`, clearing `<= 99%`)** and **`0.8502 × R*`** (clearing BOTH **`0.95 × R*`** [`1,347,229`] and **`0.90 × R*`** [`1,276,322`]), while error-write beat static session-cache `p4_cache_static` (`215,027,237,344` Q12 / `1,211,644` micro-BPB) by `−1,048,145,246` Q12 bits (controlled 95% CI **`[−1,104,702,968, −987,695,514]` Q12 < `0`**) and `R` mean Brier fell every epoch (`1, 2, 3`). Cold-loaded Stage B5a `c_ppmi_s4.uors4p01` (`cd0e4385`, `1,952,698 B`) reproduced `0` GPU/CPU diffs and cleared all reply bars (`R1 = 128/128`, `R2 = 256/256`, `R4 = 200/240`, `R4 ShuffledRole = 18/240`, `NoMemory = 0/240`).**

## What did not work, first

1. **None of the four `P3` predictors cleared the stretch `0.80 × R*` target (`1,134,508` micro-BPB):** `p4_full_errwrite` reached `1,205,738` micro-BPB (`0.8502 × R*`, `71,230` micro-BPB above `0.80 × R*`), leaving `0.80 × R*` for the scored Phase `P5` geometric core (`DECISION.md` §4).
2. **`p4_c_skip_static` and `p4_c_skip_errwrite` without the causal session cache did NOT beat `P1(A1s)` on `DEV-test`:** Just as on `DEV-tune` in Stage B4, removing the causal session cache left disjoint-`C` `S2-skip` + `golden_shell` at `100.04%` (`1,387,358` micro-BPB, CI `[+99,231,033, +121,262,390]` Q12 above `P1(A1s)`) and `100.03%` (`1,387,148` micro-BPB, CI `[+61,397,039, +84,662,345]`), even though `errwrite` on `p4_c_skip` still improved over static `[−41,171,197, −33,753,926]` Q12. Nearly all of the `-13.06%` overall gain comes from the exact causal `uor_addr` session cache (`p4_cache_static = 87.37%` of `P1(A1s)`), with `p4_full_errwrite` contributing `-5,906` micro-BPB (`[−1,104.7M, −987.7M]` Q12) of genuine non-cache error-write improvement on top.
3. **Cold `S4` `R4` preserved the Stage B5a misses on two OOV templates:** Under cold loading of `cd0e4385`, `"State my occupation."` scored `0/20` (`20` abstained via `role_class = 3`) and `"Clarify my quarters."` scored `0/20`, leaving pooled `R4` at `200/240` (`80/120` OOV-synonym, `120/120` in-stem) and `R4 ShuffledRole` at `18/240` (`117` abstained). Zero tuning or re-runs were performed on `DEV-test`.
4. **Operational notes (`clippy::manual_is_multiple_of` pre-FIT fix):** Before locking [`FIT.md`](FIT.md), Wick's Rust `1.97` toolchain flagged `clippy::manual_is_multiple_of` inside the included [`pick_device.rs`](../../../../../kn-reference/instrument/src/pick_device.rs#L46) harness oracle; annotating `mod pick_device;` in [`main.rs`](p5b-eval/src/main.rs#L6-L9) with `#[allow(unknown_lints, clippy::manual_is_multiple_of)]` kept `crates/**` and `kn-reference` untouched, all `13` file SHA-256s were recomputed and audited by the non-author `pro` FIT reviewer before execution, and the single-shot sealed `DEV-test` dispatch executed cleanly on Wick RTX 4090 (`$0.00` pod spend, `0` pods active).

## What changed

All changes stay strictly inside the three allowed paths (`docs/loop/tasks/written-state-count-model/antigravity-15/**`, `docs/research/notebook/m4-serving.md`, and `docs/roadmap/items/written-state-count-model.md`); zero `crates/**` edits, zero floats/matmul/softmax/runtime CUDA divides, zero `DEV-test` tuning:

1. **Pre-registration, non-author `pro` FIT audit & single-shot dispatch (`antigravity-15/`):**
   - [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (`TEST FITNESS: FIT` across all 13 file SHA-256s), [`dispatch.sh`](dispatch.sh) (`L-002` `RUN_DIR` existence refusal, `flock -n`, `CUDA_DISABLE_PTX_JIT=1`, `sm_89` compute-cap check, Gate 1 PTX/SASS check, Gate 2 `525`-probe `finite` check bound to `cubin_sha256`, Gate 3 `devtest` single-shot check, Gate 4 cold `reply-verify` check, and `receipts.sha256` sealing).
2. **Standalone `p5b-eval` crate (`antigravity-15/p5b-eval/`):**
   - [`p5b_kernels.cu`](p5b-eval/src/p5b_kernels.cu) & [`cuda.rs`](p5b-eval/src/cuda.rs): Pure-integer CUDA kernels (`p5b_kernels.sm_89.cubin` `209249f0`); `525/525` Gate 2 `finite` probes (including 4-class `role_class = 3` S4 probes) matched CPU bit-for-bit.
   - [`devtest_eval.rs`](p5b-eval/src/devtest_eval.rs): `#![forbid(unsafe_code)]`; verifies frozen inputs, replays Stage B4 on `R` (`r.tokens.u16`) FIRST and asserts exact reproduction of all four Stage B4 `UORS6B01` SHA-256s (`e0d5a657`, `a0d50fdc`, `fe7f5d06`, `5783ba3b`), checks monotone `R` epoch Brier descent (`epochs 1..=3`), freezes exact `err_i8` buffers (`p5b_s3_err_p.i8`, `p5b_s3_err_b.i8`, `p5b_s2_skip_errs.i8`), then scores `DEV-test` in record-aligned `2,097,152`-token causal chunks against `R*` (`59f07609`) and `P1(A1s)` with 1,000-draw paired record bootstrap CIs, `PickDevice` parity (`13,607` probes), and `S6`/`S3`/`S2` runtime spot checks (`25,718` probes).
   - [`reply_verify.rs`](p5b-eval/src/reply_verify.rs): `#![forbid(unsafe_code)]`; cold-loads frozen Stage B5a `c_ppmi_s4.uors4p01` (`cd0e4385`) and re-verifies `m2-fresh256` (`1de2fc28`) and `R4` across `active`, `NoMemory`, and 3-candidate Sattolo `ShuffledRole` on GPU and CPU.

## Measurements

Executed as a single shot on Wick RTX 4090 (`sm_89`, `CUDA_DISABLE_PTX_JIT=1`), archived permanently on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/` (`receipts.sha256` `15/15 OK`, `checks_verified = 38`, `STAGE_B5B_SEALED_DEVTEST_COMPLETE`, `$0.00` pod spend):

### 1. Sealed D87 `DEV-test` `P3` Arms (`5,321` records, `21,349,576` non-BOS targets; `14,795,227` novel; `18,751,676` response)

| Arm / Baseline | All Bits (`Q12`) | All (`micro-BPB`) | Novel (`micro-BPB`) | Response (`micro-BPB`) | `% P1(A1s)` | `× R*` | `95% CI vs P1(A1s)` All (`Q12`) | Gate (`<= 99% P1` / `0.95` / `0.90` / `0.80 × R*`) |
|---|---|---|---|---|---|---|---|---|
| `R*` (`59f07609`) | `251,672,950,854` | `1,418,136` | `1,743,746` | `—` | `—` | `1.0000×` | `—` | Baselines: `0.95=1,347,229` / `0.90=1,276,322` / `0.80=1,134,508` |
| `P1(A1s)` (`e2fbf41d`) | `246,099,676,775` | `1,386,732` | `1,681,689` | `1,496,526` | `100.00%` | `0.9779×` | `0` | Canonical `P1(b)` reference |
| `p4_c_skip_static` (`e0d5a657`) | `246,210,815,316` | `1,387,358` | `1,681,931` | `1,497,062` | `100.04%` | `0.9782×` | `[+99,231,033, +121,262,390]` | `false` / `false` / `false` / `false` |
| `p4_c_skip_errwrite` (`a0d50fdc`) | `246,173,576,172` | `1,387,148` | `1,681,779` | `1,496,861` | `100.03%` | `0.9781×` | `[+61,397,039, +84,662,345]` | `false` / `false` / `false` / `false` |
| `p4_cache_static` (`fe7f5d06`) | `215,027,237,344` | `1,211,644` | `1,451,254` | `1,300,228` | `87.37%` | `0.8543×` | `[−32,421,599,264, −29,796,371,282]` | `true` / `true` / `true` / `false` |
| **`p4_full_errwrite` (`5783ba3b`)** | **`213,979,092,098`** | **`1,205,738`** | **`1,443,370`** | **`1,293,392`** | **`86.94%`** | **`0.8502×`** | **`[−33,536,913,645, −30,780,167,797]`** | **`true` / `true` / `true` / `false`** |

- **Controlled error-write deltas (`95%` paired record CI on `DEV-test`, `1,000` draws, seed `4390`):**
  - **`p4_full_errwrite - p4_cache_static` (All Q12):** `−1,048,145,246` Q12 (`−5,906` micro-BPB; **95% CI `[−1,104,702,968, −987,695,514]` Q12** [`< 0`]).
  - `p4_c_skip_errwrite - p4_c_skip_static` (All Q12): `−37,239,144` Q12 (`−210` micro-BPB; **95% CI `[−41,171,197, −33,753,926]` Q12**).
  - `p4_full_errwrite - R*` (All Q12): **95% CI `[−39,521,894,252, −35,903,386,621]` Q12** (`novel`: `185,214,497,448` vs `223,758,978,608` `R*` and `215,795,693,587` `P1(A1s)`, CI vs `P1` `[−31,939,479,809, −29,278,311,645]`).
- **`R` training mean Brier monotonicity (`Q32`, replayed Stage B4 `b4_full`):** Epoch 1 `1,959,327,005 -> 1,896,678,544` (`brier_fell = true`), Epoch 2 `-> 1,837,804,784` (`true`), Epoch 3 `-> 1,781,316,530` (`true`); `b4_skip_err` fell monotonically `1,956,742,338 -> 1,946,919,885 -> 1,943,333,814 -> 1,940,830,640` (`true`).

### 2. Cold-Loaded `S4` Verification (`cd0e4385`, `1,952,698 B`) & Gate Receipts

| Check / Panel | Measured | Bar / Expected | Evidence |
|---|---|---|---|
| Gate 1 PTX / SASS Float & MMA | `0 / 0` (`p5b_kernels.sm_89.cubin` `209249f0`) | `0 / 0` | [`sass-check.txt`](evidence/sass-check.txt) |
| Gate 2 Finite GPU vs CPU | `525/525` (`0` diffs; `0:00.13`, `176,968 KiB`) | `525/525`, `0` diffs | [`finite.json`](evidence/finite.json), [`finite.time`](evidence/finite.time) |
| Gate 3 Wall, RSS & Runtime Parity | `5:58.26`, `7,134,116 KiB`; `shifted_batch_parity = true`, `13,607` Pick & `25,718` spot probes (`0` diffs) | `0` diffs, `shifted = true` | [`devtest_report.json`](evidence/devtest_report.json), [`devtest.time`](evidence/devtest.time) |
| **`m2-fresh256` (`R1` / `R2`)** | **`128/128` pairs / `256/256` rows** (`NoMem 0/256` `+256` > `32`; `Shuf 0/256` w/ `135` abst, `+256` > `32`) | `>= 52/128`, `>= 200/256` + controls | [`reply_verify.json`](evidence/reply_verify.json), [`reply_verify.time`](evidence/reply_verify.time) |
| **`R4` Held-Out Phrasing** | **`200/240`** (`120/120` stem + `80/120` OOV, `80/120` pairs, `20` abst; `NoMem 0/240` `+200` > `30`; **`Shuf 18/240`** w/ `117` abst, `+182` > `30`) | `>= 200/240` + controls | [`reply_verify.json`](evidence/reply_verify.json) |
| `S4` Runtime GPU vs CPU Parity | `0` diffs (`s4_gpu_vs_cpu_divergences = 0`, `0:23.35`, `3,976,656 KiB`) | `0` diffs | [`reply_verify.json`](evidence/reply_verify.json) |

## Checks run at the head

Run in `/Users/casey.allard/uor-r4/.worktrees/antigravity-b5b-devtest`:
- Local `cargo fmt --check`, `cargo test --offline --locked`, `cargo clippy --offline --locked --no-deps --all-targets -- -D warnings`, `bash -n dispatch.sh`: `PASS`
- Wick CUDA `cargo clippy --offline --locked --features cuda --no-deps --all-targets -- -D warnings`, `cargo test --offline --locked --features cuda --no-run`, standalone `nvcc` + `cuobjdump --dump-sass` (`0` float/MMA opcodes): `PASS`
- Wick `GO.json` single-shot dispatch (`STAGE_B5B_SEALED_DEVTEST_COMPLETE`, `checks_verified = 38`) and `sha256sum --strict -c receipts.sha256` (`15/15 OK`): `PASS`
- `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status`: verified `0` pods owned by `antigravity` (`$0/h`)
- `/Users/casey.allard/uor-r4/local/harness-target/debug/uor-harness`: `loop report --check`, `claim-wording`, `shard-check`, `check-formats`, `ratchet d23|d25|d32 --quiet`, `check --only harness.native_space_contract`, and `loop pr-check --local`: run at final Commit 2 head prior to PR push.

## Skills used

- `native-space-cycle` (`.agents/skills/native-space-cycle/SKILL.md`): verified shared pure-integer operators (`p5b_kernels.sm_89.cubin`), `525/525` finite parity, exact Stage B4 `UORS6B01` SHA-256 replay, `25,718` runtime spot checks (`0` diffs), and `15/15` `receipts.sha256`.
- `uor-research-architect` (`.agents/skills/uor-research-architect/SKILL.md`): enforced zero `DEV-test` tuning, non-author `pro` FIT audit (`TEST FITNESS: FIT`) matching all 13 file SHAs, single-shot Wick execution, controlled `errwrite - cache_static` disclosure (`L-010`), and Wick archival.
- `verification-before-completion` (`verification-before-completion/SKILL.md`): verified every local, CUDA, sealed archive, pod, and `uor-harness` gate prior to commit and PR.

## Still open

1. **Phase `P4` — M1 serving integration:** Integrate `P3` (`S6 + S3 + S2-skip + S4`) + causal session cache into ordinary M1 serving (`<= 4,000,000,000 B` artifact & RSS) using the same shared integer state and operators.
2. **Phase `P5` — Scored geometric core & `0.80 × R*` (`1,134,508` micro-BPB):** Build and score the shared geometric core to clear the stretch `0.80 × R*` target while preserving `S4` memory fidelity and causal invariants.

## Plain summary

This single-shot sealed `DEV-test` evaluation qualified Stage B4's frozen `P3` predictors alongside Stage B5a's cold-loaded `S4` memory table (`cd0e4385`) on Wick's RTX 4090 without tuning on `DEV-test` (`$0.00` pod spend, `0` PTX/SASS floats, `0` GPU-vs-CPU divergences across all finite, Pick, mixture, and `S4` checks, and `15/15 OK` on `receipts.sha256`). Across all `21,349,576` non-BOS `DEV-test` targets (`5,321` records), `p4_full_errwrite` scored **`1,205,738` micro-BPB** (`213,979,092,098` Q12 bits), using **`13.06%` fewer bits than `P1(A1s)`** (`86.94%`, clearing the `<= 99%` bar) and reaching **`0.8502 × R*`** (clearing both `0.95 × R*` and `0.90 × R*`, while stopping short of `0.80 × R*`), with genuine non-cache error-write improvement of **`-1,048,145,246` Q12 bits** (`−5,906` micro-BPB) over `p4_cache_static` (`87.37%` `P1(A1s)`) and cold `S4` preserving `R1 = 128/128`, `R2 = 256/256`, `R4 = 200/240`, and `R4 ShuffledRole = 18/240`. That closes Stage B5b (`P3`) and opens **Phase `P4`** (M1 serving integration) and **Phase `P5`** (geometric core).
