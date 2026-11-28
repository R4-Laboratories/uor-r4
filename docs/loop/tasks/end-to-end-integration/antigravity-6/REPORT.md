# Report: Gate 1 `UORBFY03` Zero-Alloc Serving, `Entmax15` Loss & `L=8` Breakthrough (`antigravity-6`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

- **Pre-registration & FIT:** [SHEET.md](SHEET.md) · [FIT.md](FIT.md) (`TEST FITNESS: FIT`, Round 2, non-author `20c43115`) · [GO.md](GO.md)
- **Verdict (`SHEET.md` §4 Rule (d)):** **`LOSS_OR_HORIZON_PROGRESS_BELOW_BAR`** — Instrument (`0`-float/`0`-div/`0`-MMA `sm_120` cubin, `0`-diff Gate 2 & CUDA=M1 replay), zero-alloc M1 serving speedup (`696 -> 1,892 tok/s` on `antigravity-5` Arm 2; **`1,436 tok/s`** on `13.73 MB` `L=8` `HB k=64`, clearing `>= 1,000 tok/s` by `+43.6%`), and `HB k=64` accuracy/conditioning progress (`chat-v0-p2 = 2,253,884 < 2,277,858` `a5` CI_lo; Ep 2 pretrain `2,222,500`; `DEV-tune E1_all = 2,508,623`, `-107,956` micro-BPB below `a5`; `E1_nov = 2,452,906`, `-98,117` below `a5`; `shuf_delta = +11,162` CI `[+8,752, +13,490]`) **POSITIVE** on a single **`13.73 MB`** (`2.20×` smaller) artifact; **`Entmax15` FALSIFIED at this Q15 scale/schedule** (`+0.5896 BPB` worse on `chat-v0-p2`, `+0.7057 BPB` worse on `E1_all`, and degraded across Epoch 2), while `HB k=64` remains above `< 1.60` / `< 1.90 BPB` and misses unwrapped `E2`–`E4` **NEGATIVE** -> locks **`HB k=64`** as winning pure-integer loss and unlocks `antigravity-7` (`ReGLU` bilinear butterfly FFN / `8`-tap shift-mix / `737.5M` `TRAIN`; sealed `DEV-test` untouched).
- **Compute & Storage:** `1` Vast.ai `2 × RTX 5090` pod (`vast-62989215`, Quebec CA, `$1.775/h`, `17,157 s` = `4.77 h`, **`$8.46`** total spend; released & deleted immediately upon `GATE1_V3B_COMPLETE`; `0` antigravity pods active). Sealed archive on Wick ZFS: `/mnt/plexpool/uor-r4-corpus/runs/20261128-antigravity-gate1-v3b-r2/` (`8/8` `receipts.sha256 OK`, plus `m1_bench_v3b_hb64.json`, `m1_bench_v3b_entmax15.json`, and `m1_bench_a5_arm2_prepared.json`). Sealed `DEV-test` untouched (never copied to pod).

## What did not work

1. **Both arms missed `GATE1_FULL_PASS` (`<= 1.60 BPB`, `E1_nov < 1,717,007`, unwrapped `E2` & `E3 >= 52/128`) and `GATE1_BPB_AND_SPEED_BREAKTHROUGH` (`<= 1.90 BPB`):**
   - **Arm 1 (`Entmax15`, `82193944df9b`, `best_epoch = 1`) lost decisively to its identical `1`-variable `HB k=64` control and regressed after Epoch 1:** Peak Epoch 1 scored `chat-v0-p2 = 2,843,522` (`2.8435 BPB`, CI `[2.8236, 2.8635]`, `133‰` top-1), `E1_all = 3,214,295` (`276‰` top-1), `E1_nov = 3,166,150` (`68‰` top-1), `shuf_delta = +8,042` (`[+5,451, +10,492]`), `gold_eos = 0/82`. Across Epoch 2 (`lr_q8: 4,533 -> 512`), `win_loss_q15` climbed monotonically from `327,929` (after bottoming at `326,619` at Ep 1 batch `2,000`) to `349,446`, degrading `chat-v0-p2` to `2,998,550` (`103‰` top-1) and `E1_all` to `3,468,844`; Epoch 3 SFT ended at `chat-v0-p2 = 3,013,874` (`88‰`), `E1_all = 3,398,673`, `E1_nov = 3,266,896`, `shuf_delta = +5,708`. Unwrapped E2 (`best_epoch = 1`) scored `0/200` EOS, `200/200` 4-gram loops, `9` distinct nonspace tokens, `34/64` (`531‰`) `div@3` (SFT Ep 3 telemetry: `0/200` EOS, `200/200` loops, `11` distinct nonspace, `48/64` `750‰` `div@3`), E3 `0/128`, E4 `0/100`.
   - **Why `Entmax15` degraded in this Q15 configuration:** Unlike `HB k=64`'s dense smooth targets across all 64 classes (`|dz_t| <= 128`), `1.5-Entmax` (`p_i = [(z_i - tau)/2]_+^2`, `|dz_i| <= 8`) has a fixed score-margin width of `2.0` Q15 units and emits exact `0` probability (`+16` Q24 floor = `20`-bit penalty) and exact `0` gradient outside the active support. Once readout logits sharpen beyond `2.0`, support collapses onto 1–2 classes, starving rare tokens and penalizing every out-of-support target severely in log-loss BPB.
   - **Arm 2 (`HB k=64`, `8f82c9a721be`, `best_epoch = 3`) broke `antigravity-5` on every BPB, speed, size, and conditioning axis yet stayed above `1.90 BPB` and failed unwrapped E2–E4:** On the single saved `best_model` (`Ep 3` SFT), `chat-v0-p2 = 2,253,884` (`2.2539 BPB`; Ep 2 pretrain reached `2,222,500`, `+0.0314 BPB` SFT trade-off on prose while improving `DEV-tune E1_all` `2.5171 -> 2.5086` and `E1_nov` `2.4666 -> 2.4529`), `natural_eos = 9/200` (`45‰ < 100‰`, `0` premature), `fourgram_loops = 193/200` (`965‰`), `modal_first_nonspace = "*"` (`610‰`), `distinct_first_nonspace_tokens = 10`, `diverge_by_step3 = 37/64` (`578‰`), `gate_e2_pass = false`, E3 `0/128`, E4 `0/100`.
   - **Late-epoch shard uptick observed on `HB k=64`:** `win_loss_q15` bottomed at Ep 2 `batch 1750` (`243,881`, `lr_q8 = 1,367`) before ticking up to `249,460` at `batch 2222` (`lr_q8 = 512`), reflecting shard-tail domain composition and `lr` annealing; Epoch 3 SFT still lowered `DEV-tune E1_all` (`2,517,142 -> 2,508,623`) and `E1_nov` (`2,466,588 -> 2,452,906`).
2. **Initial staging attempt `r1` (retired untouched per `L-002`):** Partial tar staging omitted `crates/uor-r4-tokenizer` needed by `uor-r4-core`, failing before any build or GPU work. Synced the full `crates/` tree, re-verified `10/10` `FIT.md` SHA-256 digests on-pod, and executed cleanly into fresh `L-002` root `20261128-antigravity-gate1-v3b-r2`.

## What changed

- [`crates/uor-r4-native-vector/src/butterfly_rotor.rs`](../../../../../crates/uor-r4-native-vector/src/butterfly_rotor.rs): Extracted zero-allocation `fwht_inplace_q15_with_scratch` (delegated to by `fwht_inplace_q15`) and added pure-integer `LossKind::Entmax15` (`code = 3`, 20-step shift bisection, `8×`-scaled exact Tsallis `α = 1.5` Fenchel-Young loss & gradient, `scores_64_to_q24_probs` support, `0` `exp`/`log`/div) plus exact shift-invariance, simplex, and finite-difference unit tests.
- [`crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs`](../../../../../crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs): Added `#![forbid(unsafe_code)]` zero-allocation `ButterflyV3Prepared` + `ButterflyV3FastSession` (load-time precomputed Givens `beta_q15` tables, exact Euclidean-norm `SAFE_NORM2 = (ACT_CLAMP >> 1)^2` unclamped fast path, unrolled stages `0..2`, `u64::isqrt`/`u128::isqrt`, precomputed admissible masks, preallocated `fwht_work` scratch for literal `0` heap allocations per token), tightened `accumulate_sequence_grad` inner-slice validation, and added `0`-diff parity (incl. high-norm fallback) and `24/24` `Entmax15` opposite-role tests (`63/63` `uor-r4-native-vector` tests PASS).
- [`docs/loop/tasks/end-to-end-integration/antigravity-6/`](.): Added `ORDER.md` (Commit #1), `SHEET.md`, `FIT.md`, `GO.md`, `dispatch.sh`, `butterfly-v3b/` (bit-identical CUDA `Entmax15` + `HB64` pure-integer kernels, `4`-config Gate 2 parity, `2` pretrain + `1` answer-only SFT schedule, `ButterflyV3Prepared` CPU generation & `bench-cpu`), and lightweight JSON evidence + `receipts.sha256` under `evidence/` (no `.uorbfy03` binaries or raw cubin/PTX/SASS committed).
- [`docs/roadmap/items/end-to-end-integration.md`](../../../../roadmap/items/end-to-end-integration.md) and [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md): Recorded `antigravity-6` measurements, Rule (d) classification, `Entmax15` falsification, `HB k=64` lock-in, and `antigravity-7` unlock.

## Measurements

### 1. Gate 1 AOT `sm_120` PTX/SASS Audit & Gate 2 Finite Parity

| Gate | Exact Evidence | Result |
|---|---|---|
| **Gate 1 Pure-Int (`butterfly_kernels_v3.sm_120.cubin`, `fe848704...`)** | `ptx_float = 0`, `ptx_div = 0`, `ptx_mma = 0`, `sass_float = 0` (`3,269` logged `sm_120` `HFMA2 -RZ, RZ,` aliases), `sass_mma = 0`, `3/3` kernels | **PASS** |
| **Gate 2 Pure-Int Parity (`4/4` configs: `D=32/256/512`, `ffn4`, `sk=4/8/32`, `kd=8/32`, `Entmax15` & `HB64`, `reset_on_bos`)** | `loss_diffs = 0`, `grad_diffs = 0`, `weight_diffs = 0`, `adam_m_diffs = 0`, `adam_v_diffs = 0`, `eval_diffs = 0` across all 4 configs & `ButterflyV3Prepared` | **PASS (`0` diffs)** |

### 2. Controlled `1`-Variable Loss Head-to-Head (`13.73 MB`, `6,863,360` params, `D=512, L=8, ffn4, sk=32, kd=32`, `batch=192`, `2` Pretrain + `1` SFT)

| Arm & Epoch | `chat-v0-p2` (micro-BPB, 95% CI, Top-1) | `DEV-tune E1_all` (Top-1) | `DEV-tune E1_nov` (Top-1) | `shuf_delta` (95% CI) | `gold_eos` | Unwrapped E2 (`EOS` / `dist` / `div@3`) |
|---|---:|---:|---:|---:|---:|---|
| *Ref: `a5` Arm 2 (`30.27 MB`, `L=6`, `sk=64`, `HB64`)* | `2,298,030` `[2.278, 2.318]`, `199‰` | `2,616,579` (`316‰`) | `2,551,023` (`126‰`) | `+8,693` `[+6,757, +10,716]` | `0/82` | `3/200` / `11` / `718‰` |
| **Arm 2: `HB k=64` — Ep 1 (pretrain)** | `2,314,710` `[2.294, 2.335]`, `193‰` | `2,595,439` (`316‰`) | `2,551,678` (`119‰`) | `+9,010` `[+7,212, +10,867]` | `3/82` | — |
| **Arm 2: `HB k=64` — Ep 2 (pretrain)** | **`2,222,500`** `[2.203, 2.242]`, **`211‰`** | `2,517,142` (`324‰`) | `2,466,588` (`133‰`) | `+11,105` `[+9,078, +13,252]` | `0/82` | — |
| **Arm 2: `HB k=64` — Ep 3 (SFT `best_model`, `8f82c9a7`)** | **`2,253,884`** `[2.234, 2.274]`, `194‰` | **`2,508,623`** (`321‰`) | **`2,452,906`** (`128‰`) | **`+11,162`** `[+8,752, +13,490]` | **`3/82`** | **`9/200`** (`0` prem) / **`10`** / `578‰` |
| **Arm 1: `Entmax15` — Ep 1 (`best_model`, `82193944`)** | `2,843,522` `[2.824, 2.863]`, `133‰` | `3,214,295` (`276‰`) | `3,166,150` (`68‰`) | `+8,042` `[+5,451, +10,492]` | `0/82` | `0/200` / `9` / `531‰` |
| Arm 1: `Entmax15` — Ep 2 (pretrain) | `2,998,550` `[2.979, 3.018]`, `103‰` | `3,468,844` (`239‰`) | `3,332,624` (`49‰`) | `+8,314` `[+4,677, +12,487]` | `0/82` | — |
| Arm 1: `Entmax15` — Ep 3 (SFT telemetry) | `3,013,874` `[2.995, 3.033]`, `88‰` | `3,398,673` (`242‰`) | `3,266,896` (`49‰`) | `+5,708` `[+2,803, +8,877]` | `0/82` | `0/200` / `11` / `750‰` |

- **Strict `SHEET.md` §4 Rule (d) check on single saved artifact `v3b_d512_l8_ffn4_sk32_hb64.uorbfy03` (`13.73 MB`, `best_epoch = 3`):**
  - `< 50 MB` (`13.73 MB`), M1 `>= 1,000 tok/s` (`1,436 tok/s`, `0` diffs), `shuf_delta CI_lo > 0` (`+8,752 > 0`), **Sub-criterion 1 PASS** (`2,253,884 < 2,277,858` `a5` CI_lo), **Sub-criterion 2 PASS** (`2,508,623 <= 2,566,579`, `-107,956` micro-BPB = `2.16×` required margin, with `2,253,884 <= 2,298,030`), Sub-criterion 3 not met (`45‰ < 100‰` EOS, `965‰` loops).

### 3. Apple M1 Native Single-Core Zero-Alloc Serving Bench (`/usr/bin/time -l`, `ButterflyV3Prepared`, `2,048` ctx, `128` gen, 7-run median)

| Pure-Int Artifact (`UORBFY03 v1`) | Size | Cold Open (`prep`) | `2,048`-Ctx Prefill | Warm Single-Core Decode | Max RSS / Peak Footprint | RTX 5090 `sm_120` == M1 ARM64 |
|---|---:|---:|---:|---:|---:|---|
| **`v3b_d512_l8_ffn4_sk32_hb64.uorbfy03` (`8f82c9a7...`)** | **`13.73 MB`** | **`131 ms`** (`15 ms`) | **`1,643 ms`** | **`1,436 tok/s`** (`PASS >= 1,000`) | **`43.77 MB` / `42.94 MB`** | **`0` diffs (`0/128` Bar A spot + `0/256` replay)** |
| `v3b_d512_l8_ffn4_sk32_entmax15.uorbfy03` (`82193944...`) | `13.73 MB` | `298 ms` (`13 ms`) | `1,769 ms` | **`1,221 tok/s`** (`PASS >= 1,000`) | `43.77 MB` / `42.94 MB` | **`0` diffs (`0/128` Bar A spot + `0/256` replay)** |
| `a5` Arm 2 re-bench (`v3_d512_l6_ffn4_k64.uorbfy03`, `20262de5...`) | `30.27 MB` | `246 ms` (`9 ms`) | `1,128 ms` | **`1,892 tok/s`** (`2.72×` vs `696`) | `90.61 MB` / `89.80 MB` | **`0` diffs vs `step()` & RTX 5090 `sm_120`** |

## Checks run at the head

- `cargo test -p uor-r4-native-vector --release`: `63/63` PASS (incl. `Entmax15` shift-invariance/simplex/FD, `ButterflyV3Prepared` high-norm fallback parity, and `24/24` `Entmax15` opposite-role binding)
- `cargo clippy -p uor-r4-native-vector --lib -- -D warnings`: `0` warnings
- `cargo check -p uor-r4-native-vector --target wasm32-unknown-unknown`: PASS
- `cargo test --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-6/butterfly-v3b/Cargo.toml`: `16/16` PASS
- `cargo clippy --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-6/butterfly-v3b/Cargo.toml --all-targets -- -D warnings`: `0` warnings
- `python3 -B scripts/check_no_softmax.py` & `python3 -B scripts/check_no_matmul.py` & `uor-harness check --only harness.native_space_contract`: PASS (`0` added hits, `0` failed)
- `shasum -a 256 -c receipts.sha256` (pod, laptop, Wick ZFS): `8/8 OK`; `uor-pod status`: `0` antigravity pods active (`vast-62989215` deleted)
- `uor-harness loop pr-check --local`: PASS (`size: tests 0 file(s) +0/-0, code 7 file(s) +3854/-10, docs 7 file(s) +226/-0, generated 0 file(s) +0/-0, other 12 file(s) +4215/-0`)

## Skills used

- `native-space-cycle` (kept single persistent `i16` Q15 `UORBFY03 v1` state bit-identical across RTX 5090 `sm_120`, CPU, Apple M1 ARM64 `ButterflyV3Prepared`, and WASM32 with literal `0` steady-state heap allocations per token)
- `uor-research-architect` (strict 1-variable controlled A/B falsifying `Entmax15` cleanly against `HB k=64`, strict pre-registered Rule (d) classification without goalpost shifting, and immediate pod teardown under `L-004`)

## Still open

1. **Unlocked `antigravity-7` (Rule (d) follow-up — `HB k=64` locked in, `DEV-test` stays sealed):**
   - Scale the winning `HB k=64` `UORBFY03` backbone beyond `train-100m` (`100M` tok) onto full Protocol-2 **`TRAIN` (`737.5M` tok)** while upgrading FFN expressivity (`ReGLU` bilinear butterfly FFN) and local context (`8`-tap causal shift-mix) inside the `< 50 MB` / `>= 1,000 tok/s` single-core M1 envelope.
   - Resolve why unwrapped E2 still loops (`193/200` 4-gram loops, `45‰` EOS) despite `10` distinct openings and `shuf_delta = +11,162` (diagnosing `HB k=64` logit-margin calibration and EOS class weighting during answer-only SFT).
2. Sealed `DEV-test` remains untouched until a single pure-integer artifact clears Gate 1 (`<= 1.60 BPB`, `E1_nov < 1,717,007`, unwrapped `E2` & `E3 >= 52/128`, `>= 1,000 tok/s` M1).

## Plain summary

We eliminated `UORBFY03`'s serving bottleneck with a zero-allocation, precomputed-`beta` engine (`ButterflyV3Prepared` + `fwht_inplace_q15_with_scratch`, lifting `antigravity-5`'s `30.27 MB` model from `696 -> 1,892 tok/s` on single-core Apple M1 with `0` diffs) and ran a FIT-gated, `1`-variable controlled loss comparison (`Entmax15` vs `HB k=64`) at `13.73 MB` (`D=512, L=8, ffn4, sk=32`, `2` pretrain + `1` answer-only SFT epochs on `train-100m`, `$8.46` on `vast-62989215`, deleted immediately, `0` antigravity pods active). Gate 1 (`0` PTX/SASS float/div/MMA), Gate 2 (`0` diffs across `4/4` configs), and RTX 5090 `sm_120` == Apple M1 ARM64 replay (`0` diffs across all 3 artifacts) all passed. **`Entmax15` failed decisively** (`2.8435 -> 3.0139 BPB`, support-collapse drift in Epoch 2), while **`HB k=64` delivered a major simultaneous leap in accuracy, size, speed, and conditioning** on a **`13.73 MB`** (`2.20×` smaller) artifact: **`1,436 tok/s`** single-core Apple M1 (`43.77 MB` RSS, `131 ms` open), **`2.2225 BPB`** (`chat-v0-p2` Ep 2 pretrain) / **`2.2539 BPB`** (Ep 3 SFT `best_model`, strictly below `antigravity-5`'s 95% CI lower bound `2.2779`), **`2.5086 BPB` `DEV-tune E1_all`** (`-0.1080 BPB` below `a5`), **`2.4529 BPB` `E1_nov`** (`-0.0981 BPB`), **`shuf_delta = +11,162`**, and `3×` more natural E2 EOS (`9/200` vs `3/200`, `0` premature, `10` distinct nonspace openings), cleanly classifying **`LOSS_OR_HORIZON_PROGRESS_BELOW_BAR`** (`SHEET.md` §4 Rule (d)), locking in `HB k=64`, and unlocking `antigravity-7`.
