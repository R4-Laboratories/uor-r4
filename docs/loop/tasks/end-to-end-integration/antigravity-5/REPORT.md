# Report: Gate 1 `UORBFY03` Capacity, Stereographic Selectivity & Loss Breakthrough (`antigravity-5`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

- **Pre-registration & FIT:** [SHEET.md](SHEET.md) · [FIT.md](FIT.md) (`TEST FITNESS: FIT`, Round 3, non-author `7124a74a`) · [GO.md](GO.md)
- **Verdict (`SHEET.md` §4 Rule (e)):** **`V3_PLATEAU_AT_RESCALED_SQUARE`** (instrument, `0`-float/`0`-div/`0`-MMA `sm_120` cubin, `0`-diff CUDA=CPU=M1 replay, and Arm 2 BPB/conditioning jump `-0.2362 BPB` `chat-v0-p2` / `-0.3843 BPB` `E1_all` / `3.38×` `shuf_delta` **POSITIVE**; Arm 2 stayed above `< 2.00 BPB` at `2.2980 BPB` AND both arms fell below the `>= 1,000 tok/s` single-core Apple M1 bar at `872 tok/s` / `696 tok/s`, triggering Rule (e) rather than Rule (d), while Arm 1 regressed to `2.7534 BPB` **NEGATIVE** -> unlocks targeted `antigravity-6` serving-path speedup + non-square lawful loss / head & X-ray surgery).
- **Compute & Storage:** `1` Vast.ai `2 × RTX 5090` pod (`vast-62969289`, Quebec CA, `$1.775/h`, `1.62 h`, **`$2.88`** total spend; earlier Hungary pod `vast-62967609` timed out on SSH and was auto-deleted at `$0`; `vast-62969289` released & deleted immediately upon `GATE1_V3_COMPLETE`; `0` pods active). Sealed archive on Wick ZFS: `/mnt/plexpool/uor-r4-corpus/runs/20261128-antigravity-gate1-v3-r2/` (`8/8` `receipts.sha256 OK`, plus `m1_bench_v3_*.json`). Sealed `DEV-test` untouched (never copied to pod).

## What did not work

1. **Both `UORBFY03` arms missed Gate 1 BPB (`< 1.60` / `< 2.00 BPB`), unwrapped E2–E4, and the `>= 1,000 tok/s` Apple M1 serving bar (`V3_PLATEAU_AT_RESCALED_SQUARE`, Rule (e)):**
   - **Single-core Apple M1 serving throughput fell below `1,000 tok/s`** on both arms: Arm 1 (`v3_d256_l8_ffn4_k32`, `19.45 MB`) measured **`872 tok/s`** (`103 ms` open, `1,920 ms` `2,048`-ctx prefill, `41.70 MB` max RSS / `40.93 MB` footprint) and Arm 2 (`v3_d512_l6_ffn4_k64`, `30.27 MB`) measured **`696 tok/s`** (`209 ms` open, `3,177 ms` prefill, `63.77 MB` max RSS / `62.90 MB` footprint). Because Arm 2 achieved `chat-v0-p2 = 2,298,030 <= 2,384,184` with monotone descent (`287,784 -> 263,286`) and `< 50 MB` (`30.27 MB`) **only while falling below `1,000 tok/s` on M1**, `SHEET.md` §4 Rule (d) fails and Rule (e) (`V3_PLATEAU_AT_RESCALED_SQUARE`) governs strictly.
   - **Root cause of the M1 serving slow-down (pure implementation overhead, zero model change):** `ButterflyV3Model::step()` currently delegates to `forward_token_internal()`, which allocates a full `LayerV3StepCache` (`> 300` heap `Vec` allocations/token) AND recomputes `givens_beta_q15(alpha)` (64-bit integer division) `~331,776` times per token at `D=512, L=6, ffn4` on static lifting shears. A zero-allocation scratch-buffer inference path with load-time precomputed `beta_q15` tables (bit-identical `0` diffs) will recover `~4–5×` single-core M1 speed (`> 2,500 tok/s`) without altering a single weight.
   - **Arm 1 (`v3_d256_l8_ffn4_k32`) regressed `+0.2192 BPB` vs `antigravity-4`:** `chat-v0-p2 = 2,753,430` (`2.7534 BPB`, CI `[2.7347, 2.7721]`, `116‰` top-1), `E1_all = 3.1487 BPB` (`241‰` top-1), `E1_nov = 3.0234 BPB` (`51‰` top-1), E2 `0/200` EOS / `200/200` 4-gram loops / `700‰` `"*"` / `437‰` `div@3`, E3 `0/128`, E4 `0/100`.
   - **Arm 2 (`v3_d512_l6_ffn4_k64`) achieved a large `-0.2362` / `-0.3843` BPB gain yet remained above `< 2.00 BPB` (`auto-SFT` did not trigger):** `chat-v0-p2 = 2,298,030` (`2.2980 BPB`, CI `[2.2779, 2.3182]`, `199‰` top-1), `E1_all = 2.6166 BPB` (`316‰` top-1), `E1_nov = 2.5510 BPB` (`126‰` top-1); unwrapped cold E2 `3/200` EOS, `197/200` 4-gram loops, `485‰` `"*"`, `718‰` `div@3` (`46/64`, up from `500‰` in `antigravity-4`), E3 `0/128`, E4 `0/100`.
2. **Initial run `r1` (retired untouched per `L-002`) caught a 1-character typo in `PIN_DEVTUNE_ABSENT_SHA256`:** Gate 1 (`0` PTX/SASS float/div/MMA) and Gate 2 (`0` diffs across all 4 configs) passed on `r1`, after which `train-int` failed closed on a 63-char hex pin string (`...cc58b` missing one `6`). Repaired in `main.rs:82-83`, re-verified `TEST FITNESS: FIT` Round 3 (`7124a74a`), and executed cleanly into fresh `L-002` root `20261128-antigravity-gate1-v3-r2`.

## What changed

- [`crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs`](../../../../../crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs) & [`lib.rs`](../../../../../crates/uor-r4-native-vector/src/lib.rs): Added clean sibling pure-integer `UORBFY03` core (`#![forbid(unsafe_code)]`, `0` float, `0` normalized exponential/squashing, `0` dense matmul) with scalar-reciprocal exact integer RMSNorm (`inv_rms_q40`, `dot_over_denom3_q40`), widened `±8.0` `S³` unit quaternion (`inv_norm_q40`) + exact clamp pullback, pure-`S¹` rational stereographic selective `S³` gate (`g(z) = z^2 / (1 + z^2)` on `rot_y[b+0] + gate_bias_q15[q]`), 4-tap causal depthwise `O(D)` shift-mix (`shift_mix_q15` + `shift_hist` carry), `ffn_branches` parallel `O(D log D)` `ReLU²` Butterfly FFN, wide `2 * pkm_key_dim` (`32 + 32 = 64`) learned Sparse PKM (`±256` value init), dedicated `head_proj: ButterflyLinear`, `build_frequency_sorted_vocab_perm`, Pure-Int Lazy-Sparse AdamW (`g_lim = 1 << 28`, `no_wd` on `shift_mix`/`gate_bias`), lossless `UORBFY03 v1` codec, and 6 new unit tests (`60/60` `uor-r4-native-vector` tests incl. central FD directional checks on all 7 new operator groups and `24/24` opposite-role binding; zero existing `UORBFY01` tests broken).
- [`docs/loop/tasks/end-to-end-integration/antigravity-5/`](.): Added `ORDER.md`, `SHEET.md`, `FIT.md`, `GO.md`, `dispatch.sh`, `butterfly-v3/` (100% pure-integer AOT `sm_120` `butterfly_kernels_v3.cu`, `cuda_backend.rs`, `main.rs`, `build.rs`, `Cargo.toml`), and lightweight JSON evidence + `receipts.sha256` under `evidence/` (no `.uorbfy03` binaries or raw PTX/SASS committed).
- [`docs/roadmap/items/end-to-end-integration.md`](../../../../roadmap/items/end-to-end-integration.md) and [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md): Recorded `antigravity-5` measurements, honest Rule (e) verdict, and next-step `antigravity-6` remediation plan.

## Measurements

### 1. Gate 1 AOT `sm_120` PTX/SASS Audit & Gate 2 Finite Parity

| Gate | Exact Evidence | Result |
|---|---|---|
| **Gate 1 Pure-Int (`butterfly_kernels_v3.sm_120.cubin`, `5859897e...`)** | `ptx_float = 0`, `ptx_div = 0`, `ptx_mma = 0`, `sass_float = 0` (`3,257` logged `sm_120` `HFMA2 -RZ, RZ,` aliases), `sass_mma = 0`, `3/3` kernels | **PASS** |
| **Gate 2 Pure-Int Parity (`4/4` configs: `D=32/256/512`, `ffn4`, `kd=8/32`, `k=32/64`, `Sparsemax64`, `reset_on_bos`)** | `loss_diffs = 0`, `grad_diffs = 0`, `weight_diffs = 0`, `adam_m_diffs = 0`, `adam_v_diffs = 0`, `eval_diffs = 0` across all 4 configs & `UORBFY03` reload | **PASS (`0` diffs)** |

### 2. Dual-Arm `UORBFY03` Pure-Integer Training on `train-100m` (`3,333` steps, `batch=128`, `seq=256`, `lr_q8=4096->512`, `wd_shift=19`)

| Arm | Geometry | Params / Size | `chat-v0-p2` (micro-BPB, 95% CI, Top-1) | `E1_all` (Top-1) | `E1_nov` (Top-1) | `shuf_delta` (95% CI) | Step `2000 -> 3333` |
|---|---|---|---:|---:|---:|---:|---|
| **`antigravity-4` Anchor** | `UORBFY01`, `D=256, L=4, ffn1, k=8` | `5.32M` / `10.65 MB` | `2,534,184` `[2.515, 2.553]` | `3,000,889` | `2,847,494` | `+2,571` `[+171, +4,897]` | `97,122 -> 93,586` |
| **Arm 1: `v3_d256_l8_ffn4_k32`** | `UORBFY03`, `D=256, L=8, ffn4, kd=32, k=32` | `9.72M` / `19.45 MB` | `2,753,430` `[2.735, 2.772]`, `116‰` | `3,148,701` (`241‰`) | `3,023,353` (`51‰`) | `+5,735` `[+4,246, +7,306]` | **`275,744 -> 247,871`** |
| **Arm 2: `v3_d512_l6_ffn4_k64`** | `UORBFY03`, `D=512, L=6, ffn4, kd=32, k=64` | `15.13M` / `30.27 MB` | **`2,298,030`** `[2.278, 2.318]`, `199‰` | **`2,616,579`** (`316‰`) | **`2,551,023`** (`126‰`) | **`+8,693`** `[+6,757, +10,716]` | **`287,784 -> 263,286`** |

**Key Findings:**
- **Arm 2 (`D=512, L=6, ffn4, k=64`) delivered our largest single-step BPB & conditioning jump in pure integer:** `-236,154` micro-BPB (`-0.2362 BPB`) on `chat-v0-p2`, `-384,310` micro-BPB (`-0.3843 BPB`) on `E1_all` (`316‰` top-1), `-296,471` (`-0.2965 BPB`) on `E1_nov`, **`3.38×` stronger prompt conditioning (`shuf_delta = +8,693`, CI `[+6,757, +10,716]`)**, and E2 `div@3` improved `500‰ -> 718‰` (`46/64`), while training at `22,669 tok/s` on 1 RTX 5090 (nearly identical speed to `D=256, L=8` `23,484 tok/s` thanks to scalar-reciprocal RMSNorm).
- **Large `Arm 2` vs `Arm 1` contrast (`-0.455 BPB` `chat-v0-p2`, `-0.532 BPB` `E1_all`, `199‰` vs `116‰` top-1):** Clearly favors wider `D=512` state/embeddings and `k=64` (`1:1` class balance) over deeper/narrower `D=256, L=8, k=32`.

### 3. Apple M1 Native Single-Core Serving Bench (`/usr/bin/time -l`, `2,048` ctx, `128` gen, 7-run median)

| Pure-Int Artifact (`UORBFY03 v1`) | Size | Cold Open | `2,048`-Ctx Prefill | Warm Single-Core Decode | Max RSS / Peak Footprint | RTX 5090 `sm_120` == M1 ARM64 |
|---|---:|---:|---:|---:|---:|---|
| `v3_d256_l8_ffn4_k32.uorbfy03` (`3acc407f...`) | `19.45 MB` | `103 ms` | `1,920 ms` | `872 tok/s` (< 1,000) | `41.70 MB` / `40.93 MB` | **`0` diffs (`0/128` Bar A spot + `0/256` replay)** |
| `v3_d512_l6_ffn4_k64.uorbfy03` (`20262de5...`) | `30.27 MB` | `209 ms` | `3,177 ms` | `696 tok/s` (< 1,000) | `63.77 MB` / `62.90 MB` | **`0` diffs (`0/128` Bar A spot + `0/256` replay)** |

## Checks run at the head

- `cargo test -p uor-r4-native-vector --release`: `60/60` PASS (incl. 7-operator central FD checks & `24/24` opposite-role binding)
- `cargo clippy -p uor-r4-native-vector --lib -- -D warnings`: `0` warnings
- `cargo check -p uor-r4-native-vector --target wasm32-unknown-unknown`: PASS
- `cargo test --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-5/butterfly-v3/Cargo.toml`: `15/15` PASS
- `cargo clippy --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-5/butterfly-v3/Cargo.toml --all-targets -- -D warnings`: `0` warnings
- `python3 -B scripts/check_no_softmax.py` & `python3 -B scripts/check_no_matmul.py` & `uor-harness check --only harness.native_space_contract`: PASS (`0` added hits, `0` failed)
- `shasum -a 256 -c receipts.sha256` (laptop + Wick ZFS): `8/8 OK`; `uor-pod status`: `0` active pods (`vast-62969289` deleted)
- `uor-harness loop pr-check --local`: PASS (`size: tests 0 file(s) +0/-0, code 7 file(s) +5130/-0, docs 7 file(s) +234/-0, generated 0 file(s) +0/-0, other 11 file(s) +3627/-0`)

## Skills used

- `native-space-cycle` (maintained one persistent `i16` Q15 `UORBFY03 v1` state with shared typed integer operators across CUDA `sm_120`, CPU, Apple M1 ARM64, and WASM32 with `0`-diff replay and zero float master)
- `uor-research-architect` (strict pre-registered Rule (e) classification, honest negative-first reporting, and immediate root-cause isolation of both M1 throughput and Arm 2 vs Arm 1 BPB scaling)

## Still open

1. **Unlocked `antigravity-6` (`V3_PLATEAU_AT_RESCALED_SQUARE` remediation):**
   - **Serving-path speedup (bit-identical `0` diffs):** Replace `step()`'s `forward_token_internal` training-cache allocations (`>300` `Vec`s/token) and per-pair runtime `givens_beta_q15` divisions (`331K` divs/token at `D=512, L=6, ffn4`) with zero-allocation scratch buffers in `ButterflyV3Session` + load-time precomputed `beta_q15` tables to clear `>= 2,000–3,000 tok/s` single-core Apple M1 at `D=512, L=6, ffn4`.
   - **X-ray `Arm 2` vs `Arm 1` and break `< 1.60 BPB`:** Inspect why `Arm 2` (`D=512, k=64`) beat `Arm 1` (`D=256, k=32`) by `-0.455 BPB` (`chat-v0-p2`) / `-0.532 BPB` (`E1_all`) while still plateauing at `2.2980 BPB` (separating coarse vs fine bits, Hui-Belkin square-loss calibration/tail gap vs lawful non-square integer loss without `exp`/`log`, and multi-epoch/SFT horizon).
2. Sealed `DEV-test` remains untouched until a pure-integer candidate clears Gate 1 on `DEV-tune`, `chat-v0-p2`, and Apple M1 `>= 1,000 tok/s`.

## Plain summary

We built `UORBFY03` — a 100% pure-integer (`0` float, `0` hardware `/`, `0` normalized exponential/squashing, `0` dense matmul) architecture adding scalar-reciprocal exact integer RMSNorm, 4-tap causal depthwise shift-mix, a pure-`S¹` stereographic selective `S³` gate (`g(z) = z^2 / (1 + z^2)`), widened `±8.0` `S³` rotors, `4×` parallel `O(D log D)` Butterfly FFN, `64`-dim learned Sparse PKM, dedicated `head_proj`, and `TRAIN`-frequency-sorted `vocab_perm` — and ran a FIT-gated dual-arm run on `train-100m` (`$2.88` GPU spend on `vast-62969289`, deleted, `0` pods active). Gate 1 (`0` PTX/SASS float/div/MMA), Gate 2 (`0` diffs across all 4 configs), and Apple M1 replay (`0` RTX 5090 == M1 ARM64 diffs) all passed cleanly. Arm 2 (`D=512, L=6, ffn4, k=64`, `30.27 MB`) dropped `chat-v0-p2` by `-0.2362 BPB` (`2.5342 -> 2.2980 BPB`) and `DEV-tune E1_all` by `-0.3843 BPB` (`3.0009 -> 2.6166 BPB`, `316‰` top-1) while raising prompt-conditioning `shuf_delta` `3.38×` (`+8,693`) — our strongest pure-integer accuracy jump yet — but missed `< 2.00 BPB` and fell below `1,000 tok/s` on single-core M1 (`696 tok/s`, traced to uncached `givens_beta_q15` divisions and training-cache allocations inside `step()`), cleanly classifying `V3_PLATEAU_AT_RESCALED_SQUARE` (`SHEET.md` §4 Rule (e)) and unlocking `antigravity-6`.
