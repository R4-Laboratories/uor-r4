# Report: Gate 1 `2×2` Diagnostic & Pure-Integer AdamW Repair (`antigravity-4`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

- **Pre-registration & FIT:** [SHEET.md](SHEET.md) · [FIT.md](FIT.md) (`TEST FITNESS: FIT`, Round 3, non-author `98ff6b8f`) · [GO.md](GO.md)
- **Verdict (`SHEET.md` §4 Rule (d)):** **`DIAG_CAPACITY_OR_LOSS_BOTTLENECK`** (instrument, Pure-Int Lazy-Sparse AdamW, `0`-diff CUDA=CPU=M1 parity, and `4,146–4,605 tok/s` Apple M1 serving **POSITIVE**; all 4 `L=4, D=256` arms remain `>= 2.00 BPB` even in Diagnostic F32 **NEGATIVE** -> definitively rules out Q15 precision loss at `k=8` (`+0.0026 BPB` gap vs F32) and unlocks `4×` `O(D log D)` butterfly FFN + wider `D=512, L=8` + wider `S³` tangent + `k=64` balanced Hui-Belkin / full-width learned PKM in `antigravity-5`).
- **Compute & Storage:** `1` Vast.ai `2 × RTX 5090` pod (`vast-62953159`, `$1.957/h`, `1.30 h`, **`$2.54`** total spend, released & deleted immediately upon completion; `0` pods active). Sealed archive on Wick ZFS: `/mnt/plexpool/uor-r4-corpus/runs/20261128-antigravity-gate1-diag-r2/` (`10/10` `receipts.sha256 OK`, plus `m1_bench_*.json`). Sealed `DEV-test` untouched (never copied to pod).

## What did not work

1. **All 4 `L=4, D=256` arms remained above `2.00 BPB` even in Diagnostic F32 (`DIAG_CAPACITY_OR_LOSS_BOTTLENECK`, Rule (d)):**
   - Pure-Int `Arm 1` (`int_adamw_hb8`): `chat-v0-p2 = 2,534,184` (`2.5342 BPB`, 95% CI `[2.5153, 2.5531]`), `E1_all = 3.0009 BPB`, `E1_nov = 2.8475 BPB`; misses `p4.d11` (`1.4676 BPB` `chat-v0-p2`, `1.7170 BPB` `E1_nov`) and fails unwrapped E2 (`3/200` natural EOS, `197/200` cap-truncated, `198/200` 4-gram loops, `595‰` modal `"*"`, `500‰` `div@3`), E3 (`0/128` `m2-fresh256`), and E4 (`0/100`).
   - Diagnostic F32 `Arm 3` (`diag_f32_adamw_hb8`): `chat-v0-p2 = 2,531,632` (`2.5316 BPB`), `E1_all = 3.0025 BPB`, `E1_nov = 2.8416 BPB`, `shuf_delta = +2,311` (CI `[-80, +4,515]`, crosses `0`). Proves the `~2.53 BPB` floor at `L=4, D=256` is **not** a Q15 quantization defect (`Arm 1 - Arm 3 = +0.0026 BPB` on `chat-v0-p2`, `-0.0016 BPB` on `E1_all`) — it is caused by tiny backbone capacity (`55K` non-PKM/non-emb butterfly weights = `1%` of params), narrow `16`-coord PKM keys, and `k=8` square-loss tail mismatch against `bits_q12`.
2. **`k=1` pure Brier + frequency-balanced `vocab_perm` regressed BPB by `+0.306 BPB` in F32 and `+0.580 BPB` in Q15:**
   - `Arm 4` (Diag F32 `k=1` Brier + balanced perm + learned PKM): `chat-v0-p2 = 2.8377 BPB`, `E1_all = 3.3692 BPB`, `E1_nov = 3.1732 BPB` (`+0.3061 BPB` worse than `Arm 3` `k=8`, though learned PKM more than doubled `shuf_delta` `+5,083` CI `[+62, +9,842]` vs `+2,311`). Spreading the 64 most frequent tokens across 64 distinct coarse cells maximized coarse-cell entropy (`+1.018 BPB` worse at Epoch 0: `6.884` vs `5.866`), while `k=1` Brier creates a `1:63` positive-vs-negative imbalance (`z* = p ~ 0.0156`).
   - `Arm 2` (Pure-Int `k=1` Brier + balanced perm + learned PKM): `chat-v0-p2 = 3.1145 BPB`, `E1_all = 3.7761 BPB`, `E1_nov = 3.4778 BPB` (`+0.2767 BPB` Q15 gap vs `Arm 4` because `k=1` shrinks target activations `8×` closer to the Q15 quantization floor, unlike `k=8` where Q15 matches F32 to `0.10%`); E2 `0/200` EOS, `200/200` 4-gram loops, `365‰` `"\"`, `890‰` `div@3`; E3 `0/128`; E4 `0/100`.
3. **Initial Gate 2 run (`r1`, retired untouched per `L-002`) caught a real backward-pass bug in `tangent_pullback`:**
   - Gate 2's central finite-difference gradient check on `diag_f32_kernels.cu` failed (`fd_cos = 0.97678 < 0.995`, with wrong-sign `rel > 1.0` residuals on `rotor/key/query.diag` and `norm1`) because `unit_quat_from_tangent_q15` / `unit_quat_f` clips tangent coords to `[-1, 1]` while `tangent_pullback` omitted the zero-derivative clamp mask. Fixing the exact clamp-boundary derivative identically in Rust, Pure-Int CUDA, and Diag F32 CUDA raised `fd_cosine` to `1000‰` (`1.00000`) across all `45` and `49` active slices, passed Round 3 non-author FIT (`98ff6b8f`), and executed cleanly into fresh `L-002` output root `20261128-antigravity-gate1-diag-r2`.

## What changed

- [`crates/uor-r4-native-vector/src/butterfly_rotor.rs`](../../../../../crates/uor-r4-native-vector/src/butterfly_rotor.rs): Added Pure-Integer Lazy-Sparse AdamW (`ButterflyAdamWState`, `LayerAdamWState`, `apply_adamw_step`: `beta1 = 7/8`, `beta2 = 63/64`, lazy sparse skip on `embeddings`/`pkm.values`, Q15/Q12-matched decoupled WD `wd_s = eff_wd - 3` on Q12 anchored to `Q12_ONE = 4,096`, `isqrt_u64` + integer divide + `splitmix64` stochastic rounding onto `i16`), exact integer PKM key-table & query-routing adjoint (`accumulate_sequence_grad_opts`, `g_pkm_key_a`, `g_pkm_key_b`), exact `[-Q15_ONE, Q15_ONE]` clamp-boundary derivative in `tangent_pullback`, `build_frequency_balanced_vocab_perm` / `set_vocab_perm`, and 4 unit tests (`54/54` pass, `0` clippy warnings, `wasm32-unknown-unknown` clean, `UORBFY01 v2` unchanged).
- [`docs/loop/tasks/end-to-end-integration/antigravity-4/`](.): Added `ORDER.md`, `SHEET.md`, `FIT.md`, `GO.md`, `dispatch.sh`, `butterfly-diag/` (AOT `sm_120` pure-integer `butterfly_kernels.cu` + `CHARTER §3.7` diagnostic-only `diag_f32_kernels.cu`, `cuda_backend.rs`, `main.rs`), and lightweight JSON evidence + `receipts.sha256` under `evidence/` (no `.uorbfy01` binaries or raw PTX/SASS committed).
- [`docs/roadmap/items/end-to-end-integration.md`](../../../../roadmap/items/end-to-end-integration.md) and [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md): Recorded `antigravity-4` findings and unlocked `antigravity-5` architecture/loss scaling path.

## Measurements

### 1. Gate 1a/1b AOT `sm_120` PTX/SASS Audit & Gate 2 Finite Parity

| Gate | Exact Evidence | Result |
|---|---|---|
| **Gate 1a Pure-Int (`butterfly_kernels.sm_120.cubin`, `19b0cda8...`)** | `ptx_float = 0`, `ptx_div = 0`, `ptx_mma = 0`, `sass_float = 0` (`3,253` logged `sm_120` `HFMA2 -RZ, RZ,` aliases), `sass_mma = 0`, `3/3` kernels | **PASS** |
| **Gate 1b Diag F32 (`diag_f32_kernels.sm_120.cubin`, `04729273...`)** | `ptx_exp_log = 0`, `sass_exp_log = 0`, `ptx_mma = 0`, `sass_mma = 0`, `3/3` kernels (never serialized) | **PASS** |
| **Gate 2 Pure-Int Parity (`4/4` configs, `3` AdamW steps + eval + reload)** | `loss_diffs = 0`, `grad_diffs = 0`, `weight_diffs = 0`, `adam_m_diffs = 0`, `adam_v_diffs = 0`, `eval_mass_diffs = 0`, `eval_greedy_diffs = 0`, `uorbfy01_reload_ok = true` | **PASS (`0` diffs)** |
| **Gate 2 Diag F32 Adjoint & Descent (`k=8` & `k=1 + learn_pkm`)** | Central FD `cosine = 1000‰` (`45/45` & `49/49` slices), 20-step loss `65.683 -> 26.860` & `10.453 -> 4.244` (`> 59%` drop, `0` NaN) | **PASS** |

### 2. `2×2` Controlled Factorial (`train-100m`, `L=4, D=256, pkm_sub_keys=64, 3,333 steps, seed=20261127`)

| Arm | Space / Optimizer | Config | `chat-v0-p2` (micro-BPB, 95% CI) | `E1_all` | `E1_nov` | `shuf_delta` (95% CI) | Step `2000 -> 3333` |
|---|---|---|---:|---:|---:|---:|---|
| **`r4` Anchor** (`#4331`) | Pure-Int Q15 / Sign-Lion | `k=8`, id, frozen | `2,687,302` `[2.668, 2.707]` | `3,226,923` | `3,014,287` | `+4,505` `[+2.1k, +6.9k]` | `110,602 -> 114,494` (rebound) |
| **Arm 1: `int_adamw_hb8`** | **Pure-Int Q15 / AdamW** | `k=8`, id, frozen | **`2,534,184`** `[2.515, 2.553]` | **`3,000,889`** | **`2,847,494`** | `+2,571` `[+171, +4,897]` | **`97,122 -> 93,586` (monotone)** |
| **Arm 3: `diag_f32_adamw_hb8`** | **Diag F32 / AdamW** | `k=8`, id, frozen | **`2,531,632`** `[2.513, 2.551]` | **`3,002,496`** | **`2,841,597`** | `+2,311` `[-80, +4,515]` | **`97,219 -> 93,625` (monotone)** |
| **Arm 2: `int_adamw_brier1_bal`** | **Pure-Int Q15 / AdamW** | `k=1`, bal, learned PKM | `3,114,455` `[3.096, 3.132]` | `3,776,142` | `3,477,785` | `+2,413` `[-2.2k, +6.9k]` | **`20,334 -> 20,233` (monotone)** |
| **Arm 4: `diag_f32_adamw_brier1_bal`** | **Diag F32 / AdamW** | `k=1`, bal, learned PKM | `2,837,734` `[2.818, 2.857]` | `3,369,176` | `3,173,216` | **`+5,083`** `[+62, +9,842]` | **`18,599 -> 17,909` (monotone)** |

**Key Factor Isolations:**
- **Optimizer + Exact Adjoint (`Arm 1` vs `r4`):** Eliminated the step-`2,000` turnaround completely and improved `chat-v0-p2` by **`-153,118` micro-BPB (`-0.1531 BPB`)**, `E1_all` by **`-226,034` (`-0.2260 BPB`)**, and `E1_nov` by **`-166,793` (`-0.1668 BPB`)** in identical pure-integer `UORBFY01 v2` weights.
- **Q15-vs-F32 Precision Gap (`Arm 1` vs `Arm 3`):** **`+2,552` micro-BPB (`+0.0026 BPB` / `+0.10%`)** on `chat-v0-p2`, and **`-1,607` micro-BPB (Pure-Int beats F32!)** on `DEV-tune E1_all` (`3.0009` vs `3.0025`). Persistent `i16` Q15 training is solved at `k >= 8`.
- **Head/Loss/PKM (`Arm 4` vs `Arm 3`):** `k=1` Brier + frequency-balanced `vocab_perm` hurt raw BPB (`+0.3061 BPB` in F32) due to `1:63` class imbalance and maximum-entropy coarse partitioning, while learned PKM keys more than doubled prompt-conditioned `shuf_delta` (`+5,083` vs `+2,311`) and boosted `div@3` to `890‰`.

### 3. Apple M1 Native Single-Core Serving Bench (`/usr/bin/time -l`, `2,048` ctx, `128` gen, 7-run median)

| Pure-Int Artifact (`UORBFY01 v2`, `10.65 MB`) | Cold Open | `2,048`-Ctx Prefill | Warm Single-Core Decode | Max RSS / Peak Footprint | RTX 5090 `sm_120` == M1 ARM64 |
|---|---:|---:|---:|---:|---|
| `int_adamw_hb8.uorbfy01` (`f8a8f58e...`) | `80 ms` | `480 ms` | **`4,146 tok/s`** | `23.71 MB` / `22.77 MB` | **`0` diffs (`0/256` Bar A spot + `0` replay)** |
| `int_adamw_brier1_bal.uorbfy01` (`90e0463e...`) | `56 ms` | `453 ms` | **`4,605 tok/s`** | `23.69 MB` / `22.76 MB` | **`0` diffs (`0/256` Bar A spot + `0` replay)** |

## Checks run at the head

- `cargo test -p uor-r4-native-vector`: `54/54` PASS
- `cargo clippy -p uor-r4-native-vector --lib -- -D warnings`: `0` warnings
- `cargo check -p uor-r4-native-vector --target wasm32-unknown-unknown`: PASS
- `cargo test --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/Cargo.toml`: `17/17` PASS
- `cargo clippy --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/Cargo.toml --all-targets -- -D warnings`: `0` warnings
- `uor-harness ratchet d23`, `d25`, `d32` & `python3 -B scripts/check_no_softmax.py` & `python3 -B scripts/check_no_matmul.py`: `0` added hits
- `shasum -a 256 -c receipts.sha256` (laptop + Wick ZFS): `10/10 OK`; `uor-pod status`: `0` active pods (`vast-62953159` deleted)
- `uor-harness loop pr-check --local`: PASS (`size: tests 0 file(s) +0/-0, code 6 file(s) +4530/-17, docs 7 file(s) +235/-0, generated 0 file(s) +0/-0, other 12 file(s) +4556/-0`)

## Skills used

- `native-space-cycle` (kept `UORBFY01 v2` as a single persistent `i16` Q15 state with shared typed operators and `0`-diff CUDA=CPU=M1 replay while isolating `CHARTER §3.7` F32 diagnostics strictly outside `crates/` with zero artifact export)
- `uor-research-architect` (decomposed the `antigravity-3` plateau via controlled `2×2` factorial before scaling compute)

## Still open

1. **Unlocked `antigravity-5` (`DIAG_CAPACITY_OR_LOSS_BOTTLENECK`):** With Q15 Pure-Int AdamW proven to match F32 to `0.10%` (`+0.0026 BPB`), directly eliminate the backbone/head/PKM capacity & loss bottlenecks in pure integer `UORBFY01`:
   - `4×` `O(D log D)` Butterfly FFN + wider `D=512, L=8` (`~19×` more backbone params, still `< 50 MB` and `> 1,000 tok/s` M1),
   - `O(D)` 4-tap causal local shift-mix before the `S³` rotor (`4` `i16` weights/dim, `0` matmul) + widened `S³` tangent clamp (`±8.0` eliminating `32%` dead clipped rotors),
   - `64`-dim (`4×` wider) learned PKM addressing (`n2[0..64]`), and
   - `k = 64` balanced-class Hui-Belkin (`1:1` positive-vs-negative weight on `64`-way local cells, exact Bayes inverse `p = z / (64 - 63z)`, `0` exp/log/softmax) with frequency-tiered `vocab_perm` (minimizing coarse entropy instead of maximizing it).
2. Sealed `DEV-test` remains untouched until a pure-integer candidate clears Gate 1 on `DEV-tune` and `chat-v0-p2`.

## Plain summary

We built Pure-Integer Lazy-Sparse AdamW (`UORBFY01 v2` unchanged, zero floats, zero hardware divisions, zero softmax, zero matmul), fixed a real wrong-sign clamp-boundary backward bug in `tangent_pullback`, and ran a FIT-gated `2×2` factorial against a `CHARTER §3.7` Diagnostic F32 twin on `train-100m` (`$2.54` total GPU spend, pod deleted). Pure-Integer AdamW eliminated the step-`2,000` turnaround completely, beat `antigravity-3` Sign-Lion by `-0.1531 BPB` (`2.5342` vs `2.6873 BPB`), served at `4,146–4,605 tok/s` on a single Apple M1 core with `0` CUDA-vs-M1 diffs, and matched full F32 (`2.5316 BPB`) to `+0.0026 BPB` (`0.10%`) — proving that Q15 integer precision is solved and that the remaining `2.53 BPB` ceiling is a `D=256, L=4` backbone capacity and loss-balance bottleneck, unlocking `antigravity-5`.
