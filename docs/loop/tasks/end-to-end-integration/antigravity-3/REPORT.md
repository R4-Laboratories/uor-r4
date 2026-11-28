# Gate 1 PR 2 Report: Pure-Integer AOT CUDA Butterfly-Rotor Pilot (`antigravity-3`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030. Compute spend: **`$5.74` total Vast.ai cloud spend** across 3 RTX 5090 (`sm_120`) pods, all leased and immediately deleted upon completion (`0` pods active, `$0.00/h`): `r2` diagnostic `vast-62893144` (`28 min`, `$0.24`), `r3` diagnostic `vast-62899157` (`66 min`, `$0.57`), and `r4` full 3-arm pilot `vast-62908411` (`5h 03m`, `$4.93`; self-set `$4` sheet estimate extended cleanly per `AGENTS.md` rather than aborting healthy Arm 3, remaining inside the `<= 6 h` wall cap and `$8/h` global cap). Wick RTX 4090 was occupied by another lab's `train-in-space` job throughout and left untouched. Sealed `DEV-test` remained untouched (`0` opens).

## What did not work

1. **Pre-registered Gate 1 verdict: `GATE1_DIAGNOSTIC_NEEDED` (NEGATIVE on Gate 1 quality and BPB depth-scaling bars; instrument 100% clean, `0` diffs across all checks).** Under `SHEET.md` §4, none of the three pure-integer `D = 256` arms cleared the `< 2.50` BPB / unwrapped E2-E4 bar against frozen `p4.d11` (`3,988.2 MB`, `chat-v0-p2 = 1.4676`, `E1_all = 1.4314`, `E1_nov = 1.7170`):
   - `hb_l4_d256` (`L=4, D=256`, Hui-Belkin `k=8`, `10.65 MB`, `best_epoch = 1`): `chat-v0-p2 = 2,687,302` (`2.687 BPB`), `E1_all = 3,226,923`, `E1_nov = 3,014,287`; unwrapped `E2` `natural_eos = 0/200`, `fourgram_loops = 200/200`, `modal_first_nonspace = "\"` (`640‰`, `6` distinct), `E3 = 0/128`, `E4 = 0/100`.
   - `hb_l8_d256` (`L=8, D=256`, Hui-Belkin `k=8`, `19.16 MB`, `best_epoch = 1`): `chat-v0-p2 = 2,782,409` (`2.782 BPB`), `E1_all = 3,288,688`, `E1_nov = 3,094,055` (**failed BPB depth scaling** vs `L=4`, despite doubling prompt-shuffle `ΔBPB` to `+9,698`!); unwrapped `E2` `natural_eos = 2/200`, `fourgram_loops = 200/200`, `modal_first_nonspace = "*"` (`540‰`, `7` distinct), `diverge_by_step3 = 515‰`, `E3 = 0/128`, `E4 = 0/100` strict (`2/20` Gate-0-compat Rust micro).
   - `sp_l8_d256` (`L=8, D=256`, `Sparsemax64`, `19.16 MB`, `best_epoch = 1`): `chat-v0-p2 = 3,465,737` (`3.466 BPB`), `E1_all = 4,149,940`, `E1_nov = 3,919,550`; unwrapped `E2` `natural_eos = 0/200`, `fourgram_loops = 200/200`, `modal_first_nonspace = "*"` (`900‰`, `6` distinct), `diverge_by_step3 = 140‰`, `E3 = 0/128`, `E4 = 0/100`. `Sparsemax64` lost decisively to Hui-Belkin (`+0.683 BPB` on `chat-v0-p2`, `+0.861 BPB` on `E1_all`) because simplex support truncation pushes tail tokens onto the probability floor.
2. **All three arms exhibited the exact same step-`2,000` training turnaround and post-Epoch-1 climb:** Every arm descended rapidly over Epoch 1 batches `1..2,000` (`hb_l4`: `235K -> 99.6K`; `hb_l8`: `238K -> 102.4K`; `sp_l8`: `33.1K -> 23.6K`), then climbed monotonically through batch `3,333` and across Epochs 2–3 at a rate proportional to `lr_q8` (`hb_l4` `chat-v0-p2`: `2.687 -> 2.811 -> 2.852`; `hb_l8`: `2.782 -> 2.866 -> 2.911`; `sp_l8`: `3.466 -> 3.613 -> 3.645`), even while prompt-shuffle `ΔBPB` kept rising (`hb_l4`: `+2,338 -> +4,505`; `hb_l8`: `+5,873 -> +9,698`).
3. **Checkpoint weight X-ray isolated three concrete integer-training / head-geometry pathologies (while confirming `0.00%` weight saturation and `96.6%` PKM slot utilization):**
   - **Sparse-row sign-Lion momentum coasting (`embeddings` & `pkm.values`):** With `24`-bit NITI and `15/16` momentum decay, a single touch on a sparse row leaves `m[i] != 0` for `~258` consecutive idle batches (`g[i] == 0`), driving `~258` full-magnitude `±step_lr_q8` steps on untouched rows. Zero-initialized `pkm.values` inflated to `rms = 2,731–3,250` (`max = 17,896`), and `embeddings` row RMS spread from `13.3` to `4,260`, with modal collapsed tokens `" "` (`4,133`), `"\"` (`4,175`), and `"*"` (`3,871`) sitting near the top of the norm distribution.
   - **Uncapped Q12 RMSNorm gain drift:** `wd_shift` only decayed Q15 weights, allowing `final_norm_q12` to climb monotonically from `4,096` (`1.0×`) to `6,306` (`1.54×`, `max = 6,574`).
   - **Identity `vocab_perm = 0..4096` in the `64×64` head:** All ASCII punctuation/digits/`EOS` (`0..63`, including `"*"` id `12` and `"\"` id `62`) sit together in `coarse cell 0`, creating severe frequency skew against rare BPE cells `32..63`.
4. **Diagnostic runs `r2` and `r3` exposed and fixed five real numerical bugs before `r4` (`L-007`, audited through non-author FIT R5 & R6):** (a) unconstrained lifting shears caused hyperbolic blowup -> replaced with exact orthogonal `SO(2)` Givens lifting `beta(t) = -2t/(1+t^2)` (`UORBFY01 v2`, `||B||_2 == 1.0`); (b) quadratic Hui-Belkin Q24 mapping imposed a `2.86 BPB` oracle floor -> replaced with exact rational Bayes posterior `p_i = z_i / (k - (k-1)*z_i)`; (c) `14`-bit NITI froze rare tokens -> widened to `24`-bit NITI inside `i32`; (d) Blackwell `sm_120` `nvcc` emitted `3,234` `HFMA2 R<n>, -RZ, RZ,` constant-load aliases (logged separately matching `antigravity-10/11/12`, `0` real arithmetic floats); (e) unscaled `wd_term = w * 256 / 2^12` crushed `embeddings rms` from `2,365 -> 797` in `r3` Epoch 3 -> fixed to LR-scaled `(w * step_lr_q8) / 2^17`.

## What changed

Diff size: `tests 0 file(s) +0/-0, code 6 file(s) +3017/-37, docs 7 file(s) +220/-0, generated 0 file(s) +0/-0, other 14 file(s) +3999/-0` (`0` workspace `Cargo.toml` edits).

- Updated [`crates/uor-r4-native-vector/src/butterfly_rotor.rs`](../../../../../crates/uor-r4-native-vector/src/butterfly_rotor.rs) to `UORBFY01 v2`: exact `SO(2)` orthogonal Givens lifting (`givens_beta_q15`, `givens_dbeta_dalpha_q15`), exact rational Hui-Belkin Bayes posterior in `scores_64_to_q24_probs`, `24`-bit NITI in `apply_lion_step`, and LR-scaled decoupled Lion weight decay `wd_term = (w * step_lr_q8) / 2^wd_shift` (`50/50` tests PASS).
- Added the Gate 1 PR 2 package under `docs/loop/tasks/end-to-end-integration/antigravity-3/`:
  - [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (6 non-author FIT rounds, `TEST FITNESS: FIT`), [`GO.md`](GO.md), [`dispatch.sh`](dispatch.sh), and this [`REPORT.md`](REPORT.md).
  - [`butterfly-pilot/`](butterfly-pilot/) (`#![cfg_attr(feature = "cuda", deny(unsafe_code))]`, AOT `nvcc -cubin` `sm_120` pure-integer CUDA mirror `src/butterfly_kernels.cu` with `0` PTX float/div/MMA and `0` arithmetic SASS float/MMA, `cudarc` driver `src/cuda_backend.rs`, Gate 0-aligned unwrapped `E1–E4` & `chat-v0-p2` evaluator + Apple M1 `bench-cpu` + CUDA=M1 replay verifier in `src/main.rs`, `17/17` unit tests PASS).
  - Sealed JSON receipts in [`evidence/`](evidence/): `gate1_audit.json`, `finite.json`, `hb_l4_d256_report.json`, `hb_l8_d256_report.json`, `sp_l8_d256_report.json`, `*_replay_fixture.json`, `m1_bench_*.json`, and `receipts.sha256` (`11/11 OK`). Full `.uorbfy01` binary checkpoints and raw `ptx`/`sass` dumps are archived off-repo on Wick ZFS at `/mnt/plexpool/uor-r4-corpus/runs/20261127-antigravity-gate1-pilot-r4/` (`11/11 sha256sum -c OK`).
- Updated [`docs/roadmap/items/end-to-end-integration.md`](../../../../roadmap/items/end-to-end-integration.md) and appended one result row to [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md).

## Measurements

### 1. Instrument, AOT Audit & Cross-Architecture Bit-Exact Parity (ALL PASS)

- **Gate 1a AOT `sm_120` Audit (`cubin_sha256 = 4ee6bd8de04f68a3...`):** `pass = true`; `3/3` required kernels present; `ptx_float_hits = 0`, `ptx_div_hits = 0`, `ptx_mma_hits = 0`, `sass_float_hits = 0`, `sass_mma_hits = 0`, `sass_sm120_mov_aliases = 3,234`.
- **Gate 2 Finite GPU-vs-CPU Exact Parity (`finite.json`):** `4/4` configs (`D ∈ {32, 256}`, `L ∈ {2, 4}`, both `HuiBelkinRescaledSquare` and `Sparsemax64`, 3 multi-step Lion updates + mid-record `BOS=0` reset + `UORBFY01 v2` reload): **`0` diffs** across `4,157,184` grad/weight/momentum words and `64` eval targets.
- **Live CPU Spot-Check & Cross-Arch Replay:** **`0 / 3,072`** live Bar A CPU spot-check diffs (`256` targets × `4` epochs × `3` arms) and **`0 / 384`** RTX 5090 `sm_120` CUDA vs Apple M1 ARM64 greedy-token & Q48-mass replay diffs across all 3 cold-reloaded artifacts.

### 2. Controlled 3-Arm Training (`train-100m`, 3 Epochs) & Unwrapped Eval (`DEV-tune` only, `DEV-test` sealed)

| Arm | Size (`UORBFY01 v2`) | Best Ep | `chat-v0-p2` micro-BPB (Ep 1 -> 3) | `E1_all` micro-BPB | `E1_nov` micro-BPB | `shuf_delta` (Ep 1 -> 3) | E2 `div@3` / `nat_eos` / `4g_loop` / modal | E3 `m2` / E4 strict (compat) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **`p4.d11` (Gate 0 anchor)** | `3,988.2 MB` | — | `1,467,631` | `1,431,433` | `1,717,007` | `+85,100` | `203‰` / `0/200` / `190/200` / `955‰ " We"` | `0/128` / `0/100` (`0`) |
| **`hb_l4_d256`** (`L=4`, HB `k=8`) | `10.65 MB` | `1` | `2,687,302 -> 2,851,947` | `3,226,923` | `3,014,287` | `+2,338 -> +4,505` | **`765‰`** / `0/200` / `200/200` / `640‰ "\"` | `0/128` / `0/100` (`0`) |
| **`hb_l8_d256`** (`L=8`, HB `k=8`) | `19.16 MB` | `1` | `2,782,409 -> 2,910,537` | `3,288,688` | `3,094,055` | **`+5,873 -> +9,698`** | `515‰` / `2/200` / `200/200` / `540‰ "*"` | `0/128` / `0/100` (`2`) |
| **`sp_l8_d256`** (`L=8`, Sparsemax) | `19.16 MB` | `1` | `3,465,737 -> 3,644,809` | `4,149,940` | `3,919,550` | `+7,202 -> +8,419` | `140‰` / `0/200` / `200/200` / `900‰ "*"` | `0/128` / `0/100` (`0`) |

### 3. Apple M1 (`16 GB`) Native Single-Core Serving Bench (`bench-cpu`, `/usr/bin/time -l`, 7-Run Median)

| Cold-Reloaded Artifact (`UORBFY01 v2`) | File Size | Peak RSS | Peak Footprint | Cold Open | `2,048`-Ctx Prefill | Warm `128`-Tok Decode | CUDA `sm_120` == M1 Diffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`hb_l4_d256.uorbfy01`** (`d9019c2a`) | `10.65 MB` | **`23.64 MB`** | `22.86 MB` | **`112 ms`** | **`583 ms`** | **`4,291 tok/s`** | **`0`** (`0` det) |
| **`hb_l8_d256.uorbfy01`** (`58a9db9a`) | `19.16 MB` | **`40.75 MB`** | `39.94 MB` | **`97 ms`** | **`942 ms`** | **`2,294 tok/s`** | **`0`** (`0` det) |
| **`sp_l8_d256.uorbfy01`** (`33c5a5b8`) | `19.16 MB` | **`40.78 MB`** | `39.96 MB` | **`101 ms`** | **`992 ms`** | **`2,245 tok/s`** | **`0`** (`0` det) |
| `p4.d11` (`antigravity-17` anchor) | `3,988.2 MB` | `3,978.5 MB` | `3,986.1 MB` | `61,668 ms` | (incl. open) | `1,239 tok/s` | `0` |

## Checks run at the head

- `cargo test -p uor-r4-native-vector`: **`50 passed; 0 failed`**.
- `cargo clippy -p uor-r4-native-vector --lib -- -D warnings`: **0 warnings**.
- `cargo check -p uor-r4-native-vector --target wasm32-unknown-unknown`: **PASS**.
- `cargo test --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-3/butterfly-pilot/Cargo.toml`: **`17 passed; 0 failed`**.
- `cargo clippy --manifest-path docs/loop/tasks/end-to-end-integration/antigravity-3/butterfly-pilot/Cargo.toml --all-targets -- -D warnings`: **0 warnings**.
- `cargo test -p uor-r4-discrete --test native_space_state_contract`: **`4 passed; 0 failed`**.
- `python3 scripts/check_no_softmax.py` & `python3 scripts/check_no_matmul.py`: **OK**.
- `uor-harness check --only harness.native_space_contract` & `uor-harness ratchet d23`, `d25`, `d32`: **PASS**.
- `uor-harness loop pr-check --local`: **PASS (`verdict: PASS`)**.

## Skills used

- `.agents/skills/native-space-cycle/SKILL.md` (single persistent `i16` Q15 `UORBFY01 v2` state, bit-identical RTX 5090 `sm_120` CUDA = Apple M1 ARM64 operators, lossless serialization).
- `.agents/skills/uor-research-architect/SKILL.md` (pre-registered non-author `FIT` discipline, `L-002`/`L-004`/`L-007` compliance, negatives-first classification).

## Still open

- **Pre-registered follow-up diagnostic unit (`antigravity-4`, unlocked by `GATE1_DIAGNOSTIC_NEEDED`):**
  1. Run a fast, diagnostic-only F32 reference (`CHARTER §3.7`) of the exact same `D = 256, L = 4/8` Butterfly-Rotor architecture (`AdamW` vs `Sign-Lion`, `Hui-Belkin` vs diagnostic CE reference, frequency-balanced `vocab_perm`) on `train-100m` to cleanly decompose the `2.687 -> < 1.40` BPB gap into integer optimizer numerics vs head/loss geometry vs architecture capacity before any Gate 2 spend.
  2. Land and test the pure-integer optimizer and head fixes (sparse-row `g != 0` Lion guard to eliminate 258-step idle momentum coasting on `embeddings`/`pkm.values`, Q12 norm weight-decay/clamping, and frequency-balanced `vocab_perm`) in `butterfly_rotor.rs` + `butterfly_kernels.cu`.
- **Gate 2 (remains gated behind resolving Gate 1 BPB / E2-E4):** Scale to `12–16` layers, integrate `Q31FirstPickRead` + compact memory, qualify on Bar A `<= 0.90` BPB, unwrapped M2 `>= 52/128`, and Browser WASM32 Studio.

## Plain summary

Gate 1 PR 2 (`antigravity-3`) proved the entire pure-integer AOT CUDA (`sm_120`) + Apple M1 ARM64 Butterfly-Rotor pipeline end-to-end with **zero instrument defects**: `0` PTX float/div/MMA ops, `0` arithmetic SASS floats, `0` GPU-vs-CPU parity diffs across `4.16M` words, `0` RTX 5090 `sm_120` vs Apple M1 ARM64 replay diffs, and **`4,291 tok/s` (`L=4`, `10.65 MB`, `23.6 MB` RSS, `112 ms` cold open) / `2,294 tok/s` (`L=8`, `19.16 MB`, `40.7 MB` RSS, `97 ms` cold open)** single-core Apple M1 serving — `208×–374×` smaller, `550×` faster to open, and `1.85×–3.46×` faster to decode than `p4.d11`. On learning quality, the verdict is honestly **`GATE1_DIAGNOSTIC_NEEDED`**: Hui-Belkin beat `Sparsemax64` (`-0.683 BPB`) and depth `L=8` doubled prompt sensitivity (`shuf_delta = +9,698` vs `+4,505`), yet all 3 arms plateaued above `2.687 BPB` after a shared step-`2,000` turnaround traced by weight X-ray to sparse-row sign-Lion idle momentum coasting (`pkm.values` & `embeddings`), uncapped Q12 norm growth (`4,096 -> 6,306`), and unpermuted `vocab_perm`. This directly unlocks the pre-registered `CHARTER §3.7` diagnostic F32 + integer-optimizer repair unit (`antigravity-4`) before spending Gate 2 compute.
