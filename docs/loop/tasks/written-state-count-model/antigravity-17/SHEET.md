# Pre-Registration Sheet: Phase P4 Full-Dose Serving Qualification & Bar A PRLL (`antigravity-17`)

- **Lab:** `antigravity`
- **Item / Issue / Phase:** `written-state-count-model` (`#4063`), **Phase `P4` PR 2 of 2** (`DECISION.md:315-333`, `M2+/P4` + `M4` Bar A PRLL)
- **Prerequisite:** Phase `P4` PR 1 (`antigravity-16`, PR `#4301`, merged at `d13c035dd`) + Stage B5b (`antigravity-15`, PR `#4298`, merged at `37ad6b3b4`) + M4 PRLL pre-registration (`docs/harness/prep/m4-lawful-likelihood-bpb-2026-10-23.md`, `TEST FITNESS: FIT`).
- **Scope & Constraints:** Pure-integer native-space execution (D85, CHARTER §3). Zero floats (`f32`/`f64`/PTX/SASS float ops), zero matrix multiply / MMA, zero normalized exponential or squashing. Frozen artifacts only — zero retraining or tuning on any evaluation stream (`DEV-test` is NOT read in this unit).

---

## 1. Instrument & Hardware

1. **Wick RTX 4090 (`sm_89`, `24 GB` VRAM, `$0.00` pod spend):**
   - Standalone crate `docs/loop/tasks/written-state-count-model/antigravity-17/p4-qual/` with AOT `nvcc -cubin -arch=sm_89 -O3` pure-integer CUDA kernels (`src/p4_kernels.cu`, including `crates/uor-r4-discrete/src/kn_score_division.cuh`).
   - Loads the exact packed `p4.d11` (`uor-r4.lut-discrete-p4/1` / `UORP4A01`) binary slices (`UORKNC03`, `UORS3C01`, `UORS2C01`, `UORS4P01`, `UORS6B01`, and `identity_pick_codes()`) directly into device memory with zero float conversion or quantization (`~4.2 GB` VRAM).
   - Executes both the full `4,095`-token `P4` forward pass (including `Pick`, `UORKNC03`, `UORS3C01` primary/backoff/Starling-7/uniform, `UORS2C01` 3 skip families, incremental `CausalSessionCache`, `UORS4P01` role voting, and `UORS6B01` 22-expert exact 80-bit mixture) and exact `PRLL` (`masses_from_scores` + `bits_q12`, verbatim `crates/uor-r4-integer/src/lawful_likelihood.rs`).
2. **Local Apple M1 (`16 GB` RAM, macOS `aarch64`):**
   - Release binary `uor-chat` (`cargo build --release -p uor-r4-integer --features p1-count-serving --bin uor-chat`) serving the identical sealed `p4.d11` file transferred from Wick.
   - Measured with `/usr/bin/time -l` (reporting both `maximum resident set size` and `peak memory footprint`) and `UOR_P1_TIMING=1` (`open_and_prefill_ns` and untraced `decode_ns` at `2,048`-token context).

---

## 2. Frozen Inputs & Data Split (Fail-Closed SHA-256 Pins)

Every input digest is verified before computation; any mismatch halts with a non-zero exit code:

| Role | Path on Wick | Exact Bytes | Exact SHA-256 |
|---|---|---:|---|
| Base D11 trunk (`seed1`) | `/home/callard/uor-codex/20261118-diagnostics/inputs/seed1.d11` | `86,536,132` | `04652a3de8e966a23539b04fbd77c7d9872d9f13a541fcb224d48e766e2c147d` |
| Bound trunk tokenizer | `/home/callard/uor-codex/20261118-diagnostics/inputs/tokenizer.json` | `265,393` | `a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377` |
| `A1s` fit manifest | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.manifest.json` | `2,523` | `26f8f2225a8a0735400a647e37f2f97d86cde9f2de8d847ef791469a6d6b02ec` |
| `A1s` Pick weights (`UORS5W02`) | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate5-eval/fit/identity/a1s.weights` | `98,604` | `e2fbf41d38d9dbe3449cdbb85c8bd335c5ee1f803e35bc7993070d65c0d33293` |
| Full-`C` count trie (`UORKNC03`) | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-uorknc03/gate3-fullc/full-c.v3` | `3,804,378,885` | `75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233` |
| `S3` Golden-Shell (`UORS3T02`) | `/mnt/plexpool/uor-r4-corpus/runs/20261125-antigravity-p2-fallback-devtune/tables/golden_shell.uors3t02` | `99,999,994` | `e4c4d1a8c571828221271c76f9062c05e87d0a7f3cc4e1922ff62dd1d8b2c42c` |
| Stage B5b `S3` primary `err_i8` | `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p5b_s3_err_p.i8` | `9,722,347` | `0884af50694b119220f15fc7401e41365e038130fe5304039a6dc065bb114c6e` |
| Stage B5b `S3` backoff `err_i8` | `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p5b_s3_err_b.i8` | `2,557,657` | `6b1dcaa559850d51d3aa08a4678a9a85c9dfa80fade1c9cfa51003050c4c6fcd` |
| Disjoint-`C` `S2-skip` (`UORS2S01`) | `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-eval-r3/evidence/artifacts/s2_skip.uors2s01` | `80,106,901` | `85307b88f6bbc43d57fe8fe896789f9c8ea97e8e25bc24b074a3932e9332c777` |
| Stage B5b `S2-skip` `err_i8` | `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p5b_s2_skip_errs.i8` | `11,401,868` | `8c97d916e7fef4a54b547389a61caf8f68124fb0a2f7cafa44823725fc032a10` |
| Stage B5a `S4` (`UORS4P01`) | `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5a-s4-r3/artifacts/c_ppmi_s4.uors4p01` | `1,952,698` | `cd0e43859bcd72acfa08342d7ff08a0827c582c4cd8e1592218e6ca4c43f8771` |
| Stage B5b `S6` (`UORS6B01`) | `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p5b-devtest/artifacts/p4_full_errwrite.uors6b01` | `464,704` | `5783ba3b8b03cb4e9c35ed335b89c8ff36797175e9e1b8d9db40ddfd64f46a2d` |
| Bar A stream (`chat-v0-p2`) | `/mnt/plexpool/uor-r4-corpus/inputs/chat-v0-p2/heldout.u16` | `12,389,242` | `e5f400b0e676791fd0458ce810163c5d2542adda80dccc8282f37d9ca8904cef` |

- **Tokenizer equivalence verified:** `a9fb13f7...` (`265,393 B`, `seed1.d11` bound tokenizer) and `d36d3e87...` (`109,457 B`, `chat-v0-p2` manifest tokenizer) have bit-identical `model.vocab` (`4,096` entries) and `model.merges`. Size budget uses the larger `265,393 B` bound tokenizer.
- **Stream byte basis (`chat-v0-p2` `heldout.u16`):** `6,194,589` `u16` tokens after `64`-byte `UORT` header (`17,576,697` stream bytes, `2.837427` bytes/token; `0` OOV; `max_id = 4095`). Target-byte sample basis is computed from `tokenizer.json` and reported strictly as a separate diagnostic.

---

## 3. Execution Steps (Gates 1–5)

### Gate 1 — AOT `sm_89` PTX & SASS Zero-Float / Zero-MMA Audit
- Build `p4-qual --release --features cuda` with `UOR_NVCC_ARCH=compute_89` (`nvcc -cubin -arch=sm_89 -O3`).
- Inspect both `nvcc -ptx` and `cuobjdump --dump-sass` on the compiled cubin: require **`0` float instructions** (`FADD`, `FMUL`, `FFMA`, `FSETP`, `F2I`, `I2F`, `DADD`, `DMUL`, `DFMA`, `MUFU`, `.f16/.f32/.f64/.bf16`) and **`0` MMA / tensor-core instructions** (`HMMA`, `IMMA`, `BMMA`, `DMMA`, `WMMA`, `GMMA`). Record `cubin_sha256`.

### Gate 2 — Synthetic Finite GPU-vs-CPU Exact Parity
- Construct a deterministic synthetic `V = 4,096` `P4State` (`UORKNC03`, `UORS3C01`, `UORS2C01`, `UORS4P01`, `UORS6B01`, `codes`) exercising every code path: `S3` primary/backoff/Starling-7/uniform miss, `S2-skip` families `0/1/2` hit/miss, `CausalSessionCache` hit states `0/1/2` (including `BOS` reset, `64`-token unigram decay, LCP `match_len > 3`, and `2,048`-token slide), `Pick` classes `0..=4` + `read_enabled = false`, `S4` active/tie/abstention, and `PRLL` (`masses_from_scores` + `bits_q12` including `target = 0` BOS floor and tie-breaking).
- Assert **`0` raw-word differences** between CUDA `p4_kernels.cu` and CPU `P4Session::step` + `lawful_likelihood` across all `4,095` token scores, all trace fields, and all `PRLL` masses & `bits_q12`.

### Gate 3 — Full-Dose `p4.d11` Pack & `96`-Prompt CUDA-vs-CPU Parity Fixture
- Run `p4-qual pack`: verify all 12 Wick input SHA-256s, losslessly encode `UORS3C01` and `UORS2C01` with Stage B5b `err_i8` buffers, write `p4.d11` via `write_p4_deployment`, assert `size(p4.d11) + 265,393 <= 4,000,000,000 B`, and cold-reload via `load_client_artifact` + `bind_embedded`.
- Run `p4-qual parity-fixture`: construct **`96` deterministic prompts** (`>= 64` bar) drawn across real `chat-v0-p2` prefixes and synthetic boundary prompts spanning:
  - Short (`1..=16` tokens), medium (`32..=256`), long (`512..=1,536`), and exact `2,048`-token context (`>= 12` prompts at `len == 2,048` + `4`-step sliding rollouts).
  - Both `read_enabled = true` (`84` prompts) and `read_enabled = false` (`12` prompts).
  - Multi-step rollouts (`1` to `4` greedy decode steps, including `> 2,048` window slides).
  - Coverage assertions logged in `parity_fixture_summary.json`: all `S3` hit modes observed (`primary`, `backoff`, `starling`, `uniform_miss` where reachable on real data), `S2-skip` hits & misses, all `Cache` hit states `0/1/2`, `Pick` classes `0/1/2/3/4` (at least `0`, `1`, and `>= 3`), and `S4` active & abstention cases.
- Run every prompt on **Wick RTX 4090 CUDA** and verify **`0` diffs** against **Wick CPU `P4Session::step`** across all `4,095` Q48 scores + every carried trace field + `mass_sha256`, writing `cuda_parity_96.json`.

### Gate 4 — Bar A (`chat-v0-p2` `heldout.u16`) GPU PRLL Scoring on Wick
Verbatim `docs/harness/prep/m4-lawful-likelihood-bpb-2026-10-23.md`:
- Verify `heldout.u16` size (`12,389,242 B`) and SHA-256 (`e5f400b0...`), token count `6,194,589`, `max_id = 4095`.
- Deterministic window rule (`context = 384`): `stride = (6,194,589 - 384 - 1) / windows`, starts `k * stride`, `384` targets `tokens[start + t + 1]` for `t in 0..384` with one fresh session per window (`BOS = 0` resets `active_prefix = [0]` and `CausalSessionCache`; intra-dialogue `EOS = 1` stays in context without aborting).
- At every position, feed `P4`'s `4,096`-token integer score vector (`scores[0] = 0`, `scores[1..4096] = p4_mass_q48[1..4096]`) through exact `PRLL` (`masses_from_scores`, every `mass >= 1`, `sum == 2^48`) and sum `bits_q12(prll_mass[target])`.
- Arms & controls scored on the identical windows:
  1. **`P4` (`p4_full_errwrite`) primary `512` windows (`196,608` targets, `stride = 12,098`)** — stream basis (`2.837427 B/tok`) + target-byte sample basis diagnostic + `non_bos` vs `bos` breakdown.
  2. **`P4` secondary `64`-window diagnostic (`24,576` targets, `stride = 96,784`)** — reported separately, never compared as one number with `512`.
  3. **`P4` determinism control:** second independent GPU pass over all `512` windows; require **`0` Q12 bit difference** on every window and total.
  4. **CPU spot-check parity:** evaluate **`>= 512` deterministic positions** (at least 1 per window across all `512` windows, including `BOS = 0` target positions!) on host CPU via `P4Session::step` + `uor_r4_integer::lawful_likelihood::{masses_from_scores, bits_q12}`; require **`0` diffs** on `p4_mass_q48`, `prll_mass_q48`, and `bits_q12`.
  5. **Baseline comparison `P1(A1s)`** (same `UORKNC03` + `A1s` `UORS5W02` weights, `w[14..22] = 0`) on the identical `512` windows through the exact same `PRLL` scorer, plus **`Uniform U` sanity floor** (`12 * 4096` Q12/tok = `4.2292` BPB on stream basis).
  6. **Interval & auto-extension rule (`m4-lawful-likelihood-bpb-2026-10-23.md` §3):** With `z = 1.96`, `se_n = 14,525 * sqrt(512 / n)` micro-BPB (`se_512 = 14,525`, `se_1024 = 10,270`, `se_2048 = 7,262`; empirical per-window SE also reported):
     - **`PASS`** iff `hi = est + z * se_n <= 900,000` micro-BPB (`0.90` BPB);
     - **`FAIL`** iff `lo = est - z * se_n > 900,000` micro-BPB (`0.90` BPB);
     - **`INCONCLUSIVE`** iff `lo <= 900,000 < hi` -> automatically extend to `n = 1,024`, then `n = 2,048`; `INCONCLUSIVE` at `2,048` is recorded as an undecided configuration with no bar claim.
- Seal all Wick outputs and SHA-256s in `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual/receipts.sha256`.

### Gate 5 — Local Apple M1 (`16 GB` RAM) `uor-chat` Qualification
- Transfer sealed `p4.d11` and `cuda_parity_96.json` from Wick to local `/tmp/p4-m1-eval/` (untracked, never committed) and verify SHA-256 identity.
- Build release `uor-chat` (`--features p1-count-serving`) and host `p4-qual` (`verify-m1`) on Apple M1 (`aarch64-apple-darwin`).
- **Parity check (`p4-qual verify-m1`):**
  1. Load `/tmp/p4-m1-eval/p4.d11` once on Apple M1 via the identical `uor-r4-integer` `load_client_artifact` + `bind_embedded` + `P4Session::step` path and verify **`0` raw-word diffs** across **all `96` prompts** (`>= 64` bar) in `cuda_parity_96.json` on every step and every carried field (`token`, `bucket`, `highest_order`, `pick_*`, `s3_*`, `s2_skip_*`, `cache_hit_state`, `s4_role_*`, `mass_q48`, `top_ties`, `mass_sha256`).
  2. Additionally spawn the compiled release `uor-chat --discrete /tmp/p4-m1-eval/p4.d11 --prompt-ids ... --p1-trace --max-tokens <steps>` binary end-to-end across a stratified **`8`-prompt CLI subset** (`short_00`, `med_00`, `long_00`, `ctx2048_00` `slide4`, `noread_00` `--read-mode no_read`, `edge_bos_only`, `edge_bosless`, `edge_repeat_lcp_pick0`; configurable via `--cli-count`) and assert **`0` raw-word & JSON diffs** between Wick RTX 4090 CUDA and the live `uor-chat` CLI output.
- **Memory & Throughput benchmark (`2,048`-token context):**
  - Run `3` fresh `/usr/bin/time -l` untraced invocations (`UOR_P1_TIMING=1`, no `--p1-trace`) on a `2,048`-token prompt with `--max-tokens 128` and `--max-tokens 256` (both `read_enabled = true` and `false` diagnostic):
    - **Resident bar:** `/usr/bin/time -l` `maximum resident set size <= 4,000,000,000 B` (also report `peak memory footprint`).
    - **Throughput bar:** warm untraced `decode_ns` throughput `>= 1,000 tok/s` at `2,048`-token context (cold `open_and_prefill_ns` reported candidly alongside warm `tok/s`).
  - If M1 RSS exceeds `4.00 GB` or warm `2,048`-context speed is below `1,000 tok/s`, any pure-integer optimization in `discrete_d11.rs` / `discrete_count.rs` / `p4_serving.rs` (e.g., dropping unused D11 trunk sparse tables after `identity_pick_codes()` extraction on `P4` load, or hoisting invariant divisions) must keep `0` bit diffs across all `96` Wick CUDA prompts and re-pass all host/WASM/zero-float/zero-matmul audits.

---

## 4. Bars, Controls & Verdict Map

| Gate | Bar | Pass | Fail |
|---|---|---|---|
| **P4-Size** | `size(p4.d11) + size(tokenizer.json) <= 4,000,000,000 B` | `<= 4,000,000,000 B` | `> 4,000,000,000 B` |
| **P4-RSS** | Apple M1 `/usr/bin/time -l` peak RSS `<= 4,000,000,000 B` | `<= 4,000,000,000 B` | `> 4,000,000,000 B` |
| **P4-Speed** | Apple M1 warm untraced decode at `2,048` ctx `>= 1,000 tok/s` | median `>= 1,000 tok/s` (cold reported) | median `< 1,000 tok/s` |
| **P4-Parity** | `>= 64` (`96`) prompts, `0` raw-word & `mass_sha256` diffs Wick CUDA vs M1 `uor-chat` | `0` diffs across all `96` prompts | `>= 1` diff (halts) |
| **Bar A (M4 PRLL)** | `chat-v0-p2` `heldout.u16` (`512`w stream basis `2.837427 B/tok`) `hi <= 0.90` BPB | `hi <= 900,000` micro-BPB (`0.90`) | `lo > 900,000` micro-BPB (`0.90`) (closes frozen `P4` config on `chat-v0-p2`, never mechanism) |

---

## 5. Cost & Stop Conditions

- **Compute cost:** `$0.00` pod spend (Wick RTX 4090 local host + local Apple M1 laptop; `0` rented cloud pods). Expected Wick GPU wall time `< 3` minutes total; Apple M1 benchmark `< 5` minutes total.
- **Hard stop conditions:**
  1. Any input SHA-256 mismatch or missing file.
  2. Any non-zero PTX/SASS float or MMA instruction in Gate 1.
  3. Any non-zero GPU-vs-CPU diff in Gate 2 (`finite`), Gate 3 (`96` prompts), Gate 4 (`>= 512` Bar A CPU spot checks or determinism repeat), or Gate 5 (Apple M1 `uor-chat`).
  4. Any `PRLL` mass `< 1` or sum `!= 2^48`, or `p4.d11` cold-reload validation failure.
  5. A negative on Bar A (e.g., `lo > 0.90` BPB on `chat-v0-p2`) is recorded honestly as a configuration result (`P4` trained on `D87` vs `chat-v0-p2` held-out), never hidden or retuned, and advances to Phase `P5` (`DECISION.md:335-346`).
