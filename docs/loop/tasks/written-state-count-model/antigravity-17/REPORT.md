# REPORT: written-state-count-model, antigravity-17

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P4` + Bar A `chat-v0-p2` PRLL). Base: `d13c035dd` (PR `#4301`). Pre-registration & FIT: [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (`TEST FITNESS: FIT` at `0c9cbee6f`, `GO_SHA = 50f06caa7`).

## What did not work, first

1. **Bar A (`chat-v0-p2` `heldout.u16`, `e5f400b0...`) PRLL missed the `0.90` BPB bar (`FAIL` — configuration closed, mechanism open):**
   - On the primary `512`-window (`196,608`-target, `context = 384`, `stride = 12,098`) stream basis (`17,576,697 / 6,194,589 = 2.837427` B/tok), frozen `D87`-trained `P4` (`p4_full_errwrite`, `p4.d11` `e0ffbba1...`) scored **`3,353,534,875` Q12 (`20,773,770,247,791,375 / 14,154,626,022,506,496` = `1,467,631` micro-BPB / `1.467631` BPB)** (primary `512`w sample target-byte basis `557,955` B: `1,467,384` micro-BPB); archived 95% CI `[1,439,162, 1,496,100]` (`se_512 = 14,525`), empirical 95% CI `[1,446,291, 1,488,971]` (`se_emp = 10,888`), both strictly above `900,000` micro-BPB (`0.90` BPB). No `1,024`/`2,048` extension was triggered because the `512`-window interval did not straddle `0.90`.
   - Secondary `64`-window diagnostic (`24,576` targets, `stride = 96,784`): `426,353,755` Q12 = **`1,492,706` micro-BPB** (`1.492706` BPB) on stream basis (`68,983` B sample target-byte basis: `1,508,926` micro-BPB); primary `512`w non-BOS diagnostic (`196,423` targets): `1,453,412` micro-BPB PRLL (`1,453,414` raw `P4`; the `185` `BOS = 0` targets receive lawful PRLL `mass = 2` [`1` base unit + `1` largest-remainder unit] = `47` bits = `35,614,720` Q12, adding `+14,219` micro-BPB).
   - **Mechanism vs instrument classification (§4.2):** Zero instrument error (`0` PTX/SASS float/MMA ops, `0` determinism repeat diffs across `196,608` targets, `0 / 696` CPU spot-check diffs including all `185` `BOS = 0` targets and `448` `P4Session` cross-checks, uniform floor exact at `4,229,183` micro-BPB). This is a genuine model-capacity / domain result: a frozen `D87` count+cache model without a trained geometric core beats `P1(A1s)` (`1,499,106` micro-BPB, `-31,475` micro-BPB / `-2.10%`) on `chat-v0-p2`, yet cannot reach `0.90` BPB without Phase `P5` (`DECISION.md:335-346`).
2. **Apple M1 cold open/prefill latency & single `no_read` diagnostic throughput:**
   - Cold `open_and_prefill` on the `3.988 GB` `p4.d11` takes median **`61,668.28 ms` (`61.67 s`)** across the `6` `read=on` runs (`61,800.55 ms` across all `7` runs including `no_read`) on Apple M1 (`16 GB` RAM) due to full-file SHA-256 verification and cold `UORKNC03`/`UORS3C01`/`UORS2C01` structural validation on every process launch.
   - The single `--read-mode no_read` diagnostic run at `2,048` context clocked **`769.85 tok/s`** (below `1,000 tok/s`, because its different trajectory `[269, 271, 714, 639, ...]` hit denser `UORKNC03` follower spans), whereas normal `read_enabled = true` serving cleared `>= 1,000 tok/s` cleanly (`1,239.16` and `1,238.05 tok/s`).
3. **Pre-run FIT and Wick `PATH` retries:**
   - Fresh-context non-author FIT Rounds 1 & 2 caught 4 operational/contract gaps (`dispatch.sh` PTX/SASS regexes & cubin pin, `L-002`/`flock`/`sm_89` guards, `bar_a.rs` tokenizer pin, `SHEET.md` Gate 5 wording) and a typo in `pack.rs:25` `PIN_C_SHA` plus `a1s.manifest.json` byte count (`2,523`), all resolved before Round 3 `TEST FITNESS: FIT` (`0c9cbee6f`).
   - Initial Wick SSH invocation failed immediately at `build.rs` because `/usr/local/cuda/bin` was not on non-interactive `PATH`; per `L-002`, the retry ran into a fresh sealed root `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/` with zero overwrites.
4. **Real `96`-prompt `S3` mode coverage:** On real `chat-v0-p2` prompts in `cuda_parity_96.json`, `S3` exercised `primary` (`215` steps) and `backoff` (`5` steps); `starling` and `uniform_miss` were verified `0`-diff on GPU vs CPU in Gate 2 synthetic `finite.json` (`primary: 2`, `backoff: 1`, `starling: 2`, `uniform_miss: 26`).

## What changed

- Added Phase `P4` PR 2 qualification suite under `docs/loop/tasks/written-state-count-model/antigravity-17/`:
  - [`ORDER.md`](ORDER.md), [`SHEET.md`](SHEET.md), [`FIT.md`](FIT.md) (verbatim Round 3 `TEST FITNESS: FIT` at `0c9cbee6f`), and [`dispatch.sh`](dispatch.sh) (`L-002` `OUT_DIR` refusal, `flock -n` Wick GPU lock, `sm_89` & VRAM preflight, `CUDA_DISABLE_PTX_JIT=1`, expanded PTX/SASS float+MMA audit & kernel non-vacuity check, post-run `cubin_sha256` cross-check, `receipts.sha256` sealing).
  - Standalone offline crate [`p4-qual/`](p4-qual/) (`Cargo.toml`, `Cargo.lock`, `build.rs`, `src/{main.rs,cuda.rs,pack.rs,parity.rs,bar_a.rs,p4_kernels.cu}`): pure-integer AOT `sm_89` CUDA kernels (`p4_parity_batch_kernel`, `p4_bar_a_windows_kernel`) reusing `crates/uor-r4-discrete/src/kn_score_division.cuh` and verbatim `crates/uor-r4-integer/src/lawful_likelihood.rs`, with `0` floats, `0` MMA/matmul, `0` normalized exponential, and `#![forbid(unsafe_code)]` on non-CUDA code. Zero edits to `crates/` or root `Cargo.toml`.
  - Committed compact evidence in [`evidence/`](evidence/): [`gate1_audit.json`](evidence/gate1_audit.json), [`finite.json`](evidence/finite.json), [`pack_manifest.json`](evidence/pack_manifest.json), [`parity_fixture_summary.json`](evidence/parity_fixture_summary.json), [`bar_a_report.json`](evidence/bar_a_report.json), [`m1_parity_summary.json`](evidence/m1_parity_summary.json), [`m1_bench.json`](evidence/m1_bench.json), and Wick [`receipts.sha256`](evidence/receipts.sha256).
- Updated [`docs/research/notebook/m4-serving.md`](../../../research/notebook/m4-serving.md) and [`docs/roadmap/items/written-state-count-model.md`](../../../roadmap/items/written-state-count-model.md).

## Measurements

All Wick artifacts & receipts are sealed at `/mnt/plexpool/uor-r4-corpus/runs/20261126-antigravity-p4-qual-r2/` (`receipts.sha256` `9/9 OK`, `$0.00` pod spend, `0` rented pods; `DEV-test` untouched).

### Native-space parity & artifact identity (D85)
- **Representation:** Original 12-bit `UORT` tokens (`V = 4,096`, `BOS = 0`, `EOS = 1`), pure-integer `UORKNC03` + `UORS3C01` + `UORS2C01` + `UORS4P01` + `UORS6B01` (`22 × 2,640` Q32 simplex) + `CausalSessionCache` (`0` B disk), packed losslessly into `uor-r4.lut-discrete-p4/1` (`UORP4A01`).
- **Backends compared:** Wick RTX 4090 AOT `sm_89` CUDA (`cubin_sha256 = 7b12da42688cdd704ca042d1a0d928683ca21178eca124e011b2f58afeffaa74`, `0` PTX/SASS float ops, `0` MMA ops) vs Wick x86_64 CPU vs local Apple M1 (`aarch64-apple-darwin`, `16 GB` RAM) release `uor-chat` & `p4-qual`.
- **Sealed `p4.d11` (`artifact_sha256 = e0ffbba19deb34df085d002c1062dd43e88adf98a2aac6aa81d469ab98507776`, `state_sha256 = 368033364f68ec16523735adf71b036b56902c5a9e9a816e6146a85a2874e778`):**
  - `p4.d11`: `3,987,940,843 B` (`trunk 86,536,132`, `counts_v3 3,804,378,885`, `s3_compact 55,418,715`, `s2_skip_compact 39,088,029`, `s4_table 1,952,698`, `s6_weights 464,704`, `a1s_weights 98,604`, `fit_manifest 2,523`).
  - `tokenizer.json` (`a9fb13f7...`): `265,393 B`.
  - **Total deployed size:** **`3,988,206,236 B <= 4,000,000,000 B`** (`11.79 MB` under cap) — **`P4-Size PASS`**.

### Parity results (`0` diffs across raw integer words and `mass_sha256`)
| Gate / Suite | Scope | Raw words / positions checked | Diffs | Verdict |
|---|---|---:|---:|---|
| Gate 1 (`gate1_audit.json`) | AOT `sm_89` PTX + SASS (`p4_parity_batch_kernel`, `p4_bar_a_windows_kernel`) | `2` kernels | `0` float / `0` MMA | **PASS** |
| Gate 2 (`finite.json`) | Synthetic `V = 4,096` `P4State` (`10` prompts, `31` steps, all `4` `S3` modes, `256` PRLL pos incl. `2` BOS) | `126,945` score words + `256` PRLL | **`0`** | **PASS** |
| Gate 3 (`parity_fixture_summary.json`) | Wick CUDA vs Wick CPU on `96` prompts (`220` steps, `cuda_parity_96.json` `735cc0af...`) | `900,900` score words + `220` full traces | **`0`** | **PASS** |
| Gate 5a (`m1_parity_summary.json`) | Wick CUDA vs Apple M1 (`aarch64`) in-proc (`96` prompts / `220` steps) + live `uor-chat` CLI (`8` prompts / `22` steps) | `96` prompts (`220` steps + `22` CLI steps) | **`0`** | **PASS** |
| Gate 4 spot + repeat (`bar_a_report.json`) | Bar A determinism repeat (`196,608` targets) + CPU spot checks (`696` pos incl. `185` BOS & `448` `P4Session`) | `196,608` + `696` | **`0`** | **PASS** |

### Apple M1 (`16 GB` RAM) resident memory & `2,048`-context throughput (`m1_bench.json`)
Prompt `ctx2048_00` (`2,048` tokens, untraced `UOR_P1_TIMING=1`, `/usr/bin/time -l`, release `uor-chat --features p1-count-serving`):

| Config | Runs | Median warm `tok/s` | Cold `open_and_prefill` (ms) | Worst `/usr/bin/time -l` RSS (B) | Worst footprint (B) | Bar | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| `ctx=2,048`, `128` tok (`read=on`) | `3` | **`1,239.16`** (`1,255.42`, `1,238.40`, `1,239.16`) | `61,072.60` | `3,978,461,184` | `3,985,837,224` | `<= 4.000 GB`, `>= 1,000 tok/s` | **PASS** |
| `ctx=2,048`, `256` tok (`read=on`) | `3` | **`1,238.05`** (`1,075.51`, `1,293.45`, `1,238.05`) | `61,800.55` | `3,978,477,568` | `3,985,886,376` | `<= 4.000 GB`, `>= 1,000 tok/s` | **PASS** |
| `ctx=2,048`, `128` tok (`no_read` diag) | `1` | `769.85` | `65,762.33` | `3,978,493,952` | `3,986,082,984` | diagnostic only | `RSS PASS` |
| **Overall Apple M1 worst / median** | **`7` (`6` `read=on`)** | **`1,239.16` / `1,238.05`** | **`61,668.28`** (`read=on`; `7`-run `61,800.55`) | **`3,978,493,952` (`3.978 GB`)** | **`3,986,082,984` (`3.986 GB`)** | **`<= 4.000 GB` & `>= 1,000 tok/s`** | **PASS** |

### Bar A (`chat-v0-p2` `heldout.u16`, `e5f400b0...`, `12,389,242 B`) GPU PRLL (`bar_a_report.json`)
| Arm | Windows / Targets | Exact Q12 bits | Stream-basis fraction (`num / den`) | Stream micro-BPB | Archived 95% CI | Empirical 95% CI | Sample-basis micro-BPB | Verdict vs `0.90` BPB |
|---|---:|---:|---|---:|---|---|---:|---|
| **`P4` primary (`512`w)** | `512` / `196,608` | `3,353,534,875` | `20773770247791375 / 14154626022506496` | **`1,467,631`** | `[1,439,162, 1,496,100]` | `[1,446,291, 1,488,971]` | `1,467,384` | **FAIL** |
| `P4` diagnostic (`64`w) | `64` / `24,576` | `426,353,755` | `2641086280831695 / 1769328252813312` | `1,492,706` | `[1,412,183, 1,573,229]` | `[1,422,834, 1,562,578]` | `1,508,926` | **FAIL** |
| `P1(A1s)` control (`512`w) | `512` / `196,608` | `3,425,454,619` | `21219283502856591 / 14154626022506496` | `1,499,106` | `[1,470,637, 1,527,575]` | `[1,477,660, 1,520,552]` | `1,498,853` | **FAIL** (`P4` beats by `-31,475`) |
| Uniform `U` floor (`512`w) | `512` / `196,608` | `9,663,676,416` | `59862503626113024 / 14154626022506496` | `4,229,183` | exact | exact | — | sanity floor |

## Checks run at the head

- `rustfmt --check --edition 2021 build.rs src/*.rs` in `antigravity-17/p4-qual/`: **PASS**.
- `cargo test --offline` in `antigravity-17/p4-qual/`: **`16/16` passed** (including all `13` verbatim `lawful_likelihood` tests, `128`-byte ABI check, synthetic `P4State` + `PRLL` cross-check, and Bar A integer math pins).
- `cargo clippy --offline --all-targets -- -D warnings` in `antigravity-17/p4-qual/`: **PASS (`0` warnings)**.
- `bash -n docs/loop/tasks/written-state-count-model/antigravity-17/dispatch.sh`: **PASS**.
- `python3 scripts/check_no_softmax.py` and `python3 scripts/check_no_matmul.py`: **PASS**.
- Wick RTX 4090 `dispatch.sh` (Gates 1–4): **PASS** (`gate1_audit` `0` float / `0` MMA, `finite` `0` diffs, `pack` `3,988,206,236 B <= 4 GB`, `parity-fixture` `0 / 900,900` diffs, `bar-a` `0` repeat & `0 / 696` spot diffs, `receipts.sha256` `9/9 OK`).
- Apple M1 Gate 5 (`verify-m1` + `m1_bench`): **PASS** (`0` diffs across `96` prompts + `8` live `uor-chat` CLI invocations; RSS `3,978,493,952 B <= 4 GB`; warm `2,048`-context decode `1,239.16` / `1,238.05 tok/s >= 1,000 tok/s`).
- `uor-harness check --only harness.native_space_contract` and `uor-harness loop pr-check --local`: verified clean at exact PR head.

## Skills used

- `native-space-cycle` (`.agents/skills/native-space-cycle/SKILL.md`)
- `uor-research-architect` (`.agents/skills/uor-research-architect/SKILL.md`)

## Still open

1. **Phase `P5` — scored geometric core (`DECISION.md:335-346`):** Train `<= 1e5` `S1` transport coordinates through `rows-K` (`crates/uor-r4-discrete/src/rowsk_device.rs:56`, `training.rs:3310`) optimizing integer Brier score of mixed mass against an equal-trial re-seeded random-coordinate control (`DEV-test` Brier falls `>= 1% / GPU-day` over the first 2 GPU-days without raising `DEV-test` BPB), needed to close the remaining gap toward `0.90` BPB on `chat-v0-p2`.
2. **Cold open latency on `p4.d11` (`61.67 s` on M1):** Optional future serving polish (e.g., streaming hash/validation optimizations or pre-validated mmap cache) if interactive startup latency becomes a product target; warm decode already exceeds `1,238 tok/s` at `2,048` context.

## Plain summary

Phase `P4` full-dose serving qualification passed all four hardware & parity bars on a real Apple M1 (`16 GB` RAM) and Wick RTX 4090 (`$0.00` pod spend): the lossless packed `p4.d11` + `tokenizer.json` fits in `3.988 GB <= 4.000 GB`, peaks at `3.978 GB` RSS (`3.986 GB` footprint `<= 4.000 GB`) on M1, sustains `1,239 tok/s` warm decode at `2,048`-token context (`61.67 s` cold load), and reproduces Wick RTX 4090 CUDA with `0` bit differences across `96` prompts (`220` steps, `900,900` score words) and `0` PTX/SASS float or MMA ops. On Bar A (`chat-v0-p2` `heldout.u16` PRLL), frozen `D87` `P4` beat `P1(A1s)` (`1.468` vs `1.499` BPB) but honestly failed the `0.90` BPB bar (`[1.439, 1.496]` BPB), confirming the need for Phase `P5`'s trained geometric core.
