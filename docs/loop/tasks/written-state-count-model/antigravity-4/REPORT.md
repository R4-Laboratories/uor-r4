# REPORT: written-state-count-model, antigravity-4 (`UORKNC03` Stage B1a)

Lab: antigravity. Item: [`written-state-count-model`](../../../roadmap/items/written-state-count-model.md). Order: [`ORDER.md`](ORDER.md) (`0324c40601c10f64fb615f950f0d3c67d1cd65ceab610a90db2505f4e3133598`). Sheet: [`SHEET.md`](SHEET.md) (`e48b3ba64c1be2429b1d657fdf5993e7bd6ebbfedc71adfee2f3733266d31a9e`). Non-author FIT: [`FIT.md`](FIT.md) (`7853715c6838c19f69d65f1a380990c05ceb5c5aba3bb55d6209750598a127e5`, `TEST FITNESS: FIT`). Pre-run & GO: [`PRE-RUN.md`](PRE-RUN.md), [`GO.json`](GO.json). Evidence: [`evidence/receipts.sha256`](evidence/receipts.sha256).

## What did not work, first

- **Round 1 FIT review returned `UNFIT` before Wick dispatch (`antigravity`, non-author R1):** Flagged three defects plus four hygiene items: (1) `cuda-v3` probe density (`ord % 4093 == 197`) reached at most `<= 9,406` probes on `R` and `<= 6,580` on `DEV-tune`, below the `>= 12,500` bar; (2) `seed1.d11` serving tokenizer (`a9fb13f7...`) needed pinning separately from `DEV-tune` census tokenizer (`d36d3e87...`), and `antigravity-4` lacked a `UORKNC03` deployment packer; (3) `compress-v3` lacked `verify-v2` row-by-row parity and a pinned `c1.tokens.u16` finite UORT slice; plus decimal `<= 3,906,250 KiB` (`4.000 GB`) RSS bound, `TMPDIR` on `/mnt/plexpool`, `deg` bounds checks in `write_v3`, `eval-v3/src/main.rs` digest binding, and `// SAFETY:` comments. All items were resolved (`cuda-v3` `ord % 1009 == 197` + explicit `12,500` floor, `compress-v3 verify-v2`/`slice-uort`, `pack-v3`, updated `SHEET.md`) and passed Round 2 as `TEST FITNESS: FIT`.
- **Initial Wick build exited `101` on `uor-r4-discrete/build.rs`:** `UOR_NVCC_ARCH=compute_89` was unset in the first remote shell; exporting `UOR_NVCC_ARCH=compute_89` built all binaries cleanly.
- **Initial Gate 2 `cuda-v3 verify-all` rejected ordinal `0` (`cuda-v3/src/main.rs`):** An off-by-one check (`i >= at` instead of `i > at` on unused-order rows) falsely rejected active order `1` (`[BOS]`) at `at = 1`. Changing `i >= at` to `i > at` (`cuda-v3` binary `1d7170d2...` -> `123df397...`, unchanged `lookup_v3.sm_89.cubin` `4c329f9f...`) resolved it; Gate 2 and Gate 4 then passed with `0` divergences across `175`, `20,821`, and `14,313` native probes.
- **Initial Gate 4 script exited `127` after `CMP_PASS`:** Escaped quotes on the CPU `verify` line outside `bash -c` caused a shell error after both `R` and `DEV-tune` `65536` vs `65521` `cmp` checks had passed; re-running CPU `cuda-v3 verify` cleanly finished Gate 4.
- **`compress-v3` release build warning:** Emits one benign `unused_imports` warning for `PrimeRouteKey` (used under `#[cfg(test)]`).
- **Cold open + prefill latency on `3.891 GB` `UORKNC03` is `14.82–14.92 s`:** Up from `9.80 s` on `T=318` `UORKNC02` (`antigravity-3`) because `UORKNC03` validates `1,112,973,423` retained followers across `216,215,852` contexts (`3.28×` more entries than `codex-37`'s `UORKNC02` `T=318` `339,293,670` followers with order-3 unpruned, and `5.99×` more than `UORKNC03` `uniform (318,318)`'s `185,809,857`). Warm traced CPU 128-token decode runs at `2,036–2,077 tok/s`, and greedy prose on a `count + Pick` base remains degenerate (`We need answer...`) because M2/M4 dialogue alignment and `P2`/`P3` state updates are separate milestones.
- **Scope boundary (`DEV-tune` only; `DEV-test` untouched):** This unit evaluated frozen fits only on `DEV-tune` (`1,400,045` micro-BPB). Sealed `DEV-test` was NOT opened or scored; `P1(b)` qualification awaits its own pre-registered Stage B1b single-shot `DEV-test` run.

## What changed

- **Pure-integer lossless reversed-suffix trie `UORKNC03` codec (`#![forbid(unsafe_code)]`) [antigravity]:**
  - [`crates/uor-r4-discrete/src/kn_ceiling/runtime/compact.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/compact.rs) & [`rank.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/runtime/rank.rs): Added `MAGIC_V3 = b"UORKNC03"`, `CompactV3Order`, `LookupOrders::{V2, V3}`, 15-code `INLINE_PAIRS`, stride-64 `parent_sb`/`fol_sb`/`esc_sb`, 12-bit packed `child_toks`, `esc16`/`esc32` escapes, incremental reversed-trie `lookup`, `visit_counts`, and `visit_next_masses` with fail-closed validation.
  - [`crates/uor-r4-discrete/src/kn_ceiling/compact_lookup.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/compact_lookup.rs), [`replay.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/compact_lookup/replay.rs), [`kn_ceiling.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling.rs), and [`ranking_tests.rs`](../../../../../crates/uor-r4-discrete/src/kn_ceiling/compact_lookup/ranking_tests.rs): Added `encode_compact_lookup_v3`, updated `canonical_digests` and `device_replay_tables` for `V3`, and added `V2 == V3` parity + escape (`>= 255`, `>= 256`, `> 65535`) + 5 planted-refusal tests. Existing `uor-r4-integer/src/discrete_count.rs` (`P1::load`, `uor-chat`) dispatches `UORKNC03` natively via `CompactCountLookup::from_bytes`.
  - [`crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs`](../../../../../crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs): Added `embedded_uorknc03_deployment_streaming_roundtrip_parity_and_refusal` under `#[cfg(feature = "p1-count-serving")]`.
- **Stage B1a tooling under `docs/loop/tasks/written-state-count-model/antigravity-4/` [antigravity]:**
  - [`compress-v3/`](compress-v3/src/main.rs): Streaming `UORKNC02 -> UORKNC03` `histogram`, `write` (with cold reload validation), `verify-v2`, and deterministic `slice-uort`.
  - [`cuda-v3/`](cuda-v3/src/main.rs): AOT `nvcc -arch=sm_89` `lookup_v3.cuh` + `cuda.rs` emitting `UORCBF01` tapes and CPU `verify` (`>= 12,500` probes).
  - [`eval-v3/`](eval-v3/src/main.rs): Reuses `codex-37` EM (`fit_device.cu`) and forward (`tune_device.cu`) CUDA kernels verbatim, parameterized by `--compact`/`--compact-sha` (`<= 3,912,936,331 B`) with hardcoded `DEV-tune` `CENSUS_SHA = 90fbc73a...` (`DEV-test` refused).
  - [`pack-v3/`](pack-v3/src/main.rs): Packs `UORP1A01` + `uor-r4.lut-discrete-count/1` (`write_deployment`) inside `<= 4,000,000,000 B` and verifies streaming readback (`load_client_artifact`).

## Native-space cycle verification (D85)

1. **Single persistent state schema:** Token IDs are 12-bit `UORT` (`0` = BOS, `1` = EOS, `V = 4096`); counts use `UORKNC03` sections `0..=10` (`44 B` header + `88 B` order headers), `KnDiscounts`, `IdentityControlV1` Pick codes, `UORS5W01`/`UORS5W02` Q32 mixture weights, Q48 masses, and Q12 `bits_q12`. Zero floats anywhere in training, serving, CUDA, or WASM (`cuobjdump --dump-sass` = `0` float insns across all 3 `.sass` dumps; serving audit `60/60` clean; WASM audit `7/7` clean).
2. **Shared typed operators:** `CompactCountLookup` on CPU/WASM (`#![forbid(unsafe_code)]`) and `lookup_v3.cuh` (`kn_score_division::divide`) on CUDA `sm_89`.
3. **Bit-identical oracle & transition replay:** `C128` `UORKNC03` matched `UORKNC02` on all `750,315` `visit_counts` rows (`0` diffs), and `cuda-v3 cuda` (`65536` and `65521`) matched `cuda-v3 oracle` on `175/175` finite rows (`0` diffs), `20,821` `R` probes (`0` diffs), `14,313` `DEV-tune` probes (`0` diffs), and `8,493` `eval-v3` mixture probes per arm (`0` diffs).
4. **Pure-integer updates:** Commutative suffix-closed pruning (`T_3 = 2`, `T_4..T_10 = 6`, `T_0..T_2 = 1`) and 96-pass integer EM on `R` (`eval-v3-fit-device`), no F32 master.
5. **Lossless serialization:** Packed `artifact.d11` (`3,891,016,460 B`, `sha256: dd217c0d...`) embeds `seed1.d11` + `UORKNC03` + `A1s` weights/manifest (`3daa02cb...`) and reloads cleanly via `DiscreteCountModel::load_client_artifact`.

## Measurements (`$0.00` pod spend, Wick RTX 4090 & CPU)

### Gate 1 & 2: SASS Zero-Float Audit & Finite `C128` Lossless Gate

| Check | Scope / Input | Measured Result | Receipt |
|---|---|---|---:|
| CUDA SASS (`lookup_v3.sm_89.cubin`) | Cubin `4c329f9f...` | `0` float insns (`sass sha256: 49387e87...`) | [`sass-summary.txt`](evidence/sass-summary.txt) |
| CUDA SASS (`eval-v3-fit-device`) | Binary `c8ca9b53...` | `0` float insns (`sass sha256: decb2947...`) | [`sass-summary.txt`](evidence/sass-summary.txt) |
| CUDA SASS (`eval-v3-tune-device`) | Binary `9d717ca3...` | `0` float insns (`sass sha256: f29c0a76...`) | [`sass-summary.txt`](evidence/sass-summary.txt) |
| `C128` lossless `compress-v3 write` (`T=1`) | `16,459,406 B` (`eea859b5...`) | **`3,879,991 B` (`4.24×`)**, `215 ms`, cold load OK | [`c128-write.json`](evidence/gate2-c128/c128-write.json) |
| `C128` `compress-v3 verify-v2` | `750,315` rows | `divergent_rows: 0`, `76 ms` | [`c128-verify-v2.json`](evidence/gate2-c128/c128-verify-v2.json) |
| `C128` `cuda-v3 oracle == 65536 == 65521` | `175` targets (`finite-c1.uort.u16`) | `cmp` `0` diff, `175/175` verify `divergent_rows: 0` | [`cuda-verify-all.json`](evidence/gate2-c128/cuda-verify-all.json) |
| `eval-v3` finite fit & finite tune | `128` EM rows; `8,193` forward rows/arm | `97` sums / `49,158` words `0` diffs (`527`, `879`) | [`finite-fit.txt`](evidence/gate2-c128/finite-fit.txt), [`finite-tune.json`](evidence/gate2-c128/finite-tune.json) |

### Gate 3: `148 GB` Full-`C` `UORKNC03` Histogram & Compression (`T3*=2, T_hi*=6`)

From [`histogram.json`](evidence/gate3-fullc/histogram.json) (`637.75 s` over `148,096,362,864 B` `compact.counts`, `1afd1c27...`):

| Configuration `(T3, T_hi)` | Output Bytes | Fits `<= 3,912,936,331 B`? | vs `UORKNC02` (`T=318`, `3,912,334,481 B`) |
|---|---:|---:|---:|
| Lossless `(1, 1)` (`6,684,784,357` followers) | `32,864,181,714` | No | `4.51×` smaller than unpruned `148.1 GB` `UORKNC02` |
| `asym_t3_1 (1, 6)` | `3,940,366,658` | No | `+27.4 MB` above compact cap |
| **`asym_t3_2 (2, 6)` (Selected per SHEET §3)** | **`3,804,378,885`** | **Yes** (`-108.6 MB`) | **`1,112,973,423` followers (`3.28×` `codex-37` `339,293,670`; `5.99×` `uniform (318,318)` `185,809,857`)** |
| `uniform (6, 6)` | `3,678,693,295` | Yes | `1,081,710,361` followers |
| `uniform (318, 318)` | `600,816,188` | Yes | **`6.51×` smaller than `UORKNC02` `T=318`** |

- **`full-c.v3` write receipt ([`full-c-write.json`](evidence/gate3-fullc/full-c-write.json)):** `3,804,378,885 B` (exact byte match to `histogram.json`), `sha256: 75ccb234e669cb6b707ffe4091822a91a225e0875a0e99edf9f21f8f2ec3f233`, `406.73 s`, `verified_cold_load: true`, `1,112,973,423` retained followers across `216,215,852` contexts (`1` order-0 context / `4,079` followers -> `20,535,585` order-10 contexts; `3.42 B/entry`).

### Gate 4: Reserved-`R` & `DEV-tune` `cuda-v3` Feature Captures

| Split (`rows`) | Batch `65536` | Batch `65521` | `cmp` Parity | Native Probes | Tape SHA-256 |
|---|---|---|---:|---:|---|
| Reserved `R` (`15,947,584`) | `21.97 s`, `239,816 KiB` RSS | `22.09 s`, `241,268 KiB` RSS | Byte-identical | **`20,821` (`0` diffs)** | [`8b6b39d2...`](evidence/gate4-tapes/r-verify.json) |
| `DEV-tune` (`10,807,417`) | `16.37 s`, `230,080 KiB` RSS | `16.30 s`, `230,892 KiB` RSS | Byte-identical | **`14,313` (`0` diffs)** | [`dc0785bf...`](evidence/gate4-tapes/tune-verify.json) |

### Gate 5: Reserved-`R` S5 EM Fit & Frozen `DEV-tune` Selection (`DEV-test` Untouched)

- **`R` fit (`15,947,584` targets, `96` passes, `677` replay probes/arm, `128` closed gates rejected):** `a0` final Q12 = `195,226,035,976` (`9ec8168a...`), `a1` = `184,961,598,291` (`a944d018...`, `33` fired hits), `a1s` = `184,956,221,990` (`e2fbf41d...`, `33` fired hits).
- **Frozen `DEV-tune` (`2,670` records, `10,807,417` targets, `21,928,475` all bytes / `15,995,822` novel bytes, `8,493` native probes/arm, `0` diffs) ([`selection.json`](evidence/gate5-eval/tune/selection.json)):**

| Arm / Baseline | All micro-BPB | Novel micro-BPB | Response | Novel Response |
|---|---:|---:|---:|---:|
| `codex-37` `A1s` (`UORKNC02` `T=318`) | `1,581,444` | `1,826,832` | `1,684,813` | `1,830,021` |
| `claude-422` canonical `R*` (`N=10` on `DEV-tune`) | `1,432,318` | `1,747,150` | — | — |
| **`antigravity-4` `a0` (pure count `UORKNC03`)** | `1,475,608` | `1,784,338` | `1,588,266` | `1,787,207` |
| **`antigravity-4` `a1` (3-class Pick `UORKNC03`)** | `1,400,067` | `1,684,574` | `1,506,747` | `1,687,944` |
| **`antigravity-4` `a1s` (5-class `UORS5W02`, SELECTED)** | **`1,400,045`** | **`1,684,543`** | **`1,506,724`** | **`1,687,913`** |

- **Paired bootstrap (`1,000` resamples, seed `4390`):** `a1s_minus_a1` = **`[−31, −13]` micro-BPB all** (`[−2,709,857, −1,217,573]` Q12, `strictly_below_zero: true`) and **`[−43, −20]` novel**; `a1s_minus_a0` = `[−81,216, −69,945]` all / `[−107,310, −92,232]` novel.
- **`DEV-tune` summary:** `a1s` (`1,400,045`) improves on `codex-37` (`1,581,444`) by **`−181,399` micro-BPB** and reaches **`0.9775 × R*`** (`−32,273` all / `−62,607` novel micro-BPB below canonical unpruned `R*` on `DEV-tune`).

### Gate 6: Packed `uor-r4.lut-discrete-count/1` & Streaming CPU `uor-chat`

- **Packed deployment ([`pack.json`](evidence/gate6-chat/pack.json)):** `deploy_bytes = 3,891,016,460 B` (`deploy_sha256: dd217c0de241f725925e62fdcecab2a78cfc4829e080200e2de708a1e5c57d05`, `state_sha256: 3daa02cb4038389fb3977b25a61d7a065ffbbe8e912840e9b38e5aca70697cae`, `under_4gb_decimal: true`).
- **Dual `/usr/bin/time -v` CPU streaming runs (`CUDA_VISIBLE_DEVICES=""`, `128` traced tokens, `temperature = 0`):**
  - Run 1 ([`chat1.time`](evidence/gate6-chat/chat1.time), [`chat1.stderr`](evidence/gate6-chat/chat1.stderr)): **`3,791,956 KiB` (`3,882,962,944 B`) peak RSS**, wall `15.09 s` (`open_and_prefill_ns = 14,922,027,407`, `decode_ns = 62,866,556` = **`2,036.06 tok/s`**).
  - Run 2 ([`chat2.time`](evidence/gate6-chat/chat2.time), [`chat2.stderr`](evidence/gate6-chat/chat2.stderr)): **`3,791,064 KiB` (`3,882,049,536 B`) peak RSS**, wall `14.97 s` (`open_and_prefill_ns = 14,820,607,589`, `decode_ns = 61,613,134` = **`2,077.48 tok/s`**).
  - `cmp`: byte-identical `chat1.stdout == chat2.stdout`; both RSS measurements clear strict decimal `<= 4,000,000,000 B` (`<= 3,906,250 KiB`) by `> 117 MB`.

## Checks run at the head

- `cargo test --locked -p uor-r4-discrete --lib kn_ceiling` — **29 passed, 0 failed**.
- `cargo test --locked -p uor-r4-count-runtime` — **passed**.
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --test discrete_sparse_table_parity --test discrete_chat_cli` — **6 passed, 0 failed** (`3` ignored benchmarks) + **4 passed, 0 failed**.
- `cargo test --manifest-path docs/loop/tasks/written-state-count-model/antigravity-4/{compress-v3,cuda-v3,eval-v3,pack-v3}/Cargo.toml` — **1 + 1 + 7 + 1 passed, 0 failed**.
- `RUSTFLAGS="-C target-feature=-bulk-memory" cargo build --release --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --lib --target wasm32-unknown-unknown` + `python3 scripts/audit_zero_matmul_wasm.py --serving target/wasm32-unknown-unknown/release/uor_r4_integer.wasm` (`e8b8fff1...`, `1,348,549 B`) — **7/7 clean (`float=0`, `unclassified=0`)**.
- `python3 scripts/audit_zero_matmul_serving.py` (`libuor_r4_integer.rlib`) — **FULL PASS (`60` symbol ranges, `0` violations)**.
- Wick Gates 1–6 verified under [`evidence/receipts.sha256`](evidence/receipts.sha256); `scripts/pod/uor-pod status` (`0` running pods, `$0.00` spent).
- `uor-harness check --staged`, `check --only harness.native_space_contract`, `ratchet d23|d25|d32`, `claim-wording`, `loop report --check`, `loop pr-check --local` — **PASS**.

### Post-FIT Source Provenance (`FIT.md` vs `HEAD`)

Five of the 23 files audited in historical pre-dispatch [`FIT.md`](FIT.md) (`7853715c...`) changed SHA-256 before `HEAD`; historical `FIT.md` / `PRE-RUN.md` / `GO.json` remain unmodified, and each delta was verified directly against Wick source (`20261125-antigravity-uorknc03/src/`):

- `docs/loop/tasks/written-state-count-model/antigravity-4/cuda-v3/src/main.rs`: `6166dbd0...` -> `d9e160c2...` (Gate 2 `i >= at` -> `i > at` off-by-one fix; exact SHA-256 compiled into Wick `cuda-v3` `123df397...` for Gates 2 & 4; `lookup_v3.sm_89.cubin` unchanged `4c329f9f...`).
- `crates/uor-r4-discrete/src/kn_ceiling/compact_lookup.rs`: `d8556645...` -> `16b1c2d0...` (pure `rustfmt --edition 2021` whitespace only; `diff -u <(rustfmt Wick_copy) HEAD` = `0` lines).
- `crates/uor-r4-discrete/src/kn_ceiling/runtime/compact.rs`: `fe2d1f9e...` -> `a8deafe8...` (pure `rustfmt --edition 2021` whitespace only; `diff -u <(rustfmt Wick_copy) HEAD` = `0` lines).
- `crates/uor-r4-discrete/src/kn_ceiling.rs`: `04701f61...` -> `9a2aa252...` (pure `rustfmt --edition 2021` whitespace only; `diff -u <(rustfmt Wick_copy) HEAD` = `0` lines).
- `crates/uor-r4-discrete/src/kn_ceiling/compact_lookup/ranking_tests.rs`: `29609140...` -> `6934b404...` (`rustfmt --edition 2021` + test-only 2-line D32 tuple-destructuring rewrite `chunk[0].0` -> `let (key, _, _, _) = chunk[0];` at lines 176–177; zero runtime code changed; re-tested locally with `29/29` pass).

## Skills used

- `native-space-cycle`

## Still open

- **Stage B1b (`P1(b)` on sealed `DEV-test`):** Separate pre-registered single-shot evaluation of frozen `artifact.d11` (`dd217c0d...`: `full-c.v3` `75ccb234...` + `a1s` `e2fbf41d...`) against canonical `R*` (`<= 1.03 × R*`).
- **Stage B2–B4 (`P2`, `P3`, Studio WASM SIMD128):** E8 snap + Starling neighbour miss (`P2`), Brier error-write update (`P3`), and Studio WASM SIMD128 acceleration (`B4`).

## Plain summary

`UORKNC03` replaces flat `12 B/entry` `UORKNC02` context tables with a pure-integer lossless reversed-suffix trie codec (`crates/uor-r4-discrete/src/kn_ceiling/runtime/{compact.rs,rank.rs}`), cutting storage ~`4.2×–6.5×` with bit-identical KN interpolation across CPU and AOT `sm_89` CUDA (`0` SASS floats). Applied to Wick's `148,096,362,864 B` full-`C` `UORKNC02` under the pre-registered `(T3*=2, T_hi*=6)` rule, `full-c.v3` fits in `3,804,378,885 B` (`3.42 B/entry` across `1.113B` / `1,112,973,423` retained followers — `3.28×` more than `codex-37`'s `339.3M` and `5.99×` `UORKNC03` `uniform (318,318)`), driving frozen `DEV-tune` `A1s` down to **`1,400,045` micro-BPB all / `1,684,543` novel** (beating `codex-37`'s `1,581,444` by `181,399` micro-BPB and beating canonical unpruned `R*` on `DEV-tune`) while serving cleanly in CPU `uor-chat` at **`3,791,956` / `3,791,064 KiB` (`3.88 GB` decimal `<= 4.000 GB`)** and `~2,036–2,077 tok/s`. Sealed `DEV-test` was untouched; Stage B1b is next.
