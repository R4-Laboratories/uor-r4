# REPORT: written-state-count-model, antigravity-3

Lab: antigravity. Order: [ORDER.md](ORDER.md).

## What did not work, first

- **Default `cargo test -p uor-r4-integer` fails on legacy test target:** Running bare `cargo test --locked -p uor-r4-integer` without `--lib` or `--features serving-wasm` fails compiling pre-existing integration test `tests/adversarial_topological_loops.rs` (which requires `--features serving-wasm`). Library tests (`--lib`) and feature-selected integration tests (`--features serving-wasm,p1-count-serving`) pass cleanly.
- **Native WASM binary target scope & legacy `clippy` lints:** Checking `--target wasm32-unknown-unknown` across `--all-targets` fails in pre-existing binary `src/bin/fbs-toks.rs`; `--lib --target wasm32-unknown-unknown` and `wasm-pack build --target web -- --features serving-wasm` pass cleanly (`float=0`). Running `cargo clippy --tests -- -D warnings` hits 118 pre-existing lints in `src/stack/tests.rs` and `src/geometric_vocabulary_actions.rs`; scoped checks across all four touched files contain zero warnings.
- **Cold open latency remains ~9.8 s on full `3.999 GB` `uor-r4.lut-discrete-count/1`:** Streaming eliminates the duplicate trunk buffer cleanly, bringing Wick `full` peak RSS inside `<= 4.00 GB` (`3,989,266,432 B`), while SHA-256 verification and `CompactCountLookup::from_bytes` validation over the `3.91 GB` `P1` state keep cold load + prefill at `9.80 s` (`open_and_prefill_ns = 9,804,693,807`). Neither P1(b) compressed-S2 (`<= 1.03 × R*`) nor `2k` M1-class decode timing was run in this unit (`$0.00` pod spend).

## What changed

- [`crates/uor-r4-integer/src/discrete_d11.rs`](../../../../../crates/uor-r4-integer/src/discrete_d11.rs): kept `#![forbid(unsafe_code)]` with zero new dependencies; added `HashingReader`, `try_reserve_exact`-guarded `DiscreteSparseCountTable::from_triple_reader`, `FrozenS2Table::from_boxed_bytes`, `IntegerDiscreteModel::from_reader` (`from_reader_with_len`), and updated `IntegerDiscreteModel::load` to stream `UORLUT01` trunk containers (`uor-r4.lut-discrete-trunk/1`, `uor-r4.lut-discrete-trunk-s2/1`, `uor-r4.lut-discrete-trunk-s2/2`) through `BufReader` without allocating the full file byte buffer (falling back for plain/wrapper schemas). Refactored `read_container` onto shared header/base-table/trunk decoders (`parse_and_validate_lut_header`, `validate_section_spec`, `decode_base_lut_sections`, `decode_*`) while leaving `from_bytes` and WASM paths unchanged.
- [`crates/uor-r4-integer/src/discrete_count.rs`](../../../../../crates/uor-r4-integer/src/discrete_count.rs) & [`crates/uor-r4-integer/src/bin/uor-chat.rs`](../../../../../crates/uor-r4-integer/src/bin/uor-chat.rs): added `ClientArtifactFile`, `load_client_artifact`, and `DiscreteCountModel::bind_embedded` so `uor-chat --discrete` under `p1-count-serving` streams the embedded `trunk_bytes` inside `uor-r4.lut-discrete-count/1` directly into `IntegerDiscreteModel::from_reader_with_len` while computing the full container SHA-256, and allocates only exact `header.state_bytes` via `try_reserve_exact` for `P1`.
- [`crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs`](../../../../../crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs) & [`discrete_count.rs`](../../../../../crates/uor-r4-integer/src/discrete_count.rs): added parity tests across sorted trunk, shuffled trunk, plain `uor-r4.lut-discrete/1`, passive/expert `S2`, and embedded vs legacy `load_client_artifact`, plus planted truncation, padding, trailing-byte, digest-mismatch, and sidecar-conflict refusals.
- [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md) & [`docs/roadmap/items/written-state-count-model.md`](../../../../roadmap/items/written-state-count-model.md): recorded macOS synthetic and Wick sealed `codex-36` RSS/parity measurements; archived compact Wick receipts under [`evidence/wick/`](evidence/wick/).

## Native-space cycle verification (D85)

- **Representation & wire compatibility:** Identical `UORLUT01` and `uor-r4.lut-discrete-count/1` wire bytes; same contiguous Flat SoA (`keys: Box<[u32]>`, `tokens: Box<[u16]>`, `counts: Box<[u32]>`, `10 B/entry`), `FrozenS2Table`, `CompactCountLookup`, `FrozenPickMixture`, and `IntegerDiscreteModel` in-memory state. Zero `unsafe`, zero float, zero matmul, zero softmax.
- **Replay & bit-identical parity:** `from_reader` / `load` / `load_client_artifact` + `bind_embedded` match `from_bytes` / `DiscreteCountModel::load` byte-for-byte on `artifact_sha256`, `DeploymentIdentity`, `evaluate_prefix`, and Wick `uor-chat` outputs (`cmp` exit 0 across `trunk` `16` tokens, `c1` `16` traced tokens, `full` `16` traced tokens, and `full` `128` untraced tokens against baseline `/mnt/plexpool/uor-r4-corpus/runs/20261120-count-identity-33/container-client-v2-target/release/uor-chat`, `sha256: 67d33ec2a36fdc8b13f14ef26a5750ed668618bbe8c37280c1e84d31f9ae4f6d`).

## Measurements (`$0.00` pod spend)

### 1. Apple Silicon macOS synthetic `codex-36` sparse cardinality (`/usr/bin/time -l`, release)

Fixture: `12` sparse sections, `7,164,583` triples, `4,297,201` distinct outer keys, `86,535,904 B` container (`sha256: b5daf74dce68ed8b2f9ecf3fdabf8fdb38f82343bafac0d6ea455044c5757330`, deleted after benchmark).

| Metric | Legacy `BTreeMap` baseline | `antigravity-1` (Flat SoA + `fs::read`) | **`antigravity-3` (Flat SoA + `from_reader`)** | Delta vs `antigravity-1` / `BTreeMap` |
|---|---:|---:|---:|---|
| Maximum resident set size (`RSS`) | `644,071,424 B` | `163,037,184 B` | **`77,348,864 B` (`73.77 MiB`)** | **`-52.6%`** vs `#4237` / **`8.33×` lower (`-88.0%`)** vs `BTreeMap` |
| Peak memory footprint | `686,834,528 B` | `161,661,480 B` | **`75,907,576 B` (`72.39 MiB`)** | **`-53.0%`** vs `#4237` / **`9.05×` lower (`-88.9%`)** vs `BTreeMap` |
| Page reclaims | `50,883` | `10,122` | **`4,894`** | `-51.6%` / `-90.4%` |
| `10,000`-probe `probe_fold` (`evaluate_prefix`) | `0x00000000004a90b1` | `0x00000000004a90b1` | **`0x00000000004a90b1`** | **Bit-identical** (`300.330 ms` load, `46,779.16 tok/s`) |

### 2. Sealed `codex-36` client processes on Wick (x86_64 Linux, CPU-only, `/usr/bin/time -v`)

Streaming binary `/mnt/plexpool/uor-r4-corpus/runs/20261124-antigravity-stream-loader/target/release/uor-chat` (`sha256: 45b0fe40dad331dfd3979c20ace7a0c2403983adfbac8ef2a62fcdcc72be6c4a`) vs sealed `codex-36` baseline [`process-memory/`](../../codex-36/evidence/client-load-only.json) (`sha256: 67d33ec2...`), using identical `51`-token prompt and `--temperature 0`.

| Artifact (disk bytes) | Run | Baseline Peak RSS (wall) | **Streaming Peak RSS (wall)** | `cmp` vs Baseline & Peak Delta |
|---|---|---:|---:|---|
| `trunk` (`seed1.d11`, `86,536,132 B`) | `--max-tokens 0` | `790,417,408 B` (`0:00.60`) | **`77,795,328 B`** (`75,972 KiB`, `0:00.15`) | **`-90.2%` (`10.16×` lower)**, `4.0×` faster |
| `trunk` (`seed1.d11`, `86,536,132 B`) | `--max-tokens 16` | — (`0:00.68` on `1` tok) | **`78,061,568 B`** (`76,232 KiB`, `0:00.12`) | `cmp` `0` diff vs baseline `16`-tok output |
| `c1` (`c1-candidate/fixture.d11`, `329,370,095 B`) | `--max-tokens 0` | `1,032,511,488 B` (`0:01.26`) | **`319,344,640 B`** (`311,860 KiB`, `0:00.90`) | **`-69.1%` (`3.23×` lower)** (`-10.0 MB` below file size) |
| `c1` (`c1-candidate/fixture.d11`, `329,370,095 B`) | `--max-tokens 16 --p1-trace` | — | **`319,930,368 B`** (`312,432 KiB`, `0:00.77`) | `cmp` `0` diff on `16`-tok JSON + full trace |
| `full` (`full-candidate/fixture.d11`, `3,998,930,520 B`) | `--max-tokens 0` | `4,702,629,888 B` (`0:10.18`) | **`3,989,266,432 B`** (`3,895,768 KiB`, `0:10.11`) | **`-713,363,456 B` (`-15.2%`) — inside `<= 4.000 GB` (`-10.73 MB`)** |
| `full` (`full-candidate/fixture.d11`, `3,998,930,520 B`) | `--max-tokens 1` | `4,702,720,000 B` (`0:10.24`) | **`3,989,807,104 B`** (`3,896,296 KiB`, `0:09.81`) | **`-712,912,896 B` (`-15.2%`) — inside `<= 4.000 GB`** |
| `full` (`full-candidate/fixture.d11`, `3,998,930,520 B`) | `--max-tokens 16 --p1-trace` | — | **`3,989,540,864 B`** (`3,896,036 KiB`, `0:09.80`) | `cmp` `0` diff (`open_and_prefill = 9.679 s`, `decode = 6.910 ms`) |
| `full` (`full-candidate/fixture.d11`, `3,998,930,520 B`) | `--max-tokens 128` (`UOR_P1_TIMING=1`) | — | **`3,989,716,992 B`** (`3,896,208 KiB`, `0:09.98`) | `cmp` `0` diff (`decode = 54.072 ms`, **`2,367.20 tok/s`**) |

## Checks run at the head

- `rustfmt --check crates/uor-r4-integer/src/discrete_d11.rs crates/uor-r4-integer/src/discrete_count.rs crates/uor-r4-integer/src/bin/uor-chat.rs crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs` — PASS.
- `cargo test --locked -p uor-r4-integer --lib` — **222 passed, 0 failed**.
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --test discrete_sparse_table_parity` — **5 passed, 0 failed** (`3` ignored scale benchmarks; macOS scale benchmark ran explicitly above).
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --test discrete_chat_cli` — **4 passed, 0 failed**.
- `cargo test --locked -p uor-r4-integer --features p1-count-serving --lib load_client_artifact` — **1 passed, 0 failed**.
- `cargo check --locked -p uor-r4-integer --features serving-wasm --lib --target wasm32-unknown-unknown` — PASS.
- `python3 scripts/audit_zero_matmul_serving.py target/release/libuor_r4_integer.rlib --tap` (`60` symbol ranges, `sha256: b7eba3a3...`) — **5/5 TAP ok (`0` violations)**.
- `python3 scripts/audit_zero_matmul_serving.py target/release/uor-chat --tap` (`55` symbol ranges, `111` transitive functions, `sha256: 9d5d3177...`) — **5/5 TAP ok (`0` violations)**.
- `RUSTFLAGS="-C target-feature=-simd128" wasm-pack build crates/uor-r4-integer --target web --no-opt -- --features serving-wasm` + `python3 scripts/audit_zero_matmul_wasm.py .../uor_r4_integer_bg.wasm --serving --call-graph --tap` (`1,285,658 B`, `sha256: ee839506...`) — **7/7 TAP ok (`float=0`, `unclassified=0`)**.
- Wick receipts verified under [`evidence/wick/receipts.sha256`](evidence/wick/receipts.sha256), plus `uor-harness check --staged`, `check --only harness.native_space_contract`, `ratchet d23|d25|d32`, `loop report --check`, `loop pr-check --local`, `uor-r4-claim-wording`, `xtask links`, and `scripts/pod/uor-pod status` (`Running: 0 pod(s)`) — PASS.

## Skills used

- `native-space-cycle`

## Still open

- **P1(b) under `4 GB`:** Compressing `S2` orders `3..10` (`DECISION.md §3.5`, per [`antigravity-2`](../antigravity-2/REPORT.md)) so predictive cost reaches `<= 1.03 × R*` on sealed `DEV-test`.
- **Cold open latency / `2k` context M4 timing:** Reducing cold `~9.8 s` `SHA-256`/validation startup on `3.9 GB` `P1` bundles and measuring `M1`-class `2,048`-token decode throughput.

## Plain summary

Using pure-safe `std::io::Read` streaming under `#![forbid(unsafe_code)]`, `IntegerDiscreteModel::load` and `uor-chat` (`load_client_artifact`) now parse `UORLUT01` trunk tables directly into Flat SoA arrays while hashing the byte stream in flight, removing the transient full-file buffer and the embedded trunk copy inside `uor-r4.lut-discrete-count/1`. On Apple Silicon macOS at `codex-36` cardinality, peak RSS dropped from `163.0 MB` (`antigravity-1`) to **`77.3 MB`** (`8.33×` lower than `BTreeMap`); on Wick's sealed `3.999 GB` `codex-36` `full` client, peak RSS dropped from **`4,702,629,888 B`** to **`3,989,266,432 B`** (`<= 4.000 GB`, `-713.4 MB`), with `cmp`-identical outputs across `trunk`, `c1`, and `full` (`2,367.20 tok/s` untraced 128-token decode).
