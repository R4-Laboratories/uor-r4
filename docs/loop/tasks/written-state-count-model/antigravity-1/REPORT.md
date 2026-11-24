# REPORT: written-state-count-model, antigravity-1

Lab: antigravity. Order: [ORDER.md](ORDER.md).

## What did not work, first

- **Scope and host limits:** All measurements below ran on Apple Silicon macOS against synthetic `UORLUT01` containers (including a `codex-36`-cardinality `7,164,583`-triple / `4,297,201`-key fixture) with `$0.00` GPU spend and `Running: 0 pod(s)`. No physical Wick `full` / `c1` / `trunk` artifact was re-measured here, and this unit makes no P1(b), M2, or M4 predictive BPB or milestone-closure claim.
- **Unmatched decode scope in `eval_10k_ms`:** `bench_codex36_scale_btreemap_baseline` (`22.209 ms`) timed only `oracle.select_token` (without loading `discrete.table`/`cells`/`routes`/`gates` or running `IntegerDiscreteModel::evaluate_prefix`), whereas `bench_codex36_scale_contiguous_trunk_loader` (`211.625 ms`, `47,253.44 tok/s`) loaded the complete `IntegerDiscreteModel` and executed full Pick + trunk + cell + distance decoding. Only load RSS, load time, and `probe_fold` parity are head-to-head apples-to-apples comparisons (with `BTREEMAP_BASELINE` being a lower bound on legacy `IntegerDiscreteModel::load`).
- **Remaining file-buffer overlap during `IntegerDiscreteModel::load`:** Replacing `BTreeMap` + transient `Vec<u32>` section buffers reduces sparse trunk memory by `> 481 MB` peak RSS, leaving `71,645,830 B` (`10 B/entry`) for the 12 live sparse tables. However, `IntegerDiscreteModel::load` still reads the raw file via `std::fs::read` before dropping `bytes` on return, so peak load RSS transiently includes one copy of `file_bytes` (`86.5 MB` + `71.6 MB` -> `155.48 MiB` measured RSS here). For `~4 GB` physical artifacts (`codex-36`), eliminating the transient `file_bytes` buffer via streaming/`mmap` remains a separate follow-up unit.
- **Pre-existing Clippy debt in untouched modules:** Running `cargo clippy --locked --no-deps -p uor-r4-integer --features serving-wasm,p1-count-serving --lib --tests -- -D warnings` fails on 82+ pre-existing lints in untouched modules (`codec.rs`, `stack/`, `paired_role.rs`, `ordered_role.rs`, `geometric_vocabulary_actions.rs`; tracked separately under `discrete-clippy-k16-coverage`). Un-flagged Clippy filtered to every line changed or added by this PR shows **0 diagnostics** in `discrete_d11.rs` (`L10-11`, `L30-35`, `L97-101`, `L861-965`, `L1179-1188`, `L2356`, `L2481-2555`, `L5001-5130`) and `discrete_sparse_table_parity.rs`.

## What changed

- [`crates/uor-r4-integer/src/discrete_d11.rs`](../../../../../crates/uor-r4-integer/src/discrete_d11.rs):
  - Replaced `BTreeMap<u32, BTreeMap<u16, u32>>` in `DiscreteSparseCountTable` with contiguous strictly-ascending `(key, token)` Flat SoA arrays (`keys: Box<[u32]>`, `tokens: Box<[u16]>`, `counts: Box<[u32]>`; 10 bytes/entry), keeping `#![forbid(unsafe_code)]`.
  - Added `DiscreteSparseCountTable::from_triple_bytes(&[u8], vocab)` to stream little-endian `[key, tok, cnt]` triples straight from aligned `UORLUT01` section bytes without intermediate `Vec<u32>` section copies, preserving single-pass validation on sorted `TrunkTables::to_triples` streams plus an exact `(key, tok)` sort-and-dedup fallback for unsorted inputs.
  - Added O(log N) binary-search `get(key, tok)` and `partition_point` slice `tokens_for_key(key) -> &[u16]`; updated `DiscreteTrunkTables::select_level3_joint_pointer_global` and `IntegerDiscreteModel::from_bytes` (`read_u32_bytes`); removed the unused `BTreeMap` import; added 2 focused unit tests (`sparse_table_flat_soa_matches_btreemap_oracle_on_sorted_and_unsorted_inputs`, `sparse_table_flat_soa_rejects_all_planted_faults`).
- [`crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs`](../../../../../crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs):
  - Added raw-word parity verification against an independent `BTreeMap` reference oracle across single-step queries, active unigram tie-break (`&[4] -> 11`), token-ID tie-break (`&[5] -> 6`), missing-key fallback (`&[7] -> 14`), `pointer_succ` interactions, sorted vs shuffled triples, 4-token autoregressive rollouts, all `60` planted-fault rejections (`12` sparse tables `0..12` × `5` fault classes), and planted count sensitivity (`tok 4 -> 2`).
  - Added separate-process `codex-36`-cardinality generator (`bench_generate_codex36_scale_fixture`) and `/usr/bin/time -l` head-to-head benchmarks (`bench_codex36_scale_btreemap_baseline`, `bench_codex36_scale_contiguous_trunk_loader`).
- [`docs/research/notebook/m4-serving.md`](../../../../research/notebook/m4-serving.md): recorded the `antigravity-1` measurement row.

## Native-space cycle verification (D85)

- **Representation:** Pure integer Flat SoA (`keys: Box<[u32]>`, `tokens: Box<[u16]>`, `counts: Box<[u32]>`, `10 B/entry`), zero float, zero matmul, zero softmax, byte-identical `UORLUT01` sections `7..=18`.
- **Single operator source & backends:** `crates/uor-r4-integer/src/discrete_d11.rs` shared across `uor-r4-integer` native library, `uor-chat` CLI (`target/release/uor-chat`), and `serving-wasm` (`uor_r4_integer_bg.wasm`).
- **Replay fixtures & comparison units:** Unrounded raw integer `prediction`, `top8`, `read_vector`, `admitted_cells`, `trace_words`, and `10,000`-probe fold `probe_fold = 0x00000000004a90b1` compared between the legacy `BTreeMap` oracle and contiguous `IntegerDiscreteModel` across sorted and shuffled inputs.

## Measurements (`codex-36` exact sparse cardinality)

Synthetic fixture matching `codex-36` sparse cardinality (`12` sections, `7,164,583` triples, `4,297,201` distinct outer keys, `85,974,996 B` sparse payload, `86,535,904 B` container, `sha256: b5daf74dce68ed8b2f9ecf3fdabf8fdb38f82343bafac0d6ea455044c5757330`), measured via `/usr/bin/time -l` in separate release processes on Apple Silicon macOS (fixture deleted after test):

| Metric | Legacy `BTreeMap` baseline (lower bound) | Contiguous Flat SoA `IntegerDiscreteModel::load` | Delta |
|---|---:|---:|---|
| Maximum resident set size (`RSS`) | `644,071,424 B` (`614.23 MiB`) | `163,037,184 B` (`155.48 MiB`) | **`-481,034,240 B` (`-74.7%`, `3.95x` lower)** |
| Peak memory footprint | `686,834,528 B` (`655.02 MiB`) | `161,661,480 B` (`154.17 MiB`) | **`-525,173,048 B` (`-76.5%`, `4.25x` lower)** |
| Load wall time | `503.623 ms` | `282.331 ms` | **`-221.292 ms` (`-43.9%`, `1.78x` faster)** |
| Page reclaims | `50,883` | `10,122` | `-40,761` (`-80.1%`) |
| Instructions retired | `7,920,074,520` | `5,883,557,528` | `-25.7%` |
| `10,000`-probe `probe_fold` | `0x00000000004a90b1` | `0x00000000004a90b1` | **Bit-identical** (`47,253.44 tok/s` full `evaluate_prefix`) |

## Checks run at the head

- `rustfmt --check crates/uor-r4-integer/src/discrete_d11.rs crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs` — PASS.
- `cargo test --locked -p uor-r4-integer --lib discrete_d11` — **28 passed, 0 failed**.
- `cargo test --locked -p uor-r4-integer --test discrete_sparse_table_parity` — **3 passed, 0 failed** (`3` ignored scale benchmarks run explicitly above).
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --lib` — **734 passed, 0 failed**, `1` ignored.
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --test discrete_chat_cli` — **4 passed, 0 failed**.
- `cargo test --locked -p uor-r4-integer --features serving-wasm,p1-count-serving --test discrete_sparse_table_parity` — **3 passed, 0 failed** (`3` ignored).
- `python3 scripts/audit_zero_matmul_serving.py target/release/libuor_r4_integer.rlib --tap` (`60` symbol ranges) — **5/5 TAP ok (`0` violations)** (`sha256: ceef8276fa66ab090ffdea6b745e770f4ff1d51ef178541e2b70e7a85f0c1d9d`).
- `python3 scripts/audit_zero_matmul_serving.py target/release/uor-chat --tap` (`55` symbol ranges, transitive call graph `111` functions) — **5/5 TAP ok (`0` violations)** (`sha256: 0bd5c9125f7db20519378d7bbb9e9cdfc282b28f50c06ada55b4153e51a225b4`).
- `RUSTFLAGS="-C target-feature=-simd128" wasm-pack build crates/uor-r4-integer --target web --no-opt` (built from identical `discrete_d11.rs` to temp dir outside git, deleted) + `python3 scripts/audit_zero_matmul_wasm.py .../uor_r4_integer_bg.wasm --serving --call-graph --tap` (`1,293,870 B`, `sha256: 7ee2e36858e3a9bd327f2051c9c833792b7857e9b7f9fb8407f49e7246bad389`) — **7/7 TAP ok (`float=0`, `unclassified=0`)**.
- `local/harness-target/debug/uor-harness check --staged`, `local/harness-target/debug/uor-harness check --only harness.native_space_contract` (`1 rules, 0 failed`), `check --only prep arena briefing budget harness mistakes prlint receipts` (`0` errors), `ratchet d23/d25/d32` (`4451/93/31`), `loop report --check`, `loop pr-check --base origin/main --head HEAD --local` (`verdict: PASS`), `cargo run -q --locked -p uor-r4-claim-wording -- .`, and `/Users/casey.allard/uor-r4/scripts/pod/uor-pod status` (`Running: 0 pod(s)`) — PASS.

## Skills used

- `native-space-cycle` (recorded in `/Users/casey.allard/.local/share/uor-r4/harness/ledger/antigravity-2026-11.jsonl`).

## Still open

- Re-measure physical Wick `trunk` (`86.5 MB`), `c1` (`329.4 MB`), and `full` (`3.999 GB`) artifacts with `uor-chat` on merged `main` (`codex-36/NEXT.md:18`).
- Address transient `file_bytes` overlap in `IntegerDiscreteModel::load` (`std::fs::read` on `~4 GB` artifacts) via streaming/`mmap` in a dedicated unit if needed to hold physical `full`-dose client peak RSS under `4.00 GB`.
- Complete PR delivery (`git push`, exact-head `pro` adversarial review, `[antigravity] uor-review v1`, CI green, squash merge, worktree/branch cleanup).

## Plain summary

Loading a trunk D11 artifact previously expanded its twelve sparse lookup tables (`85.97 MB` on disk, `7.16M` triples across `4.30M` outer keys) into nested binary-tree maps and extra per-section copies, consuming over `644 MB` of RAM even before opening full-dose count tables. This change streams those aligned section bytes straight into three parallel contiguous integer arrays (`keys`, `tokens`, `counts`) and searches them directly in memory. On a synthetic trunk fixture matching the exact `codex-36` sparse cardinality on Apple Silicon macOS, peak RSS dropped from `644.1 MB` to `163.0 MB` (`3.95x` smaller), peak memory footprint fell `76.5%`, load time dropped `43.9%`, all `10,000` test probes stayed bit-identical, and both native and WebAssembly static serving audits remained clean.

