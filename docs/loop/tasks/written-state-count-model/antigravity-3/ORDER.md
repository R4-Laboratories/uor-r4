# ORDER: written-state-count-model, antigravity-3

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md). Base: `1aa59e5f0`.

## Goal

Eliminate transient whole-file `std::fs::read` byte buffers when loading `UORLUT01` (`IntegerDiscreteModel::load`) and embedded `uor-r4.p1-count-deployment/1` containers (`uor-chat --features p1-count-serving`) using pure-safe `std::io::Read` streaming under `#![forbid(unsafe_code)]`, keeping the wire bytes identical while bringing full-dose `codex-36` (`3.999 GB`) client peak RSS inside `<= 4,000,000,000 B`.

Follows [`antigravity-1`](../antigravity-1/REPORT.md) (`DiscreteSparseCountTable` Flat SoA) and [`antigravity-2`](../antigravity-2/REPORT.md).

## Files in scope

- `crates/uor-r4-integer/src/discrete_d11.rs`
- `crates/uor-r4-integer/src/discrete_count.rs`
- `crates/uor-r4-integer/src/bin/uor-chat.rs`
- `crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs`
- `docs/loop/tasks/written-state-count-model/antigravity-3/**`
- `docs/research/notebook/m4-serving.md`
- `docs/roadmap/items/written-state-count-model.md`

## Done when

1. **Pure-safe streaming loader:** `DiscreteSparseCountTable::from_triple_reader` and `IntegerDiscreteModel::from_reader` stream `UORLUT01` trunk containers directly via `BufReader` + `Sha256` without allocating the full file byte buffer, and `count_deployment::load_client_artifact` streams the embedded trunk in `uor-r4.p1-count-deployment/1` into `IntegerDiscreteModel::from_reader` before allocating only `header.state_bytes` (`Vec<u8>`) for `P1`. `#![forbid(unsafe_code)]` holds; `from_bytes` and WASM paths remain unchanged.
2. **Bit-identical parity & fault tests:** `cargo test --locked -p uor-r4-integer --features serving-wasm` and `--features p1-count-serving`, `cargo clippy --locked -- -D warnings`, and `wasm32-unknown-unknown` build checks pass; new tests prove bit-identical `IntegerDiscreteModel`, `artifact_sha256`, `DiscreteCountModel`, `evaluate_prefix`, and planted-fault refusals between streaming load and `from_bytes`.
3. **Host & Wick RSS measurements:** MacOS synthetic `codex-36`-cardinality `/usr/bin/time -l` peak RSS vs `antigravity-1` (`163,037,184 B`) AND Wick CPU-only `uor-chat` release measurements on sealed `codex-36` (`trunk`, `c1`, and `full` `3,998,930,520 B`) confirm bit-identical generation/trace and measure whether `full` peak RSS clears `<= 4,000,000,000 B`. `$0.00` pod spend.
4. **Static gates & records:** `uor-harness check --staged`, `harness.native_space_contract`, `uor-harness ratchet d23|d25|d32`, `uor-r4-claim-wording`, `antigravity-3/REPORT.md`, `m4-serving.md`, and `written-state-count-model.md` pass and merge cleanly.

## Stop and ask if

- Eliminating the trunk copy still leaves sealed `codex-36` `full` peak RSS over `4,000,000,000 B` (allocator fragmentation or `P1` state layout decision needed).
- A clean merge into `main` collides structurally with `#4136` or `#4139` on `discrete_d11.rs`.
