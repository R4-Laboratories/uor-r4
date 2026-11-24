# ORDER: written-state-count-model, antigravity-1

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (#4063, M2+/P1–P4 serving memory). Base: `d11ce98d1`.

## Goal

Eliminate the `~703.5 MB` sparse-trunk loader overhead diagnosed in [`codex-36`](../codex-36/NEXT.md#L18) by replacing nested `BTreeMap<u32, BTreeMap<u16, u32>>` trees and transient `Vec<u32>` buffers in `IntegerDiscreteModel`'s 12 sparse trunk tables (`crates/uor-r4-integer/src/discrete_d11.rs`) with a contiguous validated Flat SoA (`keys: Box<[u32]>`, `tokens: Box<[u16]>`, `counts: Box<[u32]>`, 10 B/entry), without changing `UORLUT01` wire bytes. Verify bit-identical serving transitions against a `BTreeMap` oracle and measure head-to-head RSS, load time, and lookup throughput at `codex-36` scale (`7,164,583` triples, `4,297,201` keys).

## Native-space cycle contract (D85)

1. **Native state schema & carried fields:** `DiscreteSparseCountTable` stores contiguous integer slices `keys: Box<[u32]>`, `tokens: Box<[u16]>`, `counts: Box<[u32]>`, strictly ascending on `(key, token)`, zero float/matmul/softmax, backing sections `7..=18` of `uor-r4.lut-discrete-trunk/1`.
2. **Shared typed operator paths:** `from_triple_bytes` streams aligned `[u8]` section bytes directly; `get(key, tok)` binary-searches `(key, tok)`; `tokens_for_key(key) -> &[u16]` slices via `partition_point`. One source shared across native Rust (`uor-r4-integer`, `uor-chat`) and `serving-wasm`.
3. **Exact transition replay & bit-identical oracle:** `crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs` checks `IntegerDiscreteModel` against a `BTreeMap<u32, BTreeMap<u16, u32>>` oracle over identical `UORLUT01` bytes (sorted & shuffled triples) for raw `read_code`, `rec_code`, `pointer_succ`, `level3_joint_pointer_score`, tie-breaks, fallback, and greedy token IDs, plus planted faults.
4. **Train/serve alignment:** `TrunkTables::to_triples` emits ascending `(key, token, count)` integer triples; `from_triple_bytes` ingests them losslessly in one pass (with exact `(key, token)` sort+dedup fallback on unsorted fixtures). Zero GPU training or float master.
5. **Lossless serialization:** Byte-identical `UORLUT01` sections `7..=18`; zero numeric conversion.

## Cancel probe (run first)

1. Confirm open PRs `#4136`/`#4139` touch only `L36/1225/1953/2538/2616/3878/4365` in `discrete_d11.rs`, disjoint from `L11`, `L859-904`, `L1118-1129`, `L2422-2529`.
2. Confirm `uor-pod status` (`Running: 0 pod(s)`) and `$0.00` GPU spend.

## Files in scope

- `crates/uor-r4-integer/src/discrete_d11.rs`
- `crates/uor-r4-integer/tests/discrete_sparse_table_parity.rs`
- `docs/loop/tasks/written-state-count-model/antigravity-1/ORDER.md`
- `docs/loop/tasks/written-state-count-model/antigravity-1/REPORT.md`
- `docs/research/notebook/m4-serving.md`

## Done when

1. Contiguous Flat SoA `DiscreteSparseCountTable` implemented and verified (`discrete_d11` tests, `discrete_sparse_table_parity.rs`, `discrete_chat_cli`, `clippy -D warnings`, `rustfmt --check`).
2. Native (`audit_zero_matmul_serving.py`) and WASM (`audit_zero_matmul_wasm.py`) audits pass (`0` float / `0` matmul / `0` softmax).
3. Separate `/usr/bin/time -l` release runs at `7,164,583` triples / `4,297,201` keys verify RSS drop and 10,000-probe parity.
4. `uor-harness check --only harness.native_space_contract`, `uor-harness check`, `ratchet d23/d25/d32`, `claim-wording`, `loop report --check`, `loop pr-check`, and `pro` exact-head review pass.

## Stop and ask if

- Any unexpected caller of `DiscreteSparseCountTable` or serving audit failure appears, peak RSS fails to drop substantially, or `#4136`/`#4139` merges and conflicts.

## Explicit non-claims

- Host RSS and parity measurement (`$0.00` GPU) on macOS with a synthetic `codex-36`-cardinality trunk fixture; no Wick 4 GB, P1(b)/M2/M4 BPB, or milestone-closure claim.
