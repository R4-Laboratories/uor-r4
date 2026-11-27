# ORDER: written-state-count-model, antigravity-18 (Phase P5 PR 1 of 2 — 4,096-row Driver cap lift & pure-integer P5 mixed-mass Brier S1 core)

Lab: antigravity. Item: [written-state-count-model](../../../../roadmap/items/written-state-count-model.md) (`#4063`, `M2+/P5`, [`DECISION.md` §4 `P5`](../../../../research/reviews/pi-review-2026-11-16/DECISION.md#L335-L346)). Base: `e90a1429c`.

## Goal & scope

Deliver **Phase `P5` PR 1 of 2** (host engine + differential/audit verification, `$0.00` GPU spend, `0` pods active) inside allowed item scope (`crates/uor-r4-discrete/**`, `docs/loop/tasks/written-state-count-model/antigravity-18/**`, `docs/research/notebook/m4-serving.md`, `docs/roadmap/items/written-state-count-model.md`; `0` `Cargo.toml`/`Cargo.lock` edits):

1. **Lift `Driver`'s `1..64`-row cap (`training.rs:307,312`):** Widen `DriverConfig::validate` to `1..=corpus::MAX_BATCH_ROWS` (`4,096`) matching `corpus.rs:15` and `device.rs:1456`, enabling `Driver::run_block_transitions` (`training.rs:3313`) and `Device::score_rows_k` (`rowsk_device.rs:56`) at `4,096` rows; update `driver_source_pin::DRIVER_SOURCE_SHA256` and add `4,096`-row boundary & `K`-coordinate schedule tests.
2. **Add `crates/uor-r4-discrete/src/p5_brier_core.rs` (`#![forbid(unsafe_code)]`, pure integer, zero floats/matmul/normalized exponential):** Native `S1` state (`16,384` `VectorBlock` coordinates `<= 1e5`), exact 4,095-token mixed-mass integer Q32 Brier objective (`R1-M6`), exact `rows-K` + `BVG-K` scoring/commit oracle, paired equal-trial re-seeded random-coordinate controls (`DECISION.md:342`), and lossless `UORLUT01` (`discrete.table`/`discrete.cells`) round-trip I/O.

## Five native-space items (`D85`, `native-space-cycle`)

1. **State schema:** `P5S1State` holds `V × B = 4,096 × 4 = 16,384` `S1` `VectorBlock` coordinates (`<= 1e5`), stored as `Cl(0,6)` 16-root E8-orbit indices `r in 0..16` (bijective with `ROOTS[r]` in `seed1.d11`'s `discrete.table` and Pick code `ROOT_CODE[r]`), plus `epoch: u64`.
2. **Shared operators:** `MOVE_ROOT[16][16]` and `ROOT_CODE[16]` derive directly from `OUT_DIR/geometry.rs`, matching `discrete.cu`, `rowsk_device.rs`, `bvgk_device.rs`, and `IntegerDiscreteModel::identity_pick_codes` / `P4Session::step` bit-for-bit. `P5RowBrierTable` precomputes exact `(pick_class in 0..5, pick_pos in 1..L-1)` Q32 multi-class Brier scores of the full 4,095-token 22-expert `P4` mixture with zero approximation; `p5_score_block_cpu` and `p5_bvgk_step_cpu` match `bvgk_source.rs` (`resulting_port` tie-break, `L(S_t) < L({best})` joint verify).
3. **Replay oracle:** Unit tests assert `0` raw-word diffs between `P5RowBrierTable` and full 4,095-token `P4Session::step` + `integer_brier_target_q32`, `0` diffs between `p5_bvgk_step_cpu` and brute-force replay across `K in {1, 8, 16, 64}`, and `0` diffs on `UORLUT01` round-trip.
4. **Update rule:** Checked `u64` Brier sums over `<= 4,096` rows, strict improvement (`losses[best] < losses[0]`), joint scratch verification, and in-place E8-orbit generator composition (`root <- MOVE_ROOT[root][lane]`). Equal-trial re-seeded controls (`ReseededInit`, `RandomCoordinateControl`) share the exact trial budget and Brier scorer.
5. **Lossless serialization:** Self-describing `UORP5S10` binary checkpoint (trailing SHA-256) and lossless `discrete.table` + `discrete.cells` rewrite for `UORLUT01` (`seed1.d11` -> `p5_s1.d11`), preserving all other sections byte-for-byte for zero-export-gap packing into `uor-r4.lut-discrete-p4/1`.

## Non-goals (Phase `P5` PR 2 `antigravity-19`)

No Wick GPU training, no `DEV-test` read, and no `P5` gate claim in PR 1; AOT `sm_89` CUDA execution and sealed `DEV-test` Brier/BPB qualification run in PR 2 (`antigravity-19`) after non-author `FIT.md`.
