# Native-space state contract (D85)

This document records the executable native-space state schema, source-level operator registry, raw-word replay fixture, planted-fault controls, and legacy-vs-native file boundary for [`native-space-state-contract`](../roadmap/items/native-space-state-contract.md) (Issue `#3973`, CHARTER §3.7, `D85`).

- **Machine-checked registry & schema:** [`crates/uor-r4-native-vector/src/contract.rs`](../../crates/uor-r4-native-vector/src/contract.rs)
- **Executable raw-word replay & planted-fault fixture:** [`crates/uor-r4-discrete/tests/native_space_state_contract.rs`](../../crates/uor-r4-discrete/tests/native_space_state_contract.rs)
- **Scope:** Contract schema, operator mapping, finite raw-word parity, and planted-fault refusal only (`$0.00` GPU spend). No training run, throughput claim, or language capability claim is made.

## 1. Versioned native-state schema (`uor-r4.native-state-schema/1`)

Every carried field across the native default paths uses an explicit integer or fixed-point domain with zero floating-point storage:

| Field | Lane | Domain & width | Scale / range | Lifetime | D85 parity field | Serialization / anchor |
|---|---|---|---|---|---|---|
| `discrete.token_id` | Discrete SpiralCore | `u32` (32-bit) | `2^0`; `0..vocab` (`2..=8192`) | Prompt | `token ids` | U32 LE; [`discrete_d11.rs`](../../crates/uor-r4-integer/src/discrete_d11.rs) |
| `discrete.table` | Discrete SpiralCore | `i32[8]` (32-bit) | `{-2, 0, 2}` (`16` Cl(0,6) roots) | Model | `carry state` | `UORLUT01` `discrete.table`; [`d11_export.rs`](../../crates/uor-r4-discrete/src/d11_export.rs) |
| `discrete.cells` | Discrete SpiralCore | `u32` (32-bit) | `root(b0)*16 + root(b_last)` (`0..256`) | Model | `memory addresses` | `UORLUT01` `discrete.cells` (verified vs table) |
| `discrete.routes` | Discrete SpiralCore | `u32` (32-bit) | `2^0`; `0..64` per `(block, stage)` | Model | `route choices` | `UORLUT01` `discrete.routes`; [`discrete.cu`](../../crates/uor-r4-discrete/src/discrete.cu) |
| `discrete.gates` | Discrete SpiralCore | `i32` (32-bit) | `{-1, 0, +1}` trit (`0` in Route) | Model | `carry state` | `UORLUT01` `discrete.gates`; [`discrete.cu`](../../crates/uor-r4-discrete/src/discrete.cu) |
| `discrete.read_vector` | Discrete SpiralCore | `i32` (32-bit) | `8 * blocks` exact `snap_3`/action coords | Per step | `carry state` | `DiscretePrefixEvaluation::read_vector` |
| `discrete.route_trace_words` | Discrete SpiralCore | `u32` (32-bit) | `header(4) + stages * 6` per block | Per step | `route choices` | `DiscreteTracedPrefixEvaluation::trace_words` |
| `discrete.active_read_trace_words` | Discrete SpiralCore | `u32` (32-bit) | `header(4) + stages * 28` + relation tail | Per step | `read choices` | `DiscreteTracedPrefixEvaluation::{trace_words, relation_trace_words}` |
| `discrete.admitted_cells` | Discrete SpiralCore | `u32` (32-bit) | `1..=kept_cells`, `< 256` | Per step | `memory addresses` | `DiscretePrefixEvaluation::admitted_cells` |
| `discrete.emitted_top8_and_prediction` | Discrete SpiralCore | `u32` (32-bit) | `0..vocab` argmin L1 inside cells | Per step | `token ids` | `DiscretePrefixEvaluation::{top8, prediction}` |
| `discrete.exact_copy_state` | Discrete SpiralCore | `[u32; 4]` / `[u32; 8]` | `[active, source, cursor, prompt_end]`; span `<= 8` | Session | `memory addresses` | `ExactCopyState::words` / `ExactCopyStep::words` |
| `discrete.epoch` | Discrete SpiralCore | `u64` (64-bit) | Strictly improving move count `0..u64::MAX` | Optimizer cursor | — | `UORLUT01` `shape.epoch` |
| `count.s0_token_id` | Written-state S0–S6 | `u16` (16-bit) | `0..4096` | Prompt | `token ids` | U16 LE stream; [`ngram_count.rs`](../../crates/uor-r4-discrete/src/ngram_count.rs) |
| `count.s2_prime_route_key` | Written-state S0–S6 | `u64` / `u128` (128-bit) | 12 bits/slot ordered suffix bijection (`1..=10`) | Model | `route choices` | `FrozenS2Table` / `NgramTriple` |
| `count.s2_count_value` | Written-state S0–S6 | `u32` (32-bit) | Exact / KN count `0..=u32::MAX` | Model | `memory addresses` | `FrozenS2Table` / `DiscreteSparseCountTable` |
| `count.s5_pick_block_code` | Written-state S0–S6 | `u16` (16-bit) | 2-bit/coord XOR+popcount block code | Session | `read choices` | Derived from `uor-r4.lut-discrete-trunk/1` |
| `count.s6_mixer_weights_and_floor` | Written-state S0–S6 | `u64` (64-bit) | Q32/Q48 integer mixer weights, floor `f >= 1` | Artifact | `token ids` | `UORP1A01` / `uor-r4.lut-discrete-count/1` |
| `q31.vector_state` | Native Q31 Vector | `i64[4]` (64-bit) | `2^-31` fixed point in `-2^31..=2^31` | Model | `carry state` | 32 B LE [`Q31VectorState`](../../crates/uor-r4-native-vector/src/lib.rs) |
| `q31.hamilton_model` | Native Q31 Vector | `i64[4]` + cursor | Gate Q31 + `epoch: u64` + `step_shift: u32` + ver `1` | Artifact | `carry state` | 48 B LE / `uor-r4.lut-stack/12` `q31.state` |
| `q31.lane_gates` | Native Q31 Vector | `K * i64[4]` + cursor | `K >= 1` Q31 gates + shared cursor | Artifact | `carry state` | `(32K+16)` B LE / `uor-r4.lut-stack/13` `q31.gates` |
| `q31.recurrence_carry` | Native Q31 Vector | `i64[4]` (64-bit) | Q31 `before` / `transport` / `after` | Session | `carry state` | `NativeQ31Carry` + `UR4SESS1..3` |
| `q31.read_address` | Native Q31 Vector | `i64[16]` (64-bit) | Exact folded sum (`width <= 4096`) | Per step | `memory addresses` | [`q31_read_address`](../../crates/uor-r4-native-vector/src/read.rs) |
| `q31.read_pick` | Native Q31 Vector | `u32` (32-bit) | 1-based picked row (`0` at `t=0`) | Per step | `read choices` | `Q31ReadOutput::picks` / `Q31ReadStep::pick` |
| `q31.step_shift_cursor` | Native Q31 Vector | `u32` (32-bit) | `0..=31` (`2^-step_shift` credit) | Optimizer cursor | — | `Q31HamiltonModel` / `Q31LaneGates` LE trailer |

## 2. Source-level operator registry (`uor-r4.native-operator-registry/1`)

Every operator ID below is checked at workspace path and symbol level by `contract_registry_files_and_symbols_exist_in_workspace` in [`native_space_state_contract.rs`](../../crates/uor-r4-discrete/tests/native_space_state_contract.rs):

| Operator ID | CUDA training | Integer serving | CPU oracle | WASM |
|---|---|---|---|---|
| `discrete_action_and_compose` | `discrete.cu::action` | `discrete_d11.rs` (`fn action`) | `discrete_route.rs::CpuDiscreteRoute` | `wasm_serve.rs::wasm_discrete_predict` |
| `discrete_route_read` | `discrete.cu::route_vector` | `discrete_d11.rs::evaluate_prefix_traced` | `discrete_route.rs::CpuDiscreteProposalScorer` | `wasm_serve.rs::wasm_discrete_predict` |
| `discrete_flock_and_relation_read` | `discrete.cu::flock_vector` | `discrete_d11.rs::evaluate_prefix_traced` | `discrete_route.rs::CpuDiscreteFlock` | `wasm_serve.rs::wasm_discrete_predict` |
| `discrete_pick_read` | `discrete.cu::pick_vector` | `discrete_d11.rs::pick_row` | `discrete_route.rs::CpuDiscretePick` | `wasm_serve.rs::wasm_discrete_predict` |
| `discrete_writer_admission_and_emission` | `discrete.cu::writer` | `discrete_d11.rs::evaluate_prefix_internal` | `discrete_route.rs::CpuDiscreteProposalScorer` | `wasm_serve.rs::wasm_discrete_predict` |
| `discrete_counterfactual_choose_apply` | `discrete.cu::discrete_apply` | — | `discrete_route.rs::CpuDiscreteProposalApplier` | — |
| `written_state_s2_count_and_mix` | `ngram_count.cu::ngram_s2_freeze` | `discrete_count.rs::DiscreteCountModel` | `ngram_count_oracle.rs::oracle_adjusted` | — |
| `q31_hamilton_rotate` | `native_vector.cu::native_q31_hamilton_rotate` | `q31_serving.rs::Q31ServingSession` | `uor-r4-native-vector::q31_hamilton_rotate` | `wasm_serve.rs::wasm_step` |
| `q31_hamilton_train_step` | `native_vector.cu::native_q31_hamilton_train_step` | — | `uor-r4-native-vector::q31_hamilton_train_step` | — |
| `q31_first_pick_read` | `native_vector.cu::native_q31_first_pick_read` | `q31_read.rs::ServedQ31Read` | `uor-r4-native-vector/src/read.rs::Q31FirstPickRead` | `wasm_serve.rs::wasm_step` |
| `uorlut01_lossless_serialization` | `d11_export.rs::export_d11` | `discrete_d11.rs::IntegerDiscreteModel` | `discrete_route.rs::CpuDiscreteProposalApplier` | `wasm_serve.rs::wasm_load_discrete` |

## 3. Unified raw-word replay & planted-fault matrix

Executed on host CPU (`needs_gpu: false`, `$0.00` GPU spend) via `cargo test --locked -p uor-r4-discrete --test native_space_state_contract -- --nocapture` (`4/4` passed). CUDA kernels (`.cu`) did not execute on this macOS host; their GPU parity relies on existing `#[cfg(feature = "cuda")]` tests (`cpu_oracle_parity.rs`, `native_vector_cuda.rs`, `pick_read_cuda.rs`).

- **Discrete replay (`39` prefixes over `3` tokens × `2` blocks, context `3`):**
  - Route: `1,248` raw `u32` route trace words, `624` raw `i32` carry words, `78` admitted cell `u32` words, `78` `top8` (`2` admitted tokens per prefix × `39`) + `39` `prediction` `u32` IDs match bit-identically before and after one native `Table` move (`epoch 0 -> 1`).
  - Flock: `4,680` raw `u32` active trace words (`126` kept node slots, `126` transport action slots), `624` carry words, `78` cell words, `78` `top8` + `39` `prediction` IDs match bit-identically.
  - Pick + `ExactCopyState`: `IntegerDiscreteModel::evaluate_prefix_traced` leaves `trace_words` empty on `DiscreteReadMode::Pick` as designed (`discrete_d11.rs:4480`), so Pick runs with `compare_trace_words = false` (`pick`/`distance`/`ties` are not emitted on `IntegerDiscreteModel`), comparing `624` raw `i32` carry words (`read_vector`), `78` cell words, `78` `top8`, `39` `prediction` IDs, and `CpuDiscretePick::read_traced` (`pick == Some(1)` and `blocks == read_vector` on `[0, 1, 2]`), plus `[active, source, cursor, prompt_end, emitted, exact_id, copied, release] = [1, 0, 1, 2, 0, 0, 1, 0]`.
- **Native Q31 replay (`/12`, `/13`, `/14`, 8-token prompt `[3, 5, 8, 13, 21, 34, 55, 89]`):**
  - `/12` `Q31HamiltonModel` (`step_reference` 16 steps): `4` gate words and `128` layer/lane transitions (`960` transport + continuity `i64` words: `8 * 16 * 4 + 7 * 16 * 4`) match bit-identically; `Q31HamiltonModel` is a transport layer with no token head, while `UR4SESS` save/restore separately preserves all `4` continuation tokens (`prompt[4..]`) and post-restore carries bit-identically.
  - `/13` `Q31LaneGates`: `8` lane gates × `2` layers × `8` steps match `Q31LaneGates::transition` raw `i64[4]` words bit-identically.
  - `/14` `Q31FirstPickRead`: `compare_q31_read_replay` decodes served `q31.read` via `read_q31_read_uorlut01` and verifies bit-identical `q31_read_address` `query`/`key` vectors (`256` `i64` address words), `8` `u32` picks, and `256` `i64` out words across `Q31FirstPickRead::{apply, step}` and `IntegerStackSession` (`ServedQ31Read`).
- **Recorded planted faults (all detected through the shared replay comparators):**

| Fault ID | Planted mutation | Failure site | Observed count / refusal |
|---|---|---|---|
| `F1_operator` | `q31_hamilton_rotate` right-shift `31 -> 30` | `CarryState` raw transport words | `112` mismatches |
| `F2_scale` | `Q31FirstPickRead` `gain_shift 4 -> 5` | `CarryState` raw out words | `7` mismatches |
| `F3_route` | Single `discrete.routes` port `1 -> 2` | `RouteChoices` raw `trace_words` | `39` row mismatches |
| `F4a_flock` | Flock gate trit `1 -> -1` | `CarryState` raw words | `64` mismatches |
| `F4b_q31` | Q31 gate coord `+ 2^24` on `/12` | `CarryState` raw `q31.state` | `1` state-header mismatch |
| `F5a_q31_pick` | Rotated/perturbed `/14` `key_gates` in `q31.read` (+ 1-LSB tie-sensitivity check) | `ReadChoices` + `MemoryAddresses` via `compare_q31_read_replay` | `3` pick + `8` address mismatches (tie `[0, 1, 2] -> [0, 1, 1]`) |
| `F5b_flock_read` | Flock token-0 block-0 root changed (`0 -> 6`) | `ReadChoices` active trace slots | `25` row mismatches |
| `F6_cell_refused` | Bit-flip `discrete.cells[0]` (`^1 -> 99`) | Fail-closed loader check | Refused (`cells != table`) |
| `F7_token` | Token `0 <-> 1` rows & cells swapped | `TokenIds` (`top8` & `prediction`) | `15` row mismatches |

## 4. Legacy float / exporter paths vs native-state implementation (`uor-r4.native-boundary-inventory/1`)

- **Native default state & lossless serialization:** [`crates/uor-r4-native-vector/src/contract.rs`](../../crates/uor-r4-native-vector/src/contract.rs), [`crates/uor-r4-discrete/src/d11_export.rs`](../../crates/uor-r4-discrete/src/d11_export.rs), [`crates/uor-r4-discrete/src/trunk.rs`](../../crates/uor-r4-discrete/src/trunk.rs).
- **Shared native operators (CUDA + host reference + WASM/serving):** [`crates/uor-r4-native-vector/src/lib.rs`](../../crates/uor-r4-native-vector/src/lib.rs), [`crates/uor-r4-native-vector/src/read.rs`](../../crates/uor-r4-native-vector/src/read.rs), [`crates/uor-r4-native-vector/src/read_train.rs`](../../crates/uor-r4-native-vector/src/read_train.rs), [`crates/uor-r4-discrete/src/discrete.cu`](../../crates/uor-r4-discrete/src/discrete.cu), [`crates/uor-r4-discrete/src/native_vector.cu`](../../crates/uor-r4-discrete/src/native_vector.cu).
- **Native serving runtimes (`#![forbid(unsafe_code)]` / WASM-clean):** [`crates/uor-r4-integer/src/discrete_d11.rs`](../../crates/uor-r4-integer/src/discrete_d11.rs), [`crates/uor-r4-integer/src/discrete_count.rs`](../../crates/uor-r4-integer/src/discrete_count.rs), [`crates/uor-r4-integer/src/stack/q31_serving.rs`](../../crates/uor-r4-integer/src/stack/q31_serving.rs), [`crates/uor-r4-integer/src/stack/q31_read.rs`](../../crates/uor-r4-integer/src/stack/q31_read.rs), [`crates/uor-r4-integer/src/wasm_serve.rs`](../../crates/uor-r4-integer/src/wasm_serve.rs).
- **Diagnostic / CPU parity oracle (D84 development only):** [`crates/uor-r4-cpu-train/src/discrete_route.rs`](../../crates/uor-r4-cpu-train/src/discrete_route.rs), [`crates/uor-r4-integer/src/lawful_likelihood.rs`](../../crates/uor-r4-integer/src/lawful_likelihood.rs) (reference `-log2` scoring).
- **Legacy float-master / export-bridge paths (historical/reference only under D85):** [`crates/uor-r4-float-train/`](../../crates/uor-r4-float-train/) (offline F32 master trainer, barred from default path), [`crates/uor-r4-integer/src/bin/fbs-export.rs`](../../crates/uor-r4-integer/src/bin/fbs-export.rs) (historical F32-to-4-bit `lut-stack/1..11` quantization exporter).
