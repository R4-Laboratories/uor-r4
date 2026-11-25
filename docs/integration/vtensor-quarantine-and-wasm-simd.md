# `afflom/vTensor` CHARTER §3 Quarantine & Pure-Integer WASM SIMD128 Roadmap

Recorded under [`studio-lexlean-mvm-workbench`](../loop/tasks/studio-lexlean-mvm-workbench/antigravity-40/ORDER.md) (Stage A5, `antigravity-40`, Issue `#3939`).

## 1. Audited `afflom/vTensor` Inventory

- **Repository:** `https://github.com/afflom/vTensor`
- **Default branch:** `codex/vtensor-implementation`
- **Audited commit:** `b0ce3f6b72ce3b12ea3081977b61cbb01b36980f`
- **License:** Dual-licensed `MIT OR Apache-2.0`
- **Observed architecture:**
  - Prism-modeled GGML device backend (`VT-01..VT-46`) written in Rust for `llama.cpp` inference and graph execution.
  - Sole compute substrate is `verdant-tensor = "=2.17.0"` (pinned in `vTensor`'s manifest to `afflom/verdant-engine` rev `4654da8724f142a248b32739f1d930066e281bd5`).
  - Executes GGML buffer/tensor operations over `F64`, `F32`, `F16`, and `BF16` IEEE floating-point types, Berkeley SoftFloat deterministic reductions, floating-point `RMS_NORM` / L2 / unary math ops, and `4x4` `F64` matrix primitives.

## 2. Binding CHARTER §3 Quarantine Boundary

`afflom/vTensor` and its underlying `verdant-tensor` substrate directly violate UOR-R4's foundational invariants:

1. **Geometry only ([CHARTER.md](../../CHARTER.md) §3.1):** Mechanism and compute signals must arise from UOR geometry, not external floating-point tensor backends.
2. **No softmax or squashing (§3.2):** Disallows smooth exponential normalization and transcendental squashing ops inside active model paths.
3. **No matmul (§3.3):** Dense matrix multiplication and `GEMM`/matrix kernels are prohibited in training and serving.
4. **One persistent native integer state (§3.7, D85):** Token addresses, vector coordinates, carry, routes, memory reads, weights, and updates stay in one native integer/fixed-point space; floats exist only in explicitly isolated reference/diagnostic harnesses.

**Quarantine rule:**

- `afflom/vTensor` and `verdant-tensor` MUST NEVER be added to workspace `Cargo.toml`, `Cargo.lock`, `crates/uor-r4-*`, `web/studio/foundry/studio/core/**`, `uor-foundry-v13-wasm`, or Studio runtime bundles.
- No `vTensor` FFI bridge, GGML backend wrapper, or float tensor op may enter M1–M8 training or browser serving paths.
- Note on pre-existing locked WASM assets: `web/studio/foundry/studio/shell/uor_foundry_v13_wasm_bg.wasm` (pinned by SHA-256 in `foundry-assets.lock.json` from `Casey-allard/lexlean-r4@0b5b6d30a`) contains an unused internal pure-integer Q15/Q16 module `lexlean_r4_wasm::vtensor_semiring_3d` (`studio_vtensor_3d_self_check_json`, never invoked by `uor-v13-worker.mjs`); it links zero `afflom/vTensor` or `verdant-tensor` crates and executes no GGML/float ops.

## 3. Pure-Integer Browser WASM SIMD128 Plan (Target: Stage B4)

Instead of importing `vTensor`, browser acceleration for Studio serving (`uor-foundry-v13-wasm` and `core.wasm`) will use **pure-integer WASM SIMD128** compiled cleanly in Rust (`-C target-feature=+simd128` via `core::arch::wasm32`) and gated behind a deterministic runtime probe.

### 3.1 Prerequisite: Zero-Matmul Auditor Opcode Classification

[`scripts/audit_zero_matmul_wasm.py`](../../scripts/audit_zero_matmul_wasm.py) (decoder `0xFD` branches at lines `236-247`, `unclassified` recording at lines `270-271`, and unit checks at lines `751-757`) currently skips immediates for `0xFD` WASM SIMD128 prefix instructions and marks **every** `0xFD` instruction as `unclassified`, which fails `--serving` checks if any `+simd128` module is built today.

Before Stage B4 ships any SIMD128 WASM binary, Stage B4 MUST update the auditor (and any Rust port of it):

- Decode the unsigned LEB128 sub-opcode after `0xFD`.
- **Allowlist strictly pure-integer `0xFD` SIMD128 sub-opcodes** (`v128` loads/stores, lane load/store/splat/shuffle/extract/replace, bitwise `and`/`or`/`xor`/`not`/`bitselect`/`any_true`, and `i8x16`/`i16x8`/`i32x4`/`i64x2` add/sub/mul/shift/neg/abs/min/max/extend/narrow/`i32x4.dot_i16x8_s`).
- **Classify every float SIMD lane sub-opcode as `float > 0`** (hard failure): `f32x4.*`, `f64x2.*`, SIMD float comparisons, `f32x4.convert_i32x4_*`, `i32x4.trunc_sat_f32x4_*`, `i32x4.trunc_sat_f64x2_*_zero`, and relaxed-SIMD FMA/float opcodes.
- Preserve fail-closed behavior for unknown `0xFD` sub-opcodes (`unclassified > 0`).

### 3.2 Pure-Integer Kernel Targets (Stage B4)

1. **`uor-foundry-v13-wasm` (`UORR4V13` / native `UORLUT01`):** Vectorize quantized integer vector products (`i32x4.dot_i16x8_s`, `i16x8` / `i32x4` widen-accumulate), integer RMS/norm scaling (pure integer reciprocal/shift), and LUT/carry scans using `core::arch::wasm32` integer intrinsics — zero `f32x4`/`f64x2` lanes.
2. **`web/studio/foundry/studio/core` (`vram_store.rs`):** Vectorize 4 KiB page entropy/equality classification (`Fill`, `Repeated`, `Sparse`, `Selector`, `Packed`) and XOR/delta diffing across 16-byte `v128` lanes inside safe wrappers isolated behind cfg target gates so non-`wasm32` targets remain scalar `#![forbid(unsafe_code)]`.

### 3.3 Bit-Identical Parity & Runtime Dispatch

- **Bit-identical parity oracle (§3.7 / D85):** Every Stage B4 `+simd128` integer kernel must pass an exact bit-identical differential suite against the scalar Rust integer implementation across token ids, routes, carry states, memory reads, and encoded vRAM page tags.
- **Runtime capability probe (landed in Stage A5):** [`uor-r4-runtime.mjs`](../../web/studio/foundry/studio/shell/uor-r4-runtime.mjs) exposes `detectWasmCapabilities()` (also surfaced via `getStatus().wasmCapabilities` and `window.__uorR4.getWasmCapabilities()`), validating a minimal 64-byte pure-integer `i32x4.dot_i16x8_s` (`0xfd 0xba 0x01`) module (`WASM_INT_SIMD128_PROBE`, zero floats) with `WebAssembly.validate`.
- **Fallback rule:** While `activeIntegerKernel` remains `'wasm-scalar-integer'` in Stage A5, Stage B4 selects the integer-SIMD artifact only when `simd128IntegerSupported === true`, transparently falling back to scalar integer WASM on browsers or sandboxes without WASM SIMD128, and pairing with Stage A3 `vram-store.mjs` for compressed shard residency.
