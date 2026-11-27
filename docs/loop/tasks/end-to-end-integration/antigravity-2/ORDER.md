# Gate 1 PR 1: Native-Integer Reversible Butterfly-Rotor Core (`UORBFY01`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030. Source baseline `6c78eeb0c`.

## Native-Space Cycle Contract (D85, `CHARTER.md` §3.7)

1. **Native state schema (`Q15` persistent integer state):**
   - **Parameters (`i16`, Q15 `[-32767, 32767]`, Q12 `4,096` scales):** `embeddings[4096][D]`, `coarse_centroids[64][D]`, per-layer lifting-Givens shears + Q12 diagonals (`rotor_proj`, `key_proj`, `value_proj`, `query_proj`, `ffn_in`, `ffn_out`), integer `RMSNorm` gains, and sparse PKM `key_a`, `key_b`, `values`.
   - **Session carry (`i32` Q15 in `[-2^22, +2^22]`):** per-layer `S³` quaternion carry `h[L][D/4][4]`, sparse PKM slot indices, coarse/fine choices, and emitted `u16` token IDs.
   - **Optimizer state (`i32` momentum + `u64` step/seed):** Integer Lion momentum `m`, deterministic `splitmix64` seed, monotonic `step`. Zero float master copy.
2. **Shared typed operators (`#![forbid(unsafe_code)]`, zero float, zero normalized exponential, zero matmul):**
   - `fwht_inplace_q15`: `i64`-accumulated orthogonal radix-2 FWHT.
   - `LiftingButterfly` / `ButterflyLinear`: `O(D log D)` bit-exact reversible 3-shear lifting-Givens mixer (`inverse(forward(x)) == x`) and exact integer reverse-mode adjoint.
   - `quaternion_rotor_step_q15` + `blelloch_inclusive_scan`: `S³` holographic quaternion binding (`k_t ⊗ v_t`), RG-LRU dyadic decay transport (`1 - 2^-s`, input shift `ceil(s/2)`), and unbinding (`conj(q_t) ⊗ h_t`), plus an exact finite-group (`{±1, ±i, ±j, ±k}`) `O(log L)` Blelloch prefix-scan primitive (`ButterflyRotorModel` runs sequential `quaternion_rotor_step_q15`).
   - `rmsnorm_forward_q15` (`isqrt_u128`), `relu2_forward_q15`, `SparsePkmMemory::read` (`K = 4` slots).
   - Factored `64×64` hierarchical head (`128` candidates/token, zero `V×D` loop) supporting `HuiBelkinRescaledSquare`, `Sparsemax64`, and `target_q48_mass_from_hidden` (`128` dot products, exact `2^48` normalization).
   - `apply_lion_step`: NITI dynamic shift + symmetric zero-decaying Integer Lion + deterministic `splitmix64` stochastic rounding directly onto `i16` weights.
   - Gate 1 PR 2 (`antigravity-3`) mirrors these operators in AOT CUDA with `0`-diff raw-word parity.
3. **Exact replay oracle:** Unit tests verify whole-window ≡ irregular-chunk ≡ 1-token incremental ≡ suffix-extension ≡ training-forward ≡ post-`UORBFY01`-reload bit-identity on `ButterflyRotorModel`, plus parallel ≡ sequential Blelloch scan identity on finite basis quaternions.
4. **Direct persistent-state training:** Exact reverse-mode adjoints + NITI + Lion update `i16` Q15 weights directly.
5. **Lossless serialization (`UORBFY01`):** Fail-closed little-endian container verifying magic, FNV-1a checksum, version, zero padding, strict `LossKind` params, `[-32767, 32767]` Q15 domain, bijective `vocab_perm`, and exact byte length.

## Scope

- `crates/uor-r4-native-vector/src/butterfly_rotor.rs`
- `crates/uor-r4-native-vector/src/lib.rs`
- `docs/loop/tasks/end-to-end-integration/antigravity-2/**`
- `docs/roadmap/items/end-to-end-integration.md`
- `docs/research/notebook/m4-serving.md`

## Cancel Probe

Confirm zero external deps, `#![forbid(unsafe_code)]`, clean host & `wasm32-unknown-unknown` builds, and all existing crate + `native_space_state_contract` tests green.

## Done Check

1. `cargo test -p uor-r4-native-vector` passes all tests (reversibility, Blelloch finite-group parity, `8/8` finite-diff adjoints, `16/16` unwrapped opposite-role learning on both loss arms, train-vs-step & chunk/suffix/reload parity, vocab bounds, symmetric Lion decay, `128`-candidate Q48 target mass normalization, fail-closed `UORBFY01` checks).
2. `cargo clippy --lib -- -D warnings`, `wasm32-unknown-unknown`, `native_space_state_contract`, `check_no_softmax.py`, `check_no_matmul.py`, `uor-harness check`, and `uor-harness loop pr-check --local` pass.
