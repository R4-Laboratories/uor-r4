# Gate 1 Diagnostic & Pure-Integer AdamW Repair (`antigravity-4`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

## Native-Space Cycle Contract (D85, `CHARTER.md` §3.7)

1. **Native state schema (`UORBFY01 v2` Q15 `i16`, byte-compatible):** Reuses `ButterflyRotorModel` (`i16` Q15 weights, Q12 scales, `u16` `vocab_perm`, `u64` step/seed) and `ButterflyRotorSession` (`i32` `S³` quaternion carry) in `crates/uor-r4-native-vector/src/butterfly_rotor.rs` with zero binary format change.
2. **Shared typed operators (0 float, 0 softmax, 0 dense matmul, 0 hardware division in default CUDA/CPU/WASM):**
   - Adds exact integer PKM key backward (`g_pkm_key_a`, `g_pkm_key_b`, `dn2[0..16]`), `TRAIN`-only frequency-balanced `vocab_perm` (`build_frequency_balanced_vocab_perm`), and **Pure-Integer Lazy-Sparse AdamW** (`ButterflyAdamWState`, `apply_adamw_step`: `beta1=7/8`, `beta2=63/64`, lazy sparse skip on `embeddings`/`pkm.values`, Q12 anchor WD toward `4,096`, `isqrt_u64` + integer divide + `splitmix64` stochastic rounding onto `i16`).
   - Mirrors both identically in AOT `sm_120` CUDA (`butterfly-diag/src/butterfly_kernels.cu`: `0` PTX float/div/MMA, `0` arithmetic SASS float/MMA).
   - Provides a `CHARTER §3.7` diagnostic-only F32 twin (`butterfly-diag/src/diag_f32_kernels.cu`, `0` softmax/squashing, `0` dense matmul/MMA, never serialized) to isolate the Q15-vs-F32 precision gap.
3. **Exact transition replay oracle:** Gate 2 verifies `0` raw-word diffs between pure-integer CUDA and CPU `butterfly_rotor` across forward, learned PKM keys, `k=1` Brier & `k=8` Hui-Belkin, balanced `vocab_perm`, multi-step `apply_adamw_step`, and `UORBFY01 v2` reload.
4. **Direct persistent-state training & lossless serving:** Pure-integer arms train persistent `i16` Q15 state on `train-100m`, serialize as `UORBFY01 v2`, and replay `0`-diff on Apple M1 (`bench-cpu`). Sealed `DEV-test` stays untouched.

## Scope

- `crates/uor-r4-native-vector/src/butterfly_rotor.rs` (Pure-Integer Lazy-Sparse AdamW, exact PKM key backward, frequency-balanced `vocab_perm` helper, unit tests)
- `docs/loop/tasks/end-to-end-integration/antigravity-4/**`
- `docs/roadmap/items/end-to-end-integration.md` (progress note)
- `docs/research/notebook/m4-serving.md` (append-only row)

## Cancel Probe

Confirm (1) pure-integer `butterfly_kernels.cu` PTX/SASS audit has `0` PTX float/div/MMA and `0` arithmetic SASS float/MMA, (2) diagnostic `diag_f32_kernels.cu` has `0` softmax/exp/squashing and `0` MMA, and (3) Gate 2 finite GPU-vs-CPU parity passes `0` diffs across all pure-integer configs (including `apply_adamw_step`, `learn_pkm_keys`, `k=1` Brier, and balanced `vocab_perm`) before running `train-100m`. Obtain non-author `TEST FITNESS: FIT` on `SHEET.md` first.

## Stop List

- Never touch or read sealed `dev-test.*`.
- Never add softmax, log-softmax, cross-entropy, squashing, or dense matmul/GEMM/MMA even in diagnostic F32 code (`CHARTER §3.2`, §3.3).
- Never serialize diagnostic F32 weights into a served artifact (`CHARTER §3.7`).
- Never leave a rented pod idle (`L-004`): archive all receipts to Wick ZFS and delete immediately upon completion.

## Done Check

1. `SHEET.md` receives fresh-context non-author `TEST FITNESS: FIT` (`FIT.md` + `GO.md`).
2. Pure-integer Gate 1a PTX/SASS audit (`0` float/div/MMA), diagnostic Gate 1b (`0` softmax/MMA), and Gate 2 finite GPU-vs-CPU parity (`0` diffs) pass.
3. All 4 `2×2` factorial arms complete Epoch 1 on `train-100m` (`DEV-test` untouched), plus Epochs 2–3 SFT + unwrapped `E2–E4` + Apple M1 `bench-cpu` (`0`-diff CUDA=M1) on any pure-integer arm breaking `< 2.00 BPB` (or best pure-integer arm).
4. Sealed receipts (`receipts.sha256`), `REPORT.md` (negatives first), `uor-harness loop pr-check --local`, fresh-context exact-head review, and CI pass.
