# Gate 1 `UORBFY03` Capacity, Stereographic Selectivity & Loss Breakthrough (`antigravity-5`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

## Native-Space Cycle Contract (D85, `CHARTER.md` §3.7)

1. **Native state schema (`UORBFY03 v1` Q15 `i16`):** Persistent `ButterflyV3Model` (`i16` Q15 weights/embeddings/centroids/`shift_mix_q15`/`gate_bias_q15`/wide PKM, Q12 norms/diagonals, `u16` freq-sorted `vocab_perm`, `u64` step/seed) and `ButterflyV3Session` (`i32` `S³` carry + `3`-step `shift_hist`) in `crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs`.
2. **Shared typed operators (`0` float, `0` normalized exponential/squashing, `0` dense matmul, `0` hardware `/` in CUDA):**
   - Scalar-reciprocal exact integer RMSNorm (`inv_rms_q40`, `dot_over_denom3_q40`) and widened `±8.0` `S³` unit quaternion (`inv_norm_q40`, exact clamp pullback).
   - 4-tap depthwise `O(D)` shift-mix, pure-`S¹` stereographic selective `S³` gate (`g(z) = z^2 / (1 + z^2)`), 4-branch `O(D log D)` `ReLU²` FFN, `64`-dim learned Sparse PKM (`±256` value init), `head_proj`, `TRAIN` freq-sorted `vocab_perm`, Hui-Belkin (`k = 32, 64`), and Pure-Int Lazy-Sparse AdamW (`butterfly_kernels_v3.cu`).
3. **Exact transition replay oracle:** Gate 2 verifies `0` raw-word diffs between CUDA and CPU `butterfly_rotor_v3` across `4` configs (`D ∈ {32, 256, 512}`, `L ∈ {2, 6, 8}`, `ffn_branches = 4`, `pkm_key_dim ∈ {8, 32}`, `k ∈ {32, 64}`, `freq_sorted` perm, multi-step AdamW, and `UORBFY03 v1` reload).
4. **Direct persistent-state training & lossless serving:** Both arms train persistent `i16` Q15 state on `train-100m`, serialize losslessly as `UORBFY03 v1` (`to_bytes`/`from_bytes` with FNV-1a checksum), and replay `0`-diff on Apple M1 (`bench-cpu`). Sealed `DEV-test` stays untouched.

## Scope

- `crates/uor-r4-native-vector/src/lib.rs` (export `butterfly_rotor_v3`)
- `crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs` (`UORBFY03` core, operators, Pure-Int AdamW, codec, unit tests)
- `docs/loop/tasks/end-to-end-integration/antigravity-5/**`
- `docs/roadmap/items/end-to-end-integration.md` (progress note)
- `docs/research/notebook/m4-serving.md` (append-only row)

## Cancel Probe

Confirm (1) `butterfly_kernels_v3.sm_<CC>.cubin` Gate 1 PTX/SASS audit has `0` PTX float/div/MMA and `0` arithmetic SASS float/MMA, and (2) Gate 2 finite GPU-vs-CPU parity passes `0` raw-word diffs across all `4` `UORBFY03` configs before running `train-100m`. Obtain non-author `TEST FITNESS: FIT` on `SHEET.md` first.

## Stop List

- Never touch or read sealed `dev-test.*`.
- Never introduce floats, hardware `/`, normalized exponential/squashing, or dense matmul/GEMM/MMA (`CHARTER §3.2`, §3.3, §3.7).
- Never break existing `butterfly_rotor.rs` (`UORBFY01`) tests.
- Never leave a rented pod idle (`L-004`): archive all receipts to Wick ZFS and delete immediately upon completion.

## Done Check

1. `SHEET.md` receives fresh-context non-author `TEST FITNESS: FIT` (`FIT.md` + `GO.md`).
2. Gate 1 PTX/SASS audit (`0` float/div/MMA) and Gate 2 finite GPU-vs-CPU parity (`0` diffs across all 4 configs) pass.
3. Both `UORBFY03` arms (`v3_d256_l8_ffn4_k32` & `v3_d512_l6_ffn4_k64`) complete Epoch 1 on `train-100m` (+ auto-SFT Ep 2–3 if `< 2.00 BPB` with monotone descent), cold unwrapped `E2–E4`, and Apple M1 `bench-cpu` (`0`-diff CUDA=M1).
4. Sealed receipts (`receipts.sha256`), `REPORT.md` (negatives first), `uor-harness loop pr-check --local`, fresh-context exact-head review, and CI pass.
