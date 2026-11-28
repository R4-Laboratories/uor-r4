# Gate 1 `UORBFY03` Zero-Alloc Serving, `Entmax15` Loss & `L=8` Breakthrough (`antigravity-6`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

## Native-Space Cycle Contract (D85, `CHARTER.md` §3.7)

1. **Native state schema (`UORBFY03 v1` Q15 `i16`, unchanged binary layout):** Persistent `ButterflyV3Model` (`i16` Q15 weights/embeddings/centroids/`shift_mix_q15`/`gate_bias_q15`/wide PKM, Q12 norms/diagonals, `u16` freq-sorted `vocab_perm`, `u64` step/seed) and `ButterflyV3Session` (`i32` `S³` carry + `3`-step `shift_hist`) in `butterfly_rotor_v3.rs`, plus `LossKind::Entmax15` (`code = 3, param = 0`) in `butterfly_rotor.rs`.
2. **Shared typed operators (`0` float, `0` normalized exponential/squashing, `0` dense matmul, `0` hardware `/` in CUDA):**
   - **Zero-allocation precomputed-`beta_q15` serving (`ButterflyV3Prepared` + `ButterflyV3FastSession`):** Load-time `beta_q15` tables, exact Euclidean-norm `SAFE_NORM2 = (ACT_CLAMP >> 1)^2` unclamped butterfly fast path, unrolled `s = 0..2` stages, `u64::isqrt`/`u128::isqrt`, and `stereographic_gate_forward_q15`, verified `0` diffs vs `step()` and RTX 5090 `sm_120` while clearing `>= 1,000 tok/s` on single-core Apple M1.
   - **Pure-integer Tsallis `α = 1.5` Fenchel-Young loss (`LossKind::Entmax15`):** Exact 20-step pure-shift integer bisection solving `sum_i [(z_i - tau)/2]_+^2 == 1` (`dz_i = p_i - 1[i = t]`, zero `exp`/`log`/`softmax`/div), shift-invariant and zero-sum.
3. **Exact transition replay oracle:** Gate 2 verifies `0` raw-word diffs between CUDA and CPU `butterfly_rotor_v3` across `4` configs (`Entmax15` & `HuiBelkin k=64`, `D ∈ {32, 256, 512}`, `L ∈ {2, 8}`, `ffn4`, `sk=32`, `kd ∈ {8, 32}`, `freq_sorted`, multi-step AdamW, `UORBFY03 v1` reload).
4. **Direct persistent-state training & lossless serving:** Controlled 1-variable dual-arm GPU run at identical `D=512, L=8, ffn4, sk=32, kd=32` (`6,863,360` params, `13,734,980 B` = `13.73 MB`): Arm 1 (`Entmax15`) vs Arm 2 (`HuiBelkin k=64`), `2` pretrain epochs on `train-100m` (`200M` tok, `batch=192`, `200`-step warmup `512->8192->512`, `wd_shift=20`) + `1` answer-only SFT epoch (`mask==1`), cold unwrapped `E1–E4`/`chat-v0-p2`, and Apple M1 `bench-cpu` (`0`-diff CUDA=M1). Sealed `DEV-test` stays untouched.

## Scope

- `crates/uor-r4-native-vector/src/butterfly_rotor.rs` (`LossKind::Entmax15` + tests)
- `crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs` (`ButterflyV3Prepared`, inner grad check, tests)
- `docs/loop/tasks/end-to-end-integration/antigravity-6/**`
- `docs/roadmap/items/end-to-end-integration.md` and `docs/research/notebook/m4-serving.md`

## Cancel Probe

Confirm (1) `butterfly_kernels_v3.sm_<CC>.cubin` Gate 1 PTX/SASS audit has `0` PTX float/div/MMA and `0` arithmetic SASS float/MMA, and (2) Gate 2 finite GPU-vs-CPU parity passes `0` diffs across all `4` configs (incl. `Entmax15`) before `train-100m`. Obtain non-author `TEST FITNESS: FIT` on `SHEET.md` first.

## Stop List

- Never touch or read sealed `dev-test.*`.
- Never introduce floats, hardware `/` in CUDA, normalized exponential/squashing, or dense matmul/GEMM/MMA (`CHARTER §3.2`, §3.3, §3.7).
- Never break existing `UORBFY01`/`UORBFY03` tests or `antigravity-5` `UORBFY03 v1` replay.
- Never leave a rented pod idle (`L-004`): archive to Wick ZFS and delete immediately on completion.

## Done Check

1. `SHEET.md` receives fresh-context non-author `TEST FITNESS: FIT` (`FIT.md` + `GO.md`).
2. Gate 1 PTX/SASS audit (`0` float/div/MMA) and Gate 2 parity (`0` diffs, `4/4` configs) pass.
3. Both `13.73 MB` arms complete 2 pretrain + 1 SFT epochs on `train-100m`, unwrapped `E1–E4`/`chat-v0-p2`, and Apple M1 `bench-cpu` (`ButterflyV3Prepared`, `0`-diff CUDA=M1, plus `antigravity-5` Arm 2 M1 re-bench).
4. `receipts.sha256`, `REPORT.md`, `uor-harness loop pr-check --local`, fresh-context review, and CI pass.
