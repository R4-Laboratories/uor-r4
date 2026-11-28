# Gate 1 PR 2: AOT CUDA Butterfly-Rotor Training Pilot & Unwrapped Eval (`antigravity-3`)

Lab: antigravity. Item: [End-to-end integration](../../../../roadmap/items/end-to-end-integration.md). References #4064 and #2030.

## Native-Space Cycle Contract (D85, `CHARTER.md` §3.7)

1. **Native state schema (`UORBFY01` Q15 `i16`):** Reuses the exact persistent `ButterflyRotorModel` (`i16` Q15 weights, Q12 scales, `u16` `vocab_perm`, `u64` step/seed), `ButterflyRotorSession` (`i32` `S³` quaternion carry), and `ButterflyLionState` (`i32` momentum) landed in `crates/uor-r4-native-vector/src/butterfly_rotor.rs` (`antigravity-2`).
2. **Shared typed operator paths (0 float, 0 softmax, 0 dense matmul, 0 hardware division in CUDA):**
   - `butterfly-pilot/src/butterfly_kernels.cu` compiles AOT (`nvcc -cubin`, `CUDA_DISABLE_PTX_JIT=1`, `0` PTX/SASS float ops, `0` PTX/SASS MMA ops) implementing bit-identical GPU mirrors of `fwht_inplace_q15`, `LiftingButterfly`, `ButterflyLinear`, `S³` holographic quaternion bind/transport/unbind, `rmsnorm_q15` (binary restoring `isqrt_u128` + division-free integer divide), `relu2_q15`, `SparsePkmMemory`, factored `64×64` head (`HuiBelkinRescaledSquare` & `Sparsemax64`), exact reverse-mode adjoints (with reversible checkpointing), NITI + Q15/Q12-matched Integer Lion + `splitmix64` stochastic rounding, and Q48 PRLL scoring.
3. **Exact transition replay oracle:** Gate 2 verifies `0` raw-word diffs between CUDA and CPU `uor_r4_native_vector::butterfly_rotor` across forward step outputs, carry words, PKM slot addresses, `i64` gradient accumulators, post-Lion `i16` weights, `UORBFY01` cold reload, and M1 CPU/WASM32 decode.
4. **Direct persistent-state training:** Trains `i16` Q15 weights directly on GPU on Gate 0's `train-100m` (`100.5M` clean Protocol-2 D87 tokens; Epoch 1 full causal LM, Epochs 2–3 answer-only `mask == 1`) with `0` float master weights.
5. **Lossless serialization:** Writes trained `UORBFY01` artifacts directly from persistent `i16` state and reloads them cold for unwrapped `E1–E4`, `chat-v0-p2`, and M1 `tok/s` measurement (`DEV-test` stays sealed/untouched).

## Scope

- `crates/uor-r4-native-vector/src/butterfly_rotor.rs` (orthogonal `SO(2)` Givens lifting `UORBFY01 v2`, rational Hui-Belkin Bayes posterior, `24`-bit NITI)
- `docs/loop/tasks/end-to-end-integration/antigravity-3/**`
- `docs/roadmap/items/end-to-end-integration.md` (progress note)
- `docs/research/notebook/m4-serving.md` (append-only row)

## Cancel Probe

Confirm Gate 1 PTX/SASS audit has `0` float and `0` MMA ops and Gate 2 finite GPU-vs-CPU parity passes `0` diffs across forward, backward, Lion update, and PRLL scoring before launching any multi-epoch `train-100m` run. Obtain non-author `TEST FITNESS: FIT` on `SHEET.md` before GPU execution.

## Done Check

1. `SHEET.md` receives fresh-context non-author `TEST FITNESS: FIT` (`FIT.md` + `GO.md`).
2. Gate 1 PTX/SASS audit (`0` floats, `0` MMA) and Gate 2 finite GPU-vs-CPU parity (`0` diffs at `D=32` and `D=256`, both loss arms) pass.
3. Controlled Gate 1 arms (`hb_l4_d256`, `hb_l8_d256`, `sp_l8_d256`) train on `train-100m` (`DEV-test` untouched), evaluate strictly unwrapped on `chat-v0-p2`, Gate 0 `E1–E4` (`DEV-tune`, aligned with PR `#4316` `hit_eos`/terminal answer checks), and Apple M1 native `tok/s` + RSS + `0`-diff CUDA=M1 replay.
4. Sealed receipts (`receipts.sha256`), `REPORT.md` (negatives first), `uor-harness loop pr-check --local`, fresh-context exact-head review, and CI pass.
