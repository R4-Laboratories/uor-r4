# Non-Author Test Fitness Review (`antigravity-4`)

- **Verdict:** `TEST FITNESS: FIT` (Round 3 Amendment)
- **Lab:** `antigravity`
- **Reviewer:** Fresh-context non-author `pro` research subagent (`98ff6b8f-636a-4088-a0b2-3b60719b36b5`), Rounds 1–3 (`2026-11-28T02:27Z` -> `2026-11-28T02:56Z`)
- **Charter Basis:** `CHARTER.md` §4.5 (pre-registration fitness review prior to `train-100m` execution)

## Round 1 -> Round 3 Defects Found & Resolved

1. **Q24 Simplex Conservation in Diagnostic F32 (`diag_f32_kernels.cu:410-456`):** Fixed `float sum` drift in `scores_64_to_q24_probs_hb_f` to `double raw[HIER_COARSE]` + `double sum` and explicit handling of both `used < Q24_ONE` and `used > Q24_ONE` (preserving `out[i] >= 1` for every entry so `sum(out) == Q24_ONE` identically).
2. **Matched AdamW `eps` & Gradient Clamp (`cuda_backend.rs`, `diag_f32_kernels.cu`, `main.rs`, `dispatch.sh`):** Replaced hardcoded `1e-6f` in `CudaDiagF32Engine` with `eps = (eps_q15.max(1) as f32) / 32768.0f32` (`--eps 16` -> `4.8828125e-4`, exact match to Pure-Int `--eps 16`) and added `±4096.0f` gradient clamp matching Q15 `±2^27`.
3. **Exhaustive Ordered §4 Decision Table (`SHEET.md:40,59-64`):** Replaced non-exhaustive thresholds with mutually exclusive, 100%-exhaustive rules `(a)–(d)` over matched twins `(Arm 1, Arm 3)` and `(Arm 2, Arm 4)`.
4. **Hardened `dispatch.sh` Gate 1a/1b & Safety (`dispatch.sh`):** Added `L-002` overwrite guard, `/usr/local/cuda/bin` on `PATH`, clean rebuild (`cargo test --bins && cargo build --release --features cuda`) before auditing the exact embedded `.cubin`s, PTX comment/metadata stripping, and full strict `antigravity-3` Gate 1a/1b regexes (`+ wgmma`, `tcgen05.mma`, `WMMA`, `UTCMMA`).
5. **Round 3 Exact `tangent_pullback` Clamp-Boundary Derivative (`butterfly_rotor.rs`, `butterfly_kernels.cu`, `diag_f32_kernels.cu`):** Gate 2's central finite-difference gradient check caught missing `[-1, 1]` (`[-Q15_ONE, Q15_ONE]`) clamp-boundary masking in `tangent_pullback` (`fd_cos = 0.97678` -> `1.000`). Added the exact `0` derivative mask when `|raw[b + c]| >= Q15_ONE` (`>= 1.0f`) identically across Rust, Pure-Int CUDA, and Diag F32 CUDA (zero forward or `UORBFY01 v2` change), and removed dead `parse_loss_kind` from `main.rs`. Dispatched into fresh `L-002` output dir `/workspace/20261128-antigravity-gate1-diag-r2`.

## Audited Exact SHA-256 Digests (`TEST FITNESS: FIT`, Round 3)

| File | Exact SHA-256 |
|---|---|
| `crates/uor-r4-native-vector/src/butterfly_rotor.rs` | `649a7d3cbd2501449c7a4c813357440b71b6a81751ef91b0d26682a0633121f3` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/ORDER.md` | `4fd65c29642f3ae165aa494ce0bbbafe7815975468f6357aad9d4c473f6106ff` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/SHEET.md` | `1b205a0455f77119e795beee221853884dcfdb31a23fbf4456343792fe973bde` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/dispatch.sh` | `6c7947ff10c2bb75013f2bdd559b8b6bc5df4088fed86b0c2dfdbf05b480cdc3` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/Cargo.toml` | `3436a6b9d04a7ba709221081e46cf4e5d7a073cfda1fe23aecc1af990f76935a` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/build.rs` | `d026ad4534b5e3b76ca28cf42e4421afbb4c067352f04e38a9cc005824213aca` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/src/butterfly_kernels.cu` | `e0bf06dec7125a8b654e3ab76ee2a45a49da5eca74dce02b8795808def54cc9d` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/src/diag_f32_kernels.cu` | `8ea3c0e5bc56b28bb33c569ea0761b8f07c548af8aa63015977c510eac0fd1a0` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/src/cuda_backend.rs` | `ec3ebeb5e60697de6d97984a52b5bbb4b99f62a972c41c4a6b8346be63b57378` |
| `docs/loop/tasks/end-to-end-integration/antigravity-4/butterfly-diag/src/main.rs` | `f8bd87b842da71f3be50cfd52a75d88aa20002b39fbfce7eb0025d83517e6a93` |
