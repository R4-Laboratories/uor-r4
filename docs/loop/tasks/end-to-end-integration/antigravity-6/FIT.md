# Non-Author Test Fitness Review (`FIT.md`) — `antigravity-6`

- **Lab:** `antigravity`
- **Reviewer:** Fresh-context non-author `pro` subagent (`20c43115-14b9-4796-bd9b-bc4affa9ccb3`)
- **Date:** 2026-11-28
- **Round 1 Verdict:** `TEST FITNESS: NOT FIT` (2 blocking + 1 non-blocking doc count fix)
  - Blocking #1: `SHEET.md` §4 Rules `(b)` and `(d)` weakened Gate 1 `E1_nov`/`E3` and had an unwrapped E2 EOS-spam / empty-response loophole -> fixed by restoring full single-artifact Gate 1 bar in `(b)` and binding non-degenerate E2 + 95% paired CI + conditioning checks in `(c)/(d)`.
  - Blocking #2: `ButterflyV3Prepared::step_greedy` still allocated one `Vec<i64>` per token inside `fwht_inplace_q15` -> fixed by extracting `fwht_inplace_q15_with_scratch` in `butterfly_rotor.rs` and preallocating `fwht_work: Vec<i64>` in `ButterflyV3FastSession` (`0` steady-state heap allocations per token).
  - Non-Blocking #3: Reconciled `ORDER.md` (amended alone as Commit #1 `524899a97`, `3,946 B <= 4,096 B`) and `SHEET.md` to exact `6,863,360` params / `13,734,980 B` (`13.73 MB`, matching `dispatch.sh:8`).
- **Round 2 Final Verdict:** **`TEST FITNESS: FIT`**

## Audited File SHA-256 Digests (`10/10` exact match)

- `dd96e65af704dfa6aa0d3432ce4210c39b17435137a2ef74f715a3b9e0dcac30  crates/uor-r4-native-vector/src/butterfly_rotor.rs`
- `48a0f9bbb2ce30c11de40861cdbaf14608e3a73b1590a32bb3141c5b58e24d9f  crates/uor-r4-native-vector/src/butterfly_rotor_v3.rs`
- `615561fb9135c49b330d5d40bd4bd01bcd6762e603e636db04505c67536d00ad  docs/loop/tasks/end-to-end-integration/antigravity-6/ORDER.md`
- `a9b051ec5a8f6f94545a73c4b9c44965df01b33e77a3f2c234235641bdfdc10e  docs/loop/tasks/end-to-end-integration/antigravity-6/SHEET.md`
- `edd738e45018e4b97a592355905a3afa381f96445ef3602727b0e292075db29d  docs/loop/tasks/end-to-end-integration/antigravity-6/dispatch.sh`
- `4809459488113ad6ad966d67dcc0c852140f033fa6118fd600abb32d80b63a69  docs/loop/tasks/end-to-end-integration/antigravity-6/butterfly-v3b/Cargo.toml`
- `0b406a1c113daff551744efe1a54ed14ad8e7ae083c36a562831bda258ba09e9  docs/loop/tasks/end-to-end-integration/antigravity-6/butterfly-v3b/build.rs`
- `d54056bd51a3444bf121cb3a50425404df7cdb91e423745d4f487f8148333f0d  docs/loop/tasks/end-to-end-integration/antigravity-6/butterfly-v3b/src/butterfly_kernels_v3.cu`
- `9822da970dc304fd0d13979667d515475a75b27e0e527fcbf845218500ac2ad2  docs/loop/tasks/end-to-end-integration/antigravity-6/butterfly-v3b/src/cuda_backend.rs`
- `70b1eae907f10a2daf35a43026d08f34e06e4e226c8ff8faa2598a0afade9316  docs/loop/tasks/end-to-end-integration/antigravity-6/butterfly-v3b/src/main.rs`

## Verified Compliance Summary
- `CHARTER.md` §3.1–§3.7 & D85: `0` float/exp/log/softmax/squashing/matmul/div in CUDA/CPU/WASM default paths; `Entmax15` bit-identical across Rust and CUDA; `SAFE_NORM2 = (ACT_CLAMP >> 1)^2` mathematically tight; `0` steady-state heap allocations in `ButterflyV3Prepared::step_greedy`.
- Resource bounds: `< 2.65 GB` VRAM per GPU (`32,168 B` shared memory), `<= 2.50 $/h` on `2 × RTX 5090`, sealed `dev-test.*` never opened.
- Independent static checks passed: `check_no_softmax.py` (`OK`), `check_no_matmul.py` (`OK`), `bash -n dispatch.sh` (`OK`).
