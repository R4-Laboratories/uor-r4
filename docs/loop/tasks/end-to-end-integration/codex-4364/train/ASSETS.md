# Source and input pins

The old SHA-256 and cubin identity come from #4361's physical receipt under `docs/loop/tasks/end-to-end-integration/codex-4361/evidence/finite.json` at commit `4101e33d1e9b66d6c2020cb9a0b2eef9e680d5cc`.

| Item | SHA-256 |
|---|---|
| Original cuda_backend.rs | 9822da970dc304fd0d13979667d515475a75b27e0e527fcbf845218500ac2ad2 |
| Original butterfly_kernels_v3.cu | d703e9b224bdf005e447ca318a330eaccff299b768511a331593443fab69f939 |
| Original #4361 physical finite sm_89 cubin | d7594dab97fd490bee2cad35d595a12054f64b2cdebe1eb628ac96ac8f56328a |
| Frozen native parent at step 3333 | 20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589 |
| Four-document TRAIN roster | 700c0bd3e78bd2dd8c0644d0abaea3e265538205eec20f69b9da0ccddca1ba92 |
| Wick CUDA 13.1 preparation candidate sm_89 cubin | 9b8185fd00e3d7896569476b1160102dc0305ddc24fe29d63cb0bd231113d533 |

The original native-vector Cargo manifest and model source were matched on both preparation hosts. Their SHA-256 digests are `9fdddda1f2e9de2585c7be3a9ef9f207711b8c51a5f32ab0a9336b729b3f77bc` and `83651cdde6f28012fecf96b4ad5404e736171db5fca6108773a6fd06a3e22c5f`. Reuse the repository's native-vector crate and vendored cudarc rather than packaging them with the checker.
