# Compact V3 grounded training with matched reset control

This package implements the two warm starts in [ORDER.md](../ORDER.md) and [SHEET.md](../SHEET.md). Install it at `docs/loop/tasks/end-to-end-integration/codex-4361/train` in the isolated project tree. The backend and CUDA file are included byte-for-byte from antigravity-5 and verified against the registered SHA-256 identities. This package reads only the frozen Arm 2 artifact and TRAIN.

## CPU-only validation on Wick

```sh
cargo test --offline --locked --manifest-path docs/loop/tasks/end-to-end-integration/codex-4361/train/Cargo.toml
cargo build --offline --locked --release --manifest-path docs/loop/tasks/end-to-end-integration/codex-4361/train/Cargo.toml
```

Use the resulting `target/release/grounded-reset-4361-train`, or the corresponding binary under the configured `CARGO_TARGET_DIR`.

```sh
train_sha=f96cc65ed0fd76c6cb9f75b5434ef6a0d7edc65d42ec01a4145781e2d2a94f4a
train_tape=/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-grounded-reset/cohort/complete-reservation/final/train.jsonl
"$runner" --validate "$train_tape" "$train_sha"
```

Validation checks the registered digest, exact tape size and group-distinct ID uniqueness, structural third-response framing, window containment of all target IDs, target order equality, and transition accounting. Article-level source identity and source separation are certified by the frozen cohort receipt. The validation output contains aggregate counts only. The validated tape has 6 near, 234 far, and 16 intermediate documents, with 2,151 supervised targets across 170,423 consumed transitions per arm. Reset uses 169,892 semantic transitions and 531 isolated padding transitions.

## Physical calibration and training

On the exclusive RTX 4090 lease, compile with CUDA nvcc and keep PTX JIT disabled. Run the original integer instruction audit on the embedded cubin under the registered source and toolchain before training. The kernel source and native model implementation are sourced from the pinned repository, and the finite receipt binds the embedded cubin and actual device UUID.

```sh
export CUDA_DISABLE_PTX_JIT=1
export UOR_NVCC_ARCH=compute_89
cargo build --release --locked --features cuda --manifest-path docs/loop/tasks/end-to-end-integration/codex-4361/train/Cargo.toml

"$runner" --finite "$private_output/fresh-finite.json"
"$runner" --train "$arm2_artifact" "$train_tape" "$train_sha" "$private_output/fresh-training-directory" "$private_output/fresh-finite.json"
```

Use the CUDA-feature binary for both commands. Release the second command only after the physical finite receipt and registered decision review. Output files and the new training directory are created exclusively. Use a new root for any retry. A training root containing snapshots and no `COMPLETE.json` remains an incomplete attempt.

The finite mode checks zero-update lossless reload, identical short-row shams, a planted response past the actual 257-ID boundary, and an independent Rust gradient and AdamW update oracle. It reads the GPU-saved forward carry and normalization words used as shift history, and compares the GPU serving greedy IDs and Q48 masses with the Rust scorer before and after an update. It checks that the reset cannot backpropagate to a tied embedding row used exclusively before the cut and that padding leaves raw gradients and weights/moments unchanged. It then runs a synthetic Arm 2-shaped capacity check with four 1,087-transition documents and 59 labels per document. The receipt records elapsed time per arm and device memory readings. Monitor GPU memory independently during this finite run to record transient peak use. The two engines' persistent base weights, gradients, moments, and activation buffers total approximately 845 MiB, plus metadata, streams, allocations, and CUDA context overhead. Use the physical pilot's per-arm wall time to budget the real 64 updates.

## Update and output semantics

- Both arms start from Arm 2 checkpoint SHA `20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589`, serialized step 3,333, with independently allocated, zero-filled integer optimizer moments.
- Every update reads the same four consecutive documents and ordered target IDs. The full arm runs whole documents. The reset arm launches every disjoint 257-ID window, including prompt-only windows, with independent forward and backward state. The native integer engine adds raw target-sum gradients across launches. Both arms pass the same post-increment step, 3,334 to 3,397, into integer AdamW. There are no per-chunk optimizer applications or chunk-count denominators.
- The reset arm schedules one isolated two-ID, zero-mask row for each historically omitted cut transition, from a fresh state. It records this work separately. Padding runs before the semantic gradient reset. Moments and weights are read-only for that launch, and the discarded padding gradient buffer cannot reach an optimizer step. The physical finite mode verifies raw gradient equality before and after padding as an independent control.
- AdamW uses fixed LR_Q8 512, weight decay shift 19, epsilon 16, sparse idle skip, and trainable PKM keys. The original permutation, geometry, loss, and state representation remain bound to the starting checkpoint.
- The runner logs native loss sums, actual checked target counts, input accounting, and duration for all 64 updates. It writes only the two final UORBFY03 checkpoint artifacts, `full.uorbfy03` and `reset.uorbfy03`, alongside `updates.jsonl` and the hash-binding `COMPLETE.json`. Both final artifacts pass lossless Rust codec readback. The receipt separately identifies whether the native parameter payload changed, independent of the updated serialized step counter.
