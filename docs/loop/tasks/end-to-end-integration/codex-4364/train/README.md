# Native opposed opening pilot checker

This crate extends the #4361 native integer training path using the source-pinned unified patch in this directory. It compiles the historical optimizer and writer as part of the instrument, and loads the archived original #4361 cubin from a hash-checked file for the control and authentic ordinary replay. Loss gradients apply only at the declared first-answer label. The gold-cell fine term is applied independently of coarse admission, and ordinary fine election remains gated by the deployed coarse choice.

Run the preparation from the repository root on the approved build host with nvcc available. The build uses Rust 1.97.1 and sm_89 CUDA machine code. The owner-approved preparation used CUDA 13.1 for the new cubin. Match the registered architecture and audit the exact locally built cubin with the #4361 opcode scanner before any dispatch.

```sh
task=docs/loop/tasks/end-to-end-integration/codex-4364/train
target_dir="$PWD/local/codex/opposed-opening/target"
CARGO_TARGET_DIR="$target_dir" cargo build --locked --release --features cuda --manifest-path "$task/Cargo.toml"
checker="$target_dir/release/grounded-opposed-opening-update"
```

The machine-code path is `release/build/grounded-opposed-opening-update-*/out/checker.sm_89.cubin` beneath the fresh target directory. Audit it using `cuobjdump --dump-sass` and `docs/loop/tasks/end-to-end-integration/codex-4361/sass-check.awk` from the pinned #4361 checkout. Capture compiler identities, source and cubin hashes, Linux executable hash, the opcode-audit receipt, device identity and launch environment in the dispatch receipt.

## Separate physical entries

```sh
CUDA_DISABLE_PTX_JIT=1 "$checker" finite "$historical_cubin" > "$finite_receipt"
CUDA_DISABLE_PTX_JIT=1 "$checker" capacity "$parent" "$historical_cubin" > "$capacity_receipt"
CUDA_DISABLE_PTX_JIT=1 "$checker" one-update "$parent" "$roster" "$finite_receipt" "$historical_cubin" "$new_output"
```

The first command runs only the planted 512-wide, six-layer fixture. It compares the original ABI and zero-gain sham with the finite Rust reference, requires both heads to carry registered-constant derivative movement, checks an amplified directional witness, tests masked BOS and invalid opening controls, and compares a reloaded checkpoint through ordinary scoring. The stdout JSON is written only on a passing check. Keep refusals and partial receipts separately.

The capacity command is a supplemental finite differential. It uses only the pinned step-3333 parent, its actual vocabulary permutation and four deterministically generated synthetic token rows with 3,938 transitions and 167 labels. It loads the original and patched training cubins on the same SM89 device. It compares the original and scanned zero-gain arms word for word across gradients, weights, first and second optimizer moments, and serialized step-3334 checkpoints. It checks forward workspace words and ordinary CUDA writer IDs and masses against both patched arms. The opening hidden and score rows are reconstructed from the matching saved pre-head words, checked against device opening traces and ordinary opening IDs and masses, and recorded as digests. The pinned training ABI exposes the layer-input, carry, shift-input and pre-head words, while individual PKM slot IDs remain unexported. Registered gain must produce a live derivative at both sites and change deployed weight words. The command emits a single passing JSON receipt only after all checks succeed.

The one-update command requires a fresh passing receipt on the same GPU UUID and cubin. It consumes the 4-document hash-pinned TRAIN roster and 167 positive labels, including EOS. The two authentic GPU states start at the same frozen checkpoint with identical zero initial moments. The original #4361 kernel supplies the control, and the active arm adds the opening term. Each arm takes one post-increment update. The authentic path uses Rust only for ordinary state replay and serialization. Both authentic endpoints use the historical ordinary CUDA writer. The output directory is newly created with private permissions and contains `sham.uorbfy03`, `active.uorbfy03`, and `report.json`. The report includes little-endian flat gradient, weight and moment hashes and distinct elected-cell and gold-cell fine diagnostics.

The build and opcode verification precede the independent physical FIT. Physical execution remains unverified until the run produces its receipts. The authentic roster, parent checkpoint, cubin, executable and output checkpoints are external custody inputs or generated private artifacts. None belongs in this source directory. Asset pins are listed in `ASSETS.md`. `SOURCES.sha256` covers the current source. `SOURCES-pilot.sha256` preserves the prior source pins for the staged pilot executable.
