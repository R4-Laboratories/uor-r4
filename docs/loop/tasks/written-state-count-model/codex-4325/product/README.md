# Pack and verify each integer fit on Wick

```sh
cargo build --locked --release -p uor-r4-integer \
  --features p1-count-serving,release-factorial-diagnostic \
  --bin pack-native-source-choice --bin serve-native-source-choice \
  --bin verify-native-source-choice
cargo test --locked --release -p uor-r4-integer \
  --features release-factorial-diagnostic --lib release_residual
```

Run `product/pack.sh RUN_ROOT ARM WEIGHTS.i16le FIT.json [BASE.d11]` for `first-period` and `joint`. The 49,158-byte input is the archived 40,966-byte linked state followed by 4,096 trained signed little-endian i16 words. The packer checks the frozen prefix, copies the exact state and frozen #4319 D11, binds the fit receipt, reloads the result, and publishes the artifact at `RUN_ROOT/product-v1/ARM/artifact.d11`. With an independently restored writer, supply its path as `BASE.d11`.

Run `product/replay.sh RUN_ROOT MODEL SPLIT PANEL.jsonl FORWARD.jsonl REVERSE.jsonl` for each model and split. Models are `first-period`, `joint`, `candidate` and `balanced`. `JOINT_PRODUCT_BIN` selects the build directory. `JOINT_CANDIDATE_ARTIFACT` and `JOINT_BALANCED_ARTIFACT` select restored copies of the two frozen controls. Their digests are pinned by the runner.

Each model and split starts a fresh ordinary serving process that reads only prompt IDs and ordered factual addresses. A second process independently reloads the artifact, follows each arm's actual committed IDs, and checks the complete transitions against device observations in forward and reverse order. The ordinary client must match every learned transition, including authenticated read positions, integer score, actual committed token and EOS or exhaustion. Factorial intervention fields and expected reply metadata stay in the verifier.

The completed output at `RUN_ROOT/product-replay-v1/MODEL/SPLIT` contains the inputs' digests, client requests and responses, complete cold observations, exact match counts, and elapsed process time and peak RSS for ordinary serving including model load. The runner publishes `SHA256SUMS` after every comparison and input recheck passes. A failed directory retains evidence for inspection. Create a new run root before retrying the same model and split.
