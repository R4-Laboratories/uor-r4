# Verify both frozen models on the ordinary integer client

Build on Wick from the reviewed source checkout.

```sh
cargo build --locked --release -p uor-r4-integer \
  --features p1-count-serving,release-factorial-diagnostic \
  --bin verify-native-release-factorial --bin serve-native-source-choice
cargo test --locked --release -p uor-r4-integer \
  --features release-factorial-diagnostic --lib release_factorial
```

For each registered split and each frozen model, run `product/replay.sh RUN_ROOT MODEL SPLIT PANEL FORWARD REVERSE`. Set `FACTORIAL_PRODUCT_BIN` to the directory with both binaries when building outside the Wick staging path. The runner defaults to the Wick paths for the balanced and linked checkpoints, and `FACTORIAL_BALANCED_ARTIFACT` and `FACTORIAL_CANDIDATE_ARTIFACT` can point to independently restored copies of the same files. The script verifies each file's pinned digest. `PANEL` is the original JSONL tape and `FORWARD` and `REVERSE` are the matching device observations in their original execution orders. These checks consume `fresh`, archived DEV and HELD, original protected count controls, and prior DEV regression tapes independently.

The ordinary client gets the prompt IDs and four ordered physical positions in a fresh process. It emits no diagnostic fields and receives no expected reply. The verifier loads the D11 independently. Its four arms feed each committed ID back into the frozen writer. Only the authenticated A-period and the authenticated requested B with the exact emitted prefix can enable a supplied action. A supplied action selects EOS or requests the next physical position through the frozen verifier. The stage after C uses the learned action. Rows preserve the prompt, positions, requested and original IDs, learned scores, both actions, native read, stage, direct commit and terminal state. The verifier checks every JSON field against forward and reverse device records and checks every ordinary writer field against the client. `SHA256SUMS` is written only when both comparisons and input digest rechecks pass.
