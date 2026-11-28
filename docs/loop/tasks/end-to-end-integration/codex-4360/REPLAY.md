# Replaying the conversation and carry comparison

The executable source was frozen at d71296428615755fcdf9c8be55a712fe4e966a2a before the rental. See the [cohort recipe](cohort/README.md) for article exclusions, overlap-screen limits, tokenizer, compiled dialogue and independent rebuild. [The physical dispatch](dispatch4360.sh) consumes only files with the [registered digests](dispatch.sha256). The historical native CUDA include is the original antigravity-5 butterfly_kernels_v3.cu, SHA-256 d703e9b224bdf005e447ca318a330eaccff299b768511a331593443fab69f939. Build from repository root:

```sh
unit=docs/loop/tasks/end-to-end-integration/codex-4360
gate=docs/loop/tasks/end-to-end-integration/antigravity-5/butterfly-v3/src
nvcc -cubin -arch=sm_89 -O3 -I "$gate" "$unit/runner/device.cu" -o probe-final.sm89.cubin
CARGO_TARGET_DIR="$PWD/local/compact-carry-target" cargo build --release --locked --manifest-path "$unit/runner/Cargo.toml"
```

The Wick ZFS run directory is `/mnt/plexpool/uor-r4-corpus/runs/20261128-codex-compact-human-carry`. A copy of the measurement packet at the [immutable private revision](evidence/restore.json) can also be checked after extraction with `sha256sum --strict -c PACKET.sha256`. The historic fixture, compiled executables, device receipts, complete raw paired outputs and unmodified private references are inside. Never post the pair tape or raw rows to the public repository. From the restored packet, recount without stepping a model:

```sh
run/reconcile assets/model.uorbfy03 assets/pairs.jsonl received/attempt01/rows.jsonl parity-wick/ordinary.jsonl > reproduced.json
cmp reproduced.json recount.json
```

For a cold ordinary serving replay on Wick, run the ordinary executable in the packet with these arguments. Compare its output with the saved parity file by SHA-256:

```sh
run/ordinary assets/model.uorbfy03 20262de550c2a96a9cc87a7fecfd178567a11cf58dbd6ad2cb1ffc3e5bddb589 assets/six.jsonl c07b07c77b1649b944302c3e9fcaa06494d0420294fefc6d2b41db30c738d7c6 1 > cold.jsonl
sha256sum cold.jsonl parity-wick/ordinary.jsonl
```

A new physical run requires a fresh attempt directory and an owned sm_89 lease. Execute the pinned dispatcher from its declared layout. Its first three commands verify the finite controls, the 2,048-input historical trajectory and the complete six-pair CPU/GPU differential before evaluating all 64 pairs. The per-reply archive exposes the first transition and every emitted ID in all three arms. The separate parity receipt commits to all compared transitions. The ordinary probe calls the same typed V3 step as the native reference and takes token IDs as input. It does not import UORBFY03 through the product text client.


The supplementary exposure and decoded-byte checks read existing records without executing the model. For the exposure check, verify the three TRAIN digests at the top of [the receipt](evidence/train-exposure.txt), then run the standalone [census](runner/src/bin/train_exposure.rs) with the Gate 0 corpus directory as its argument. Compare its output with the receipt after the three SHA lines. For the decoded comparison, use the cohort package's [decoded binary](cohort/src/bin/decoded.rs) with the arguments in its usage message. Pin the tokenizer to a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 and the GPU rows to d4765effc6322f0969700fb6a9d1a9d1d350200228bec4d1bc3b41e2352fda74 before running. Byte comparisons preserve case and punctuation and trim ASCII whitespace at the boundaries. These supplementary checks were added after the physical run and read its fixed evidence.
