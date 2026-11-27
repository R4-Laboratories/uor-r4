# Restore the trained release action and its exact replies

The [private evidence packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/114e7dfcbc6ebf46e319c5b2ca3647b103a1226a/results/codex/native-source-choice-20261127/source-choice.tar.zst) is 2,231,626 bytes, SHA-256 `f862c76d506fcef1de30c29ce91ab0831b0abc84979d135f8785fb9ed0edd0ab`. The full checkpoint resides on Wick at `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-source-choice/product-qualified-v1/artifact.d11`. The packet stores the learned words and fit receipt needed to reproduce it from the [pinned #4319 base](../codex-4319/RESTORE.md), together with the tapes, GPU records, client traces, finite controls and grading outputs.

Run in a new directory with `REPO` set to the source checkout, `BASE` to the recovered #4319 artifact, and `UNIT` to this experiment directory.

```sh
tar --zstd -xf source-choice.tar.zst
(cd measurement-v1 && sha256sum --quiet --strict -c OUTPUTS.sha256)
(cd panel-v1 && sha256sum -c SHA256SUMS && for split in train finite held-primary held-secondary; do (cd "$split" && sha256sum -c SHA256SUMS); done)
cargo build --locked --release --manifest-path "$REPO/Cargo.toml" -p uor-r4-integer \
  --features p1-count-serving --bin pack-native-source-choice --bin verify-native-source-choice \
  --bin serve-native-source-choice
BIN="$REPO/target/release"
FIT=$(sha256sum measurement-v1/fit/train.json | cut -d' ' -f1)
"$BIN/pack-native-source-choice" "$BASE" measurement-v1/fit/weights.i16le restored.d11 "$FIT"
sha256sum restored.d11
"$BIN/verify-native-source-choice" restored.d11 panel-joined-v1/rows.jsonl \
  measurement-v1/measured/observations.jsonl > replay.jsonl
cmp replay.jsonl product-qualified-v1/cold.jsonl
"$BIN/serve-native-source-choice" restored.d11 < product-qualified-v1/requests.jsonl > client.jsonl
cmp client.jsonl product-qualified-v1/client.jsonl
jq -n --slurpfile p panel-joined-v1/rows.jsonl --slurpfile t replay.jsonl -f "$UNIT/audit.jq"
jq -s -f "$UNIT/pairs.jq" replay.jsonl
```

The trained file is 1,202,177,144 bytes with SHA-256 `634784e0d1e46e905b36ff96677ad5f18a3c6ce78a17bcd9d1680b4a62c77dc7`. A clean download at the pinned revision reproduced both the checkpoint and every event in the cold trace. [Verified readback hashes](evidence/storage-readback.sha256)
