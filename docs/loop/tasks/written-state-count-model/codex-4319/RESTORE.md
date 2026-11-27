# Restore the word-ending experiment

Download the [pinned private evidence packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/bf2726d5364020e2b30be343d2f482824ba1232c/results/codex/native-word-ending-20261127/word-ending.tar.zst). Its length is 995,900 bytes and SHA-256 is `c61be1faae720ac1a998edc867648072c54bef1adb0c7671c920706256f3a3c6`. Extract it with `tar --zstd -xf word-ending.tar.zst` in a new directory. The Wick original is in `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-word-ending`. [Restored model and cold output digests](evidence/storage-readback.sha256)

The ordinary model packs the frozen #4315 native D11 byte-for-byte with the trained 4,096 signed i16 words and a reference to the fit receipt. Restore the original D11 and tokenizer using the [parent recipe](../codex-4315/RESTORE.md). The copy implementation and its original-ID/address behavior are pinned by [#4318](../codex-copy-id/REPORT.md).

Build the native tools and set `BASE` to the verified #4315 artifact. From a fresh extraction of the evidence packet, repack the trained state and check the actual predictions against the GPU.

```sh
cargo build --manifest-path "$REPO/Cargo.toml" --release --locked -p uor-r4-integer \
  --features p1-count-serving --bin pack-native-word-ending --bin verify-native-word-ending --bin uor-chat
BIN="$REPO/target/release"
FIT_SHA=$(sha256sum measurement-v1/fit/train.json | cut -d' ' -f1)
"$BIN/pack-native-word-ending" "$BASE" measurement-v1/fit/weights.i16le restored.d11 "$FIT_SHA"
"$BIN/verify-native-word-ending" restored.d11 panel-v1/held/rows.jsonl \
  measurement-v1/measured/observations.jsonl > replay.jsonl
cmp replay.jsonl product-qualified-v1/cold.jsonl
```

The panel includes each raw prefix, independent reference through natural EOS, supplied factual source address, fact and instruction family, and decoded conversation. The output observations preserve the original source commit, native proposal, chosen instruction and actual native third event for every arm. The third step is present only when the second step emitted a non-EOS ID. The [independent trace audit](audit.jq) works on the frozen panel and raw observations:

```sh
jq -n --slurpfile p panel-v1/held/rows.jsonl \
  --slurpfile t measurement-v1/measured/observations.jsonl -f "$UNIT/audit.jq"
jq -n --slurpfile obs measurement-v1/measured/observations.jsonl -f "$UNIT/wording.jq"
```
