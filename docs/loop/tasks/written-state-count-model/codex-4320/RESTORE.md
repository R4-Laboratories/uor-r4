# Restore the post-period trace

The [pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/16a7a33507bbe832e85889bfac19953adb4d7692/results/codex/postperiod-completion-20261127/postperiod.tar.zst) is 2,243,410 bytes, SHA-256 `f7631f02d15afe336b510fd275a2cea487603b9d5d0730d0661b9cb2d30b8ba4`. Wick holds the original in `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-postperiod`. The packet carries the panel, reference keys, compiled diagnostic, finite calibration and parity, qualified GPU and cold Rust events, ordinary client checks and both independent analyses. Obtain the unchanged D11 with the [#4319 model restore recipe](../codex-4319/RESTORE.md). The exact D11 SHA-256 is `93147372d495b36fda03fa0115a983e75bf1fd1b132d826212b4e1e0e6a1e89a`.

Extract the packet into an empty directory and verify the source manifest before scoring. Use `UNIT` for this task directory, `REPO` for the restored source checkout, and `ARTIFACT` for the #4319 integer model.

```sh
tar --zstd -xf postperiod.tar.zst
(cd measurement-v1 && sha256sum --strict -c OUTPUTS.sha256)
(cd panel-v1 && sha256sum -c SHA256SUMS && cd held && sha256sum -c SHA256SUMS)
cargo build --locked --release --manifest-path "$REPO/Cargo.toml" -p uor-r4-integer \
  --features p1-count-serving,postperiod-diagnostic --bin verify-native-postperiod
"$REPO/target/release/verify-native-postperiod" "$ARTIFACT" panel-v1/held/rows.jsonl \
  measurement-v1/measured/observations.jsonl > restored.jsonl
cmp restored.jsonl product-qualified-v1/cold.jsonl
jq -n --slurpfile p panel-v1/held/rows.jsonl \
  --slurpfile t measurement-v1/measured/observations.jsonl -f "$UNIT/audit.jq"
jq -s -f "$UNIT/trace.jq" measurement-v1/measured/observations.jsonl
jq -s -f "$UNIT/product/distill.jq" restored.jsonl
```

The archive was downloaded again at its pinned revision, compared byte-for-byte, unpacked into a new Wick directory and replayed. [Download and replay hashes](evidence/storage-readback.sha256) · [Panel and decoded scorer](panel/README.md) · [Build the SM89 instrumentation](device/README.md)
