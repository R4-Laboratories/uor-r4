# Restore the four first-period fits

The [private evidence packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/4b1566d8b293c308e5666ea76efdab4a4947602a/results/codex/first-period-margin-20261127/period-margin.tar.zst) contains 7,492,125 bytes with SHA-256 `7449cc2ae1880997c264d8136acd27ac58bd790035e44a1605bc214de16b7cdd`. The four reconstructed artifacts are also on Wick under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-period-margin/cold-v1/{baseline,balanced,weak,both}/artifact.d11`. The finalist’s native release weights have SHA-256 `45d517f31074aa8061307a36f5d522e3af4bb464b434e5e6ff1779f4dd82c05e`.

The archive contains the four native weight files, fit receipts, frozen DEV and protected tapes, both directions of every GPU evaluation, exactness grades, replay outputs and the rejected compiler output. It also preserves the sealed HELD tape. The [parent experiment](../codex-4321/RESTORE.md) supplies the original TRAIN rows, and the [base restore](../codex-4319/RESTORE.md) supplies the frozen D11. Each new D11 embeds its own fit receipt digest. The baseline model words reproduce the parent byte for byte.

Download the packet at the revision above and extract it in an empty directory on Wick. Set `BASE` to the restored #4319 D11 and `REPO` to the verified source checkout.

```sh
tar --zstd -xf period-margin.tar.zst
(cd phase1-v1 && sha256sum --quiet --strict -c OUTPUTS.sha256)
(cd panel-v1 && sha256sum --quiet --strict -c SHA256SUMS)
for split in dev protect held; do (cd "panel-v1/$split" && sha256sum --quiet --strict -c SHA256SUMS); done
cargo build --locked --release --manifest-path "$REPO/Cargo.toml" -p uor-r4-integer \
  --features p1-count-serving --bin pack-native-source-choice --bin verify-native-source-choice \
  --bin serve-native-source-choice
bin="$REPO/target/release"
for arm in baseline balanced weak both; do
  fit="phase1-v1/output/$arm/fit"
  trace="phase1-v1/output/$arm/dev/observations.jsonl"
  "$bin/pack-native-source-choice" "$BASE" "$fit/weights.i16le" "$arm.d11" "$(sha256sum "$fit/train.json" | cut -d' ' -f1)"
  test "$(sha256sum "$arm.d11" | cut -d' ' -f1)" = "$(jq -r .artifact_sha256 "cold-v1/$arm/artifact.json")"
  "$bin/verify-native-source-choice" "$arm.d11" panel-v1/dev/rows.jsonl "$trace" > "$arm.dev.jsonl"
  cmp "$arm.dev.jsonl" "cold-v1/$arm/dev/cold.jsonl"
  "$bin/serve-native-source-choice" "$arm.d11" < "cold-v1/$arm/dev/requests.jsonl" > "$arm.client.jsonl"
  cmp "$arm.client.jsonl" "cold-v1/$arm/dev/client.jsonl"
done
```

All four artifacts and their DEV device replays were rebuilt from the freshly downloaded private packet. [Readback digests](evidence/storage-readback.sha256)
