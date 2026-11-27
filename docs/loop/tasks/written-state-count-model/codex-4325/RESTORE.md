# Restore and replay both trained models

The [pinned training and replay packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/eb536a8c3421b58226fc51e452e831dafded3529/results/codex/joint-release-20261127/joint-release.tar.zst) occupies 8181947 bytes, SHA-256 `a5bb3087a0d4b53b1543083a2f292c1e94e49624f9d9bdfa86c355e2290bd554`. The private-store download reproduced both fitted artifacts and their complete fresh client and GPU-matched records.

The study root on Wick is `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-joint-release`. Its packet carries the fitted state bytes, frozen-prefix identities, signed update schedule, actual verified B trajectories for TRAIN, scientific observations in both directions, keys, original reference IDs, transfer comparisons and ordinary-client replay. Rebuilding from a stored fit preserves every i16 word. The scientific source was `a3852a3a8f1a83457bacf58c00789982aa7e05f7`. The newly packed states contain 49,158 bytes apiece.

Extract the revision-pinned packet in a new directory. Restore the [base writer D11](../codex-4319/RESTORE.md) and set `BASE` to its absolute file path. Build the binaries following the [product instructions](product/README.md) and point `JOINT_PRODUCT_BIN` to their directory. Reconstruct both D11 files and replay ordinary and four-arm diagnostic records on the full fresh holdout.

```sh
(cd measured-v1 && sha256sum --strict --quiet -c OUTPUTS.sha256)
(cd measured-v1/raw/fit && sha256sum --strict --quiet -c sealed-weights.sha256)
for s in train held; do (cd "panel-v1/$s" && sha256sum --strict --quiet -c SHA256SUMS); done
for model in first-period joint; do
  dest="$PWD/new-$model"
  mkdir "$dest"
  bash staging/product/pack.sh "$dest" "$model" \
    "measured-v1/raw/fit/$model.i16le" measured-v1/raw/fit/fit.json "$BASE"
  test "$(jq -r .artifact_sha256 "$dest/product-v1/$model/artifact.json")" = \
    "$(jq -r .artifact_sha256 "product-v1/$model/artifact.json")"
  gpu="$PWD/measured-v1/raw/held/trained/$model"
  bash staging/product/replay.sh "$dest" "$model" held "$PWD/panel-v1/held/rows.jsonl" \
    "$gpu.jsonl" "$gpu-reverse.jsonl"
  for f in client.jsonl cold.jsonl cold-receipt.json; do
    cmp "$dest/product-replay-v1/$model/held/$f" "product-replay-v1/$model/held/$f"
  done
done
```

The [previous release experiment](../codex-4324/RESTORE.md) supplies the historical balanced/linked measurements for the five regression panels, with original queries and joins supplied by the [scope](../codex-4323/RESTORE.md) and [count](../codex-4322/RESTORE.md) packets. New output records in `grade-v1/SPLIT/MODEL.rows.jsonl` retain each model's exact first divergent ID, stage scores, terminal state and original addresses; `comparison.json` records every gained and lost complete pair by identity. [Verified readback](evidence/readback.sha256)
