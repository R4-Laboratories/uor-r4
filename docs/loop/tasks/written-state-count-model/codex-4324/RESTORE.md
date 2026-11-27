# Restore and replay the two release decisions

The [revision-pinned private packet](https://huggingface.co/datasets/caseyallard/uor-r4-store/resolve/5637d1ffa8f182405dd02985c14d6cc445c589f9/results/codex/release-factorial-20261127/release-factorial.tar.zst) occupies 3792694 bytes and has SHA-256 `6103061970e54f0b6237fed0a199530c27a546f9bd687caa1e203588d874515a`. Independent download and replay verified both artifacts on every fresh row.

Wick keeps the immutable raw results under `/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-release-factorial`. The archived study includes the new Linux device binary, source commit, FIT, manifest, launch, raw records in both orders, panel, identity rows and packed-client receipts. The two frozen D11s can be restored from the [content-link study](../codex-4323/RESTORE.md) and [balanced study](../codex-4322/RESTORE.md). This run created no weights. The GPU scientific source is `b08db9e5ec8f1f99055457556e58675a1f733690`.

With the packet extracted in the current directory and the restored D11 paths exported as `FACTORIAL_BALANCED_ARTIFACT` and `FACTORIAL_CANDIDATE_ARTIFACT`, build the two integer binaries following the [product build instructions](product/README.md). Set `FACTORIAL_PRODUCT_BIN` to their absolute directory. A fresh serving process can then reproduce all recorded client and diagnostic transitions on the new panel.

```sh
(cd measured-v1 && sha256sum --quiet --strict -c OUTPUTS.sha256)
(cd grade-v1 && sha256sum --quiet --strict -c SHA256SUMS)
(cd panel-v1/fresh && sha256sum --quiet --strict -c SHA256SUMS)
for model in candidate balanced; do
  out="$PWD/new-$model"
  mkdir "$out"
  path="$PWD/measured-v1/raw/fresh/$model"
  bash staging/product/replay.sh "$out" "$model" fresh \
    "$PWD/panel-v1/fresh/rows.jsonl" "$path.jsonl" "$path-reverse.jsonl"
  for file in cold.jsonl client.jsonl cold-receipt.json; do
    cmp "$out/product-replay-v1/$model/fresh/$file" "product-replay-v1/$model/fresh/$file"
  done
done
```

For every archived controller and release, `grade-v1/SPLIT/MODEL/rows.jsonl` contains the expected and committed IDs, source addresses, native scores, eligibility and terminal verdicts. `grade-v1/SPLIT/identities.json` lists gained and lost pair IDs on the original cohorts. Historical panels are in the predecessor packets. The independent device and cold paths both followed real self-fed tokens. [Downloaded bytes and replay hashes](evidence/readback.sha256)
