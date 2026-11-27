#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 4 ]]; then
  echo 'usage: readback.sh NEW_ROOT PACKET.tar.zst EXPECTED_SHA256 BASE.d11' >&2
  exit 2
fi
out=$(realpath "$1")
packet=$(realpath "$2")
expected_packet=$3
base=$(realpath "$4")
mkdir "$out/payload" "$out/rebuilt"
printf '%s  %s\n' "$expected_packet" "$packet" | sha256sum --strict --quiet -c -
tar --zstd -C "$out/payload" -xf "$packet"
cd "$out/payload"
sha256sum --strict --quiet -c PAYLOAD.sha256
for dir in components-v1 product-inputs-v1 geometry-v3; do
  (cd "$dir" && sha256sum --strict --quiet -c SHA256SUMS)
done
(cd measured-v1 && sha256sum --strict --quiet -c OUTPUTS.sha256)
bash source/product/historical.sh measured-v1 product-inputs-v1 > "$out/historical.tsv"
cmp historical.tsv "$out/historical.tsv"
bash source/product/pack.sh "$out/rebuilt" components-v1 product-inputs-v1/original-fit.json "$base"
bash source/product/replay.sh "$out/rebuilt" product-inputs-v1/panels/held.jsonl measured-v1/raw/held
for arm in zero period post-b full; do
  (cd "product-replay-v1/$arm" && sha256sum --strict --quiet -c SHA256SUMS)
  cmp "product-v1/$arm/artifact.json" "$out/rebuilt/product-v1/$arm/artifact.json"
  for file in requests.jsonl client.jsonl cold.jsonl cold-receipt.json; do
    cmp "product-replay-v1/$arm/$file" "$out/rebuilt/product-replay-v1/$arm/$file"
  done
done
cd "$out"
{
  printf '%s  %s\n' "$expected_packet" "$packet"
  sha256sum historical.tsv rebuilt/product-v1/*/{artifact.d11,artifact.json} \
    rebuilt/product-replay-v1/*/{requests.jsonl,client.jsonl,cold.jsonl,cold-receipt.json}
} > readback.sha256
sha256sum --strict --quiet -c readback.sha256
