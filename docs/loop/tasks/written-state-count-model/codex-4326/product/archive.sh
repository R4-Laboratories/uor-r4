#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 3 ]]; then
  echo 'usage: archive.sh STUDY_ROOT STAGED_STUDY_SOURCE PRODUCT_INPUTS_DIR' >&2
  exit 2
fi
root=$(realpath "$1")
source=$(realpath "$2")
inputs=$(realpath "$3")
archive="$root/product-archive-v1"
mkdir "$archive"
payload="$archive/payload"
mkdir "$payload" "$payload/product-v1"
(cd "$root/geometry-v3" && sha256sum --strict --quiet -c SHA256SUMS)
cp -a "$root/"{components-v1,measured-v1,product-replay-v1,geometry-v3} "$payload/"
cp -a "$inputs" "$payload/product-inputs-v1"
cp -a "$source" "$payload/source"
cp "$root/product-historical-v1/historical.tsv" "$payload/"
cp "$root/components-source-v1/base-full-v1/roots" "$payload/frozen-roots.u32le"
printf '%s  %s\n' 875a0857dfef840d88a64c90a5777a6154a1674cde08a71429e2cca93fb7bb5a \
  "$payload/frozen-roots.u32le" | sha256sum --strict --quiet -c -
for arm in zero period post-b full; do
  packed="$root/product-v1/$arm"
  (cd "$packed" && sha256sum --strict --quiet -c SHA256SUMS)
  mkdir "$payload/product-v1/$arm"
  cp "$packed/"{artifact.json,INPUTS.sha256} "$payload/product-v1/$arm/"
done
(
  cd "$payload"
  manifest=$(find . -type f -print0 | sort -z | xargs -0 sha256sum)
  printf '%s\n' "$manifest" > PAYLOAD.sha256
  sha256sum --strict --quiet -c PAYLOAD.sha256
)
tar -C "$payload" --zstd -cf "$archive/release-components.tar.zst" .
(
  cd "$archive"
  sha256sum release-components.tar.zst > SHA256SUMS
)
