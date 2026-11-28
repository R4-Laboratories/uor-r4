#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -lt 4 ]]; then
  echo 'usage: archive.sh WICK_RUNS STAGED_UNIT PINNED_TOOLS FRESH_ARCHIVE_DIR [EXTRA_CAPTURE_DIR ...]' >&2
  exit 2
fi
runs=$(realpath "$1")
unit=$(realpath "$2")
tools=$(realpath "$3")
archive=$(realpath -m "$4")
shift 4
root="$runs/20261127-codex-ordinary-source"
[[ $archive == "$root/"* ]]
tokenizer="$runs/20261127-codex-paired-release/product-archive-v1/payload/tokenizer.json"
payload="$archive/payload"
(
  cd "$root/panel-v1"
  sha256sum --strict --quiet -c SEALED.sha256
)
for name in runtime-v1 controls-v1; do
  (cd "$root/$name" && sha256sum --strict --quiet -c SHA256SUMS && sha256sum --strict --quiet -c INPUTS.sha256)
done
(cd "$root/release-v1" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$root" && sha256sum --strict --quiet -c release-v1/INPUTS.sha256)
(cd "$tools" && sha256sum --strict --quiet -c SHA256SUMS)
mkdir "$archive" "$payload"
cp -a "$root/"{panel-v1,runtime-v1,controls-v1,controls-test-v1,grade-v1,matched-v1,release-v1,release-grade-v1} "$payload/"
cp -a "$unit" "$payload/source"
cp -a "$tools" "$payload/tools"
cp "$tokenizer" "$payload/tokenizer.json"
cp "$root/controls-panel-replay-v1/build.log" "$payload/panel-rebuild.log"
for extra in "$@"; do
  [[ $(realpath "$extra") == "$root/"* ]]
  cp -a "$extra" "$payload/"
done
(
  cd "$payload"
  find . -type f ! -path ./PAYLOAD.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > PAYLOAD.sha256
  sha256sum --strict --quiet -c PAYLOAD.sha256
)
tar --zstd -cf "$archive/ordinary-source.tar.zst" -C "$payload" .
(
  cd "$archive"
  sha256sum ordinary-source.tar.zst > SHA256SUMS
  sha256sum --strict --quiet -c SHA256SUMS
)
