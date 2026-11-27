#!/usr/bin/env bash
set -euo pipefail
umask 077
study=$(realpath -e "$1"); destination=$2; release=$3
panel="$study/panel"
test -s "$release/FIT.md"
test -s "$release/PRE-REVIEW.md"
mkdir "$destination"
destination=$(realpath -e "$destination")
cd "$destination"
mkdir -p regression near fresh controls
for part in regression near fresh; do
    (cd "$panel/$part" && sha256sum --strict --quiet -c SHA256SUMS)
    cp "$panel/$part/ordered/text.bin" "$part/text.bin"
    cp "$panel/$part/masks.tsv" "$part/masks.tsv"
    if [[ $part != fresh ]]; then cp "$study/seed-v1/previous/$part/original.jsonl" "$part/original.native.jsonl"; fi
done
cp "$panel/fresh/"{gold-aid.bin,swapped-aid.bin} controls/
cp "$release/"{FIT.md,PRE-REVIEW.md} .
cp "$study/release-scripts-v1/qualify.sh" .
find regression near fresh controls qualify.sh FIT.md PRE-REVIEW.md -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum --strict --quiet -c SHA256SUMS
