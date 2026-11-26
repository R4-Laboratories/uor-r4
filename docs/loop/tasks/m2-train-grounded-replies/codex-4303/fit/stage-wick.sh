#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 2 ]]
here=$(cd "$(dirname "$0")" && pwd -P)
inherited=$(realpath -e "$1")
out=$2
(cd "$inherited" && sha256sum --quiet --strict -c SHA256SUMS)
mkdir -- "$out"
mkdir -- "$out/sources"
for name in build.sh run.sh fixture.rs finite.jq oracle.jq third-fit.cu README.md; do
    cp -- "$here/$name" "$out/sources/"
done
for name in textlinks-shared.cuh switch-shared.cuh switch-competition.cu codex-4278/fit/device.cu; do
    mkdir -p -- "$out/sources/$(dirname "$name")"
    cp -- "$inherited/$name" "$out/sources/$name"
done
cp -- "$inherited/SHA256SUMS" "$out/sources/inherited.SHA256SUMS"
(cd "$out/sources" && find . -type f -print0 | sort -z | xargs -0 sha256sum > ../sources.pending)
mv -- "$out/sources.pending" "$out/sources/SHA256SUMS"
tar -C "$out" -czf "$out/third-fit-sources.tar.gz" sources
(cd "$out" && sha256sum third-fit-sources.tar.gz sources/SHA256SUMS > receipt.sha256)
