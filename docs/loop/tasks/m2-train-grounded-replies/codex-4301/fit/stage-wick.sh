#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
r=/mnt/plexpool/uor-r4-corpus/runs
fit="$r/20261126-codex-paired-occurrence-margin/fit"
reader="$r/20261126-codex-text-governor-links/learner/build-v2"
frozen="$r/20261125-codex-within-prompt-scope-margin/device-build-v1"
oracle="$r/20261126-codex-switch-source-competition/device/build-v1"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $# == 1 && $1 =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
[[ $(findmnt -n -o FSTYPE --target "$fit") == zfs && $here == "$fit"/methods-v* ]]
for dir in "$reader" "$frozen" "$oracle"; do
    (cd "$dir" && sha256sum --quiet --strict -c SHA256SUMS)
done
sha256sum --quiet --strict -c "$reader/dependencies.sha256"
out="$fit/portable-$1"
mkdir "$out" "$out/sources"
sources="$out/sources"
mkdir -p "$sources/codex-4278/fit"
cp -- "$here/"{paired-fit.cu,fixture.rs,build.sh,run.sh,finite.jq,oracle.jq,README.md} "$sources/"
cp -- "$frozen/source/codex-4278/fit/device.cu" "$sources/codex-4278/fit/"
cp -- "$oracle/source/switch-competition.cu" "$sources/"
for file in "$reader/source/learner.cu" "$oracle/source/switch-competition.cu"; do
    [[ $(grep -c '^int main(int argc, char \*\*argv) {$' "$file") == 1 ]]
done
sed '/^int main(int argc, char \*\*argv) {$/,$d' "$reader/source/learner.cu" > "$sources/textlinks-shared.cuh"
sed '/^int main(int argc, char \*\*argv) {$/,$d' "$sources/switch-competition.cu" > "$sources/switch-shared.cuh"
cp -- "$reader/source/learner.cu" "$sources/learner.cu"
sha256sum "$reader/SHA256SUMS" "$frozen/SHA256SUMS" "$oracle/SHA256SUMS" \
    "$reader/source/learner.cu" "$oracle/source/switch-competition.cu" "$frozen/source/codex-4278/fit/device.cu" \
    > "$sources/inherited.sha256"
(cd "$sources" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$out/sources.pending")
mv -- "$out/sources.pending" "$sources/SHA256SUMS"
tar -C "$out" -czf "$out/paired-fit-sources.tar.gz" sources
sha256sum "$sources/SHA256SUMS" "$out/paired-fit-sources.tar.gz" > "$out/receipt.sha256"
cat "$out/receipt.sha256"
