#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
root="$runs/20261126-codex-paired-occurrence-margin/fit/recovery"
prior="$runs/20261126-codex-fixed-assignment-median"
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $# == 1 && $1 =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
[[ $(findmnt -n -o FSTYPE --target "$root") == zfs && $here == "$root"/methods-v* ]]
version=$1
target="$root/crossovers-$version"
mkdir "$target" "$target/source" "$target/packed"
cp -- "$here/"{native.rs,crossover.rs,prepare-wick.sh} "$target/source/"
original="$prior/methods-v2/inputs/protected.state"
median="$prior/train-v1/states/ordered.state"
bundle="$prior/methods-v2/inputs/bundle"
{
    printf '%s  %s\n' \
        a2f8d37a5304e557996d2a364ac064e96f9aeeed034357738a3c2989a81b8d8a "$original" \
        0edd2cb3ee9251b9fa40cd829b52e3aff2fc477bf05d515c787cea9435af01ea "$median" \
        8355e7518359cdb1aa7bf0511593990116a343c2284ae08b84596d99d38d6d7f "$bundle"
} > "$target/inherited.sha256"
sha256sum --strict --quiet -c "$target/inherited.sha256"
date -u +%FT%TZ > "$target/prepared-utc.txt"
rustc --version > "$target/rustc.txt"
rustc --edition 2021 -D warnings "$target/source/crossover.rs" -o "$target/crossover"
"$target/crossover" "$original" "$median" "$target/states"
[[ $(wc -l < "$target/states/candidates.tsv") -eq 128 ]]
while IFS=$'\t' read -r name state; do
    payload="$target/packed/$name.bridge"
    "$bundle" pack ordered "$target/states/$state" "$payload"
    [[ $(stat -c %s "$payload") == 13352 ]]
    [[ $("$bundle" unpack "$payload" "$target/packed/$name.unpacked") == ordered ]]
    cmp "$target/states/$state" "$target/packed/$name.unpacked"
done < "$target/states/candidates.tsv"
(
    cd "$target"
    find states packed -type f -print0 | sort -z | xargs -0 sha256sum > payload.SHA256SUMS
    sha256sum --quiet --strict -c payload.SHA256SUMS
    find . -type f -print0 | sort -z | xargs -0 sha256sum > ../"crossovers-$version".pending
)
mv -- "$root/crossovers-$version.pending" "$target/SHA256SUMS"
archive="$root/crossovers-$version.tar.gz"
tar -C "$target" -czf "$archive" states packed payload.SHA256SUMS inherited.sha256 prepared-utc.txt
sha256sum "$target/SHA256SUMS" "$archive" > "$root/crossovers-$version.receipt.sha256"
cat "$root/crossovers-$version.receipt.sha256"
