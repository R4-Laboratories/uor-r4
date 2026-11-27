#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
[[ $# == 2 ]]
src=$(realpath -e "$1")
packet=$(realpath -e "$2")
runs=$(dirname "$(dirname "$src")")
verify() (cd "$1" && sha256sum --strict --quiet -c "${2:-SHA256SUMS}")
for cohort in fresh near regression; do
    verify "$src/$cohort"
    cp -a -- "$src/$cohort" "$packet/panel/"
done
for receipt in PANELS.sha256 guards.sha256 authority.sha256; do
    verify "$src" "$receipt"
    cp -a -- "$src/$receipt" "$packet/panel/"
done
authority="$packet/inputs/panel-authority"
mkdir "$authority" "$packet/methods/panel"
while read -r hash source; do
    [[ $hash =~ ^[0-9a-f]{64}$ && $source == "$runs/"* ]]
    rel=${source#"$runs/"}
    [[ $rel =~ ^[a-zA-Z0-9_./-]+$ && /$rel/ != */../* ]]
    mkdir -p "$authority/${rel%/*}"
    cp -a -- "$source" "$authority/$rel"
    printf '%s  %s\n' "$hash" "$rel"
    if [[ $source == "$src"/methods-v*/prepare-wick.sh ]]; then
        cp -a -- "${source%/*}/"{fresh-design.json,grammar.json,mask-evidence.jq,prepare-wick.sh} "$packet/methods/panel/"
    fi
done < "$src/authority.sha256" > "$authority/SHA256SUMS"
test -f "$packet/methods/panel/prepare-wick.sh"
verify "$authority"
mkdir "$packet/methods/panel/build-sources"
for item in \
    20261126-codex-third-margin-retain/panel/generator-build-v1 \
    20261126-codex-all-outranker-interval/panel/build-v3 \
    20261126-codex-paired-occurrence-margin/panel/generator-build-v3
do
    dest="$packet/methods/panel/build-sources/${item%%/*}"
    mkdir "$dest"
    mkdir "$dest/source" "$dest/source/crates"
    cp -a -- "$runs/$item/source/docs" "$dest/source/"
    for crate in uor-r4-integer uor-r4-tokenizer uor-r4-discrete uor-r4-native-vector uor-r4-vision; do
        cp -a -- "$runs/$item/source/crates/$crate" "$dest/source/crates/"
    done
    cp -a -- "$runs/$item/"*.log "$runs/$item/"*.sha256 "$runs/$item/SHA256SUMS" "$dest/"
done
