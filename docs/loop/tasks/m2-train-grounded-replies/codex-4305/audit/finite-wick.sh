#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 3 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${3%/*}") == zfs ]]
build=$(realpath -e "$1")
physical=$(realpath -e "$2")
out=$3
raw=$physical/finite-v1
fixture=$physical/device/fixture
source=$physical/device/source
one=$physical/build-v1
for member in "$physical" "$build" "$raw" "$fixture" "$source" "$one"; do
    (cd "$member" && sha256sum --strict --quiet -c SHA256SUMS)
done
mkdir "$out"
"$build/argument-audit" argument-captured "$fixture/fixture.text" "$fixture/masks.tsv" \
    "$fixture/fixture.state" "$fixture/fixture.bank" "$raw" "$out/native.json"
jq -ne --slurpfile base "$raw/baseline.jsonl" --slurpfile g "$raw/G.jsonl" \
    --slurpfile n "$raw/N.jsonl" --slurpfile gn "$raw/GN.jsonl" \
    --slurpfile wrongg "$raw/GN-wrong-G.jsonl" --slurpfile wrongn "$raw/GN-wrong-N.jsonl" \
    --slurpfile gold "$raw/gold.jsonl" --slurpfile swap "$raw/swapped.jsonl" '
    def p($v; $sign): [$v.links[$sign].word,$v.links[$sign].site,$v.tuple.ties];
    [p($base[0];0) == [1,0,1], p($g[0];0) == [0,4,1],
     p($n[0];0) == [1,2,1], p($gn[0];0) == [0,1,1],
     p($wrongg[0];0) == [3,2,1], p($wrongn[0];0) == [0,4,1],
     p($gold[0];0) == [0,1,1], p($swap[0];0) == [0,2,1],
     $gold[0].links[0].noun != $swap[0].links[0].noun,
     $gold[0].links[1] == $swap[0].links[1],
     $base[1].tuple.ties == 8 and $base[1].tuple.ordinal == null,
     p($gn[2];1) == [0,3,1] and $gn[2].links[0].governor == $gn[2].links[1].governor]
    | if all then {passed:length} else error("independent finite outcome or falsifier failed") end
' > "$out/controls.json"
sha256sum "$build/SHA256SUMS" "$physical/SHA256SUMS" "$one/argument-score" \
    "$raw/SHA256SUMS" "$source/SHA256SUMS" "$fixture/SHA256SUMS" "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 > SHA256SUMS)
