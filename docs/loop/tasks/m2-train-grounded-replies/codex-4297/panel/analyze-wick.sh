#!/usr/bin/env bash
# Reduce the sealed physical TRAIN and held captures. This only reads their records.
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-all-outranker-interval"
here=$(cd "$(dirname "$0")" && pwd)
measure=${1:?TRAIN version}
capture=${2:?held version}
version=${3:?unused analysis version}
[[ $# == 3 && $here == "$study"/panel/methods-v* ]]
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$study") == zfs ]]
for v in "$measure" "$capture" "$version"; do [[ $v =~ ^v[1-9][0-9]*$ ]]; done
m="$study/measurement-$measure"
h="$study/held-$capture"
p="$study/panel/panel-v1"
out="$study/panel/analysis-$version"
partial="$out.partial"
[[ ! -e $out && ! -e $partial ]]
premise="$runs/20261126-codex-protected-transport/measurement-v1/premise.json"
printf '%s  %s\n' 5543f2c9da1a7841773f2a7b3e74b0893bd331844456284f25ba08579572dadc \
    "$premise" | sha256sum --strict --quiet -c -
for tree in "$m" "$h" "$p"; do
    (cd "$tree" && sha256sum --strict --quiet -c SHA256SUMS)
done
for seal in "$m/SHA256SUMS" "$p/SHA256SUMS"; do
    grep -Fxq "$(sha256sum "$seal")" "$h/inputs.sha256"
done

mkdir "$partial"
(
    for round in 1 2; do
        batch="$m/train/ordered-all/round-$round/grade"
        jq -sc --argjson round "$round" '{round:$round,candidates:.}' "$batch/candidates.jsonl"
    done
) | jq -sce --slurpfile premise "$premise" -f "$here/train-candidates.jq" > "$partial/train-candidates.json"
endpoint="$m/train/strongest/round-1/device/base"
ordered="$m/train/ordered-all/round-1/device/base"
cmp "$endpoint.state" "$ordered.state"
cmp "$ordered.state" "$m/states/ordered-all.state"
state=$(sha256sum "$ordered.state" | cut -d' ' -f1)
jq -e --arg state "$state" '.["ordered-all"] | .round_ids==[null,null] and .state_sha256==$state' \
    "$m/selection.json" >/dev/null
jq -ce --arg state "$state" --slurpfile summary "$partial/train-candidates.json" \
    --slurpfile endpoint "$endpoint.reader.jsonl" --slurpfile ordered "$ordered.reader.jsonl" \
    -f "$here/train-source.jq" "$p/mixed/reference.json" > "$partial/train-source.json"

lanes=(correct_link exchanged_link baseline strongest all ordered_base ordered-all)
for arm in "${lanes[@]}"; do
    jq -nce --arg arm "$arm" --slurpfile rows "$h/contrast/$arm.rows.jsonl" \
        --slurpfile device "$h/contrast/$arm.raw.jsonl" --slurpfile ref "$p/contrast/reference.json" \
        --slurpfile matched "$p/contrast/matched.json" -f "$here/contrast-readout.jq" \
        > "$partial/$arm.contrast.json"
done
jq -sce 'map({key:.arm,value:.roles}) | from_entries' "$partial/"*.contrast.json > "$partial/by-role.json"
cmp <(jq -Sc 'del(.arm)' "$partial/ordered_base.contrast.json") \
    <(jq -Sc 'del(.arm)' "$partial/ordered-all.contrast.json")
sha256sum "$m/SHA256SUMS" "$h/SHA256SUMS" "$p/SHA256SUMS" "$premise" \
    "$here/"{analyze-wick.sh,train-candidates.jq,train-source.jq,contrast-readout.jq} > "$partial/inputs.sha256"
(
    cd "$partial"
    sha256sum ./*.json inputs.sha256 > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS inputs.sha256
)
chmod -R a-w "$partial"
mv -T --no-clobber "$partial" "$out"
test ! -e "$partial"
sha256sum "$out/"{SHA256SUMS,by-role.json,train-candidates.json,train-source.json}
