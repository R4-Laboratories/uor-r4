#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-protected-transport"
mixed="$runs/20261126-codex-mixed-transport"
measurement="$study/measurement-v1"
panel="$mixed/panel/panel-v1"
grader="$mixed/methods-v2"
old_batch="$measurement/train/root/device"
new_batch="$measurement/train/protected/device"
selection="$measurement/selection.json"
version=${1:?new diagnosis version}
[[ $# == 1 && $version =~ ^v[1-9][0-9]*$ && $(hostname -s) == wick ]]
[[ $(findmnt -no FSTYPE --target "$study") == zfs ]]
out="$study/far-residuals-$version"

member() (
    cd "$1"
    awk -v path="./$2" '$2 == path {print; ++matches} END {if (matches!=1) exit 1}' SHA256SUMS |
        sha256sum --strict --quiet -c -
)
for file in selection.json selection.sha256 train/root/device/SHA256SUMS train/protected/device/SHA256SUMS; do
    member "$measurement" "$file"
done
sha256sum --strict --quiet -c "$measurement/selection.sha256"
for file in mixed/reference.json mixed/blind/coordinates.jsonl; do member "$panel" "$file"; done
member "$grader" grade/join.jq
for file in base.reader.jsonl base.edges.jsonl 234.reader.jsonl 234.edges.jsonl 234.state; do
    member "$old_batch" "$file"
done
for file in base.state 240.reader.jsonl 240.edges.jsonl 240.state; do member "$new_batch" "$file"; done
cmp "$old_batch/234.state" "$new_batch/base.state"
cmp "$old_batch/234.state" "$measurement/states/protected-first.state"
cmp "$new_batch/240.state" "$measurement/states/protected.state"
jq -e '.protected_first.id==234 and .protected.id==240' "$selection" >/dev/null

phase() {
    local stem=$1
    jq -nc --arg split mixed --arg kind learned --slurpfile reference "$panel/mixed/reference.json" \
        --slurpfile coords "$panel/mixed/blind/coordinates.jsonl" --slurpfile physical "$stem.reader.jsonl" \
        -f "$grader/grade/join.jq" |
        jq -cse --slurpfile ranks "$stem.reader.jsonl" --slurpfile edges "$stem.edges.jsonl" \
            '{rows:.,ranks:$ranks,edges:$edges}'
}
mkdir "$out"
jq -ne --slurpfile parent <(phase "$old_batch/base") --slurpfile first <(phase "$old_batch/234") \
    --slurpfile final <(phase "$new_batch/240") --slurpfile selection "$selection" \
    --slurpfile refs "$panel/mixed/reference.json" --slurpfile coords "$panel/mixed/blind/coordinates.jsonl" \
    -f "$here/far-residuals.jq" > "$out/summary.json"
jq -e --slurpfile selection "$selection" '
    .common_signed_edges as $edges | $selection[0] as $s |
    $edges.first.hinge_delta==($s.protected_first.hinge-$s.parent.hinge) and
    $edges.second.hinge_delta==($s.protected.hinge-$s.protected_first.hinge)
' "$out/summary.json" >/dev/null
sha256sum "$measurement/SHA256SUMS" "$measurement/selection.sha256" "$selection" \
    "$old_batch/SHA256SUMS" "$new_batch/SHA256SUMS" "$panel/SHA256SUMS" "$grader/SHA256SUMS" \
    "$here/far-residuals-wick.sh" "$here/far-residuals.jq" > "$out/inputs.sha256"
(cd "$out" && sha256sum --strict --quiet -c inputs.sha256 && sha256sum summary.json inputs.sha256 > SHA256SUMS)
