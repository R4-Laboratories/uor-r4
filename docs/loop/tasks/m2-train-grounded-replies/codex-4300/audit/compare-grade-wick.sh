#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
study=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-switch-source-competition
audit="$study/audit"
here=$(cd "$(dirname "$0")" && pwd)
replay=$(realpath -e "${1:?sealed independent replay path}")
grade=$(realpath -e "${2:?sealed primary grade path}")
capture=$(realpath -e "${3:?sealed shared physical capture path}")
out=$(realpath -m "${4:?new comparison directory}")
[[ $# == 4 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs &&
   $here == "$audit"/recipe-v* && $replay == "$audit"/replay-v* &&
   $grade == "$study"/grade-v* && $capture == "$study"/capture-v* && $out == "$audit"/comparison-v* ]]
for directory in "$replay" "$grade" "$capture"; do
    (cd "$directory" && sha256sum --strict --quiet -c SHA256SUMS)
done
shared=$(sha256sum "$capture/SHA256SUMS")
grep -Fxq "$shared" "$replay/inputs.sha256"
grep -Fxq "$shared" "$grade/inputs.sha256"
mkdir "$out"
for cohort in historical train-original train-derived; do
    jq -nes --slurpfile old "$replay/$cohort-before.json" --slurpfile new "$replay/$cohort-after.json" \
       --slurpfile grade "$grade/$cohort.rows.jsonl" -f "$here/compare-grade.jq" > "$out/$cohort.jsonl"
    count=$(wc -l < "$grade/$cohort.rows.jsonl")
    test "$(wc -l < "$out/$cohort.jsonl")" -eq "$count"
    printf '%s\t%s\n' "$cohort" "$count"
done > "$out/rows.tsv"
sha256sum "$0" "$here/compare-grade.jq" "$replay/SHA256SUMS" "$grade/SHA256SUMS" "$capture/SHA256SUMS" > "$out/inputs.sha256"
(
    cd "$out"
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
