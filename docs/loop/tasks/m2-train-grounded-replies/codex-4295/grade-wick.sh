#!/usr/bin/env bash
# This reducer grades recorded physical decisions. It can also re-grade a downloaded copy.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/common-wick.sh"
physical=$(realpath "${1:?sealed measurement}"); panel=$(realpath "${2:?sealed panel}"); out=${3:?new grade directory}
[[ $physical == "$study/"* && $panel == "$study/"* && $out == "$study/"* ]]
verify "$physical"; verify "$panel"; historical
pin "$(digest "$old_raw")" "$physical/raw/neutral.jsonl"
mkdir "$out"
for arm in neutral winner control; do
    raw="$physical/raw/$arm.jsonl"
    jq -nce --slurpfile reference "$original/reference.json" --slurpfile coords "$original/blind/coordinates.jsonl" \
        --arg split mixed --arg kind learned --slurpfile physical "$raw" -f "$join" > "$out/$arm.joined.jsonl"
    jq -nce --arg arm "$arm" --slurpfile panel "$panel/rows.jsonl" --slurpfile raw "$raw" \
        --slurpfile rank "$physical/raw/$arm.ranks.jsonl" --slurpfile oldrank "$physical/raw/neutral.ranks.jsonl" \
        --slurpfile ref "$original/reference.json" --slurpfile coord "$original/blind/coordinates.jsonl" \
        --slurpfile joined "$out/$arm.joined.jsonl" --argjson tsv "$(jq -Rn '[inputs]' < "$panel/masks.tsv")" \
        -f "$here/grade/rows.jq" > "$out/$arm.residuals.jsonl"
done
jq -ne --slurpfile n "$physical/raw/neutral.jsonl" --slurpfile w "$physical/raw/winner.jsonl" \
    --slurpfile c "$physical/raw/control.jsonl" --slurpfile selected "$panel/rows.jsonl" '
    ($selected|map(.row)) as $ids | ([range(256)]-$ids) as $untouched |
    if ([$n,$w,$c]|all(.[];length==256)) and ($untouched|length)==208 and
        all($untouched[];. as $i | $n[$i]==$w[$i] and $n[$i]==$c[$i]) then
      {native_rows_identical_to_previous:256,untouched_rows:208,unchanged_raw_control_comparisons:416}
    else error("unmasked read changed") end' > "$out/invariance.json"
jq -ne --slurpfile neutral "$out/neutral.residuals.jsonl" --slurpfile winner "$out/winner.residuals.jsonl" \
    --slurpfile control "$out/control.residuals.jsonl" --slurpfile residual "$old/far-residuals-v1/summary.json" \
    -f "$here/grade/summary.jq" > "$out/analysis.json"
jq 'del(.per_prompt)' "$out/analysis.json" > "$out/summary.json"
jq -c '.per_prompt[]' "$out/analysis.json" > "$out/prompt-records.jsonl"
jq '[.per_prompt[]|select(.pair_id=="enamel-display/road/18")]' "$out/analysis.json" > "$out/medallion.json"
sha256sum "$physical/SHA256SUMS" "$panel/SHA256SUMS" "$here/grade/"{rows,summary}.jq "$join" > "$out/inputs.sha256"
seal "$out"
