#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
m=$(realpath "${1:?TRAIN capture}"); panel=$(realpath "${2:?panel}"); out=${3:?output}
mkdir -p "$out"
for arm in endpoint ordered; do
    a="$out/$arm"; mkdir -p "$a"
    for id in $(states); do
        join_rows mixed "$panel/mixed" learned "$m/$arm/$id.raw.jsonl" | pair_rows mixed > "$a/$id.pairs.json"
        jq -ce --arg candidate "$id" --slurpfile score <(jq -c --arg id "$id" 'select(.candidate==$id)' "$m/device/$arm/scores.jsonl") \
            --slurpfile prior "$here/inputs/mixed-pairs.json" --slurpfile protected "$here/inputs/premise.json" \
            -f "$here/grade/candidate.jq" "$a/$id.pairs.json" >> "$a/candidates.jsonl"
    done
    jq -es --arg arm "$arm" --argjson schedule "$(states | jq -R . | jq -sc .)" \
        -f "$here/grade/elect.jq" "$a/candidates.jsonl" > "$a/selected.json"
done
jq -ne --slurpfile a "$out/endpoint/selected.json" --slurpfile b "$out/ordered/selected.json" \
    --slurpfile keep "$here/inputs/premise.json" \
    'if $a[0].base.near_full==$keep[0].protected and $a[0].base.far_full==$keep[0].far and
        ($b[0].base.lost_protected|length)==8 and ($b[0].base.far_full|length)==0
     then {endpoint:$a[0],ordered:$b[0]} else error("original protected or ordered reader changed") end' > "$out/selection.json"

