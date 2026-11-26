#!/usr/bin/env bash
# Deterministic reducer for recorded physical reads, usable on an immutable download.
source "$(dirname "$0")/common-wick.sh"
m=$(realpath "${1:?measurement}"); raw=$(realpath "${2:?held capture}"); panel=$(realpath "${3:?panel}"); g=${4:?new grade path}
for tree in "$m" "$raw" "$panel"; do [[ $tree == "$s"/* ]]; verify "$tree"; done
[[ $g == "$s"/* ]]; mkdir "$g"
all_arms=(native unfitted gap ordered_gap baseline ordered_base far_only strongest all ordered-all correct_link exchanged_link)
for domain in held contrast; do
    d="$raw/$domain"; out="$g/$domain"; path="$panel/$domain"; mkdir "$out"
    count=256; if [[ $domain == contrast ]]; then count=64; fi
    for arm in "${all_arms[@]}"; do
        jq -nc --arg arm "$arm" --slurpfile records "$d/$arm.pairs.json" '{key:$arm,value:$records[0]}' >> "$out/pair-lanes.jsonl"
        # An all-correct control has no first source error and legitimately returns JSON null.
        jq -s --slurpfile reference "$path/reference.json" -f "$here/grade/witness.jq" "$d/$arm.rows.jsonl" > "$out/$arm.first-source.json"
        if [[ $domain == contrast ]]; then
            jq -es -f "$here/grade/contrast.jq" "$d/$arm.rows.jsonl" > "$out/$arm.flips.json"
        fi
    done
    jq -es 'from_entries' "$out/pair-lanes.jsonl" > "$out/arms.json"
    jq -ne --argjson row_count "$count" --slurpfile arms "$out/arms.json" -f "$here/original-summary.jq" > "$out/ids.json"
    jq -e 'walk(if type=="object" and has("earlier") and has("exact") then .exact|=length|.full|=length|.earlier|=length else . end)' "$out/ids.json" > "$out/counts.json"
    jq -c '.distance|map_values(. as $d|
        [["baseline","strongest"],["strongest","all"],["all","ordered-all"],["ordered_base","ordered-all"]]
        |map(. as [$a,$b]|{from:$a,to:$b,
            full_gained:($d[$b].full-$d[$a].full),full_lost:($d[$a].full-$d[$b].full),
            exact_gained:($d[$b].exact-$d[$a].exact),exact_lost:($d[$a].exact-$d[$b].exact)}))' "$out/ids.json" > "$out/movements.json"
done
d="$raw/held"
for arm in native baseline strongest all ordered-all correct_link exchanged_link; do
    jq -nc --arg arm "$arm" --slurpfile rows "$d/$arm.rows.jsonl" '{key:$arm,value:$rows}' >> "$g/client-lanes.jsonl"
done
jq -sce 'from_entries as $lanes | range(256) as $i | $lanes.baseline[$i] as $r |
    if all($lanes[]; .[$i].case_id==$r.case_id and .[$i].prompt_sha256==$r.prompt_sha256) then
      ($r|{case_id,pair_id,admission,family,distance,prompt_sha256,gold,gold_tokens}) + {arms:($lanes|map_values(.[$i].arm))}
    else error("writer lane enrollment mismatch") end' "$g/client-lanes.jsonl" > "$g/endpoints.jsonl"
for arm in strongest all ordered-all; do
    jq -e --arg arm "$arm" --arg ref_sha "$(sha "$panel/held/reference.json")" --arg ep_sha "$(sha "$g/endpoints.jsonl")" \
        --slurpfile training "$m/train/$arm/selected.json" -f "$here/grade/gate.jq" "$g/held/ids.json" > "$g/$arm.gate.json"
done
jq -es '.[0] as $first|map({key:.arm,value:.})|from_entries as $arms |
    {pairs:128,arms:$arms,passing:[$arms[]|select(.pass)|.arm],
     reference_sha256:$first.reference_sha256,endpoints_sha256:$first.endpoints_sha256,
     gate:{at_least_one_trained_arm:any($arms[];.pass)},pass:any($arms[];.pass)}' \
    "$g/strongest.gate.json" "$g/all.gate.json" "$g/ordered-all.gate.json" > "$g/summary.json"
sha256sum "$m/SHA256SUMS" "$raw/SHA256SUMS" "$panel/SHA256SUMS" "$here/SHA256SUMS" > "$g/inputs.sha256"
seal "$g"
