#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
m=$(realpath "${1:?TRAIN}"); h=$(realpath "${2:?raw or rejoined held}"); panel=$(realpath "${3:?panel}"); g=${4:?new grade}
mkdir "$g"
arms=(native unfitted gap ordered_gap baseline ordered_base far_only endpoint ordered endpoint_full ordered_full correct_link exchanged_link)
for domain in held contrast; do
    d="$h/$domain"; path="$panel/$domain"; out="$g/$domain"; mkdir "$out"
    count=256; if [[ $domain == contrast ]]; then count=64; fi
    for arm in "${arms[@]}"; do
        jq -nc --arg arm "$arm" --slurpfile rows "$d/$arm.pairs.json" '{key:$arm,value:$rows[0]}' >> "$out/lanes.jsonl"
        if [[ $domain == contrast ]]; then
            jq -es -f "$here/contrast.jq" "$d/$arm.rows.jsonl" > "$out/$arm.flips.json"
        else
            for distance in near far; do
                jq -c --arg distance "$distance" 'select(.distance==$distance)' "$d/$arm.rows.jsonl" |
                    jq -s --slurpfile reference "$path/reference.json" -f "$here/witness.jq" > "$out/$arm.first-$distance.json"
            done
        fi
    done
    jq -es 'from_entries' "$out/lanes.jsonl" > "$out/arms.json"
    jq -ne --argjson row_count "$count" --slurpfile arms "$out/arms.json" -f "$here/original-summary.jq" > "$out/ids.json"
    jq -e 'walk(if type=="object" and has("earlier") and has("exact") then .exact|=length|.full|=length|.earlier|=length else . end)' \
        "$out/ids.json" > "$out/counts.json"
    jq -e '.distance|map_values(. as $r|
        [["baseline","endpoint"],["ordered_base","ordered"],["endpoint","ordered"],
         ["baseline","endpoint_full"],["ordered_base","ordered_full"],["endpoint_full","ordered_full"]]|
        map(. as [$a,$b]|{from:$a,to:$b,full_gained:($r[$b].full-$r[$a].full),full_lost:($r[$a].full-$r[$b].full),
            exact_gained:($r[$b].exact-$r[$a].exact),exact_lost:($r[$a].exact-$r[$b].exact)}))' "$out/ids.json" > "$out/movements.json"
    jq -e --argjson half "$((count/2))" '.all.correct_link.full==$half and .all.exchanged_link.prompts==0' "$out/counts.json" >/dev/null
done
for arm in native baseline endpoint ordered correct_link exchanged_link; do
    jq -nc --arg arm "$arm" --slurpfile rows "$h/held/$arm.rows.jsonl" '{key:$arm,value:$rows}' >> "$g/client-lanes.jsonl"
done
jq -sce 'from_entries as $lanes | range(256) as $i | $lanes.baseline[$i] as $r |
    if all($lanes[]; .[$i].case_id==$r.case_id and .[$i].prompt_sha256==$r.prompt_sha256) then
      ($r|{case_id,pair_id,admission,family,distance,prompt_sha256,gold,gold_tokens}) + {arms:($lanes|map_values(.[$i].arm))}
    else error("writer lane enrollment mismatch") end' "$g/client-lanes.jsonl" > "$g/endpoints.jsonl"
for arm in endpoint ordered; do
    jq -e --arg arm "$arm" --arg ref_sha "$(sha "$panel/held/reference.json")" --arg ep_sha "$(sha "$g/endpoints.jsonl")" \
        --slurpfile training "$m/$arm/selected.json" -f "$here/gate.jq" "$g/held/ids.json" > "$g/$arm.gate.json"
done
jq -es '.[0] as $first|map({key:.arm,value:.})|from_entries as $arms |
    {pairs:128,arms:$arms,passing:[$arms[]|select(.pass)|.arm],
     reference_sha256:$first.reference_sha256,endpoints_sha256:$first.endpoints_sha256,
     gate:{at_least_one_trained_arm:any($arms[];.pass)},pass:any($arms[];.pass)}' \
    "$g/endpoint.gate.json" "$g/ordered.gate.json" > "$g/summary.json"
sha256sum "$m/SHA256SUMS" "$h/SHA256SUMS" "$panel/SHA256SUMS" "$here/SHA256SUMS" > "$g/inputs.sha256"
seal "$g"

