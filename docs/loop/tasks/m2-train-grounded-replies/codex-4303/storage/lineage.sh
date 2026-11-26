#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
[[ $# == 2 ]]
root=$(realpath -e "$1")
out=$(realpath -e "$2")
here="$root/methods/storage"
train="$root/physical/train"
fit="$train/fit-out"
election="$out/election"
seed="$root/inputs/train-input/models/slot3.state"
winner=$(jq -er .choice.m "$election/election.json")
eligibility=$(jq -er .choice.e "$election/election.json")
source=$(awk -F '\t' -v id="$winner" '$1 == id {print $6}' "$out/factors.tsv")
[[ $source =~ ^[a-zA-Z0-9-]+$ ]]
case "$source" in
projection-*)
    projections="$root/physical/backoff/projected"
    read -r parent hash < "$root/inputs/backoff-input/source.tsv"
    [[ $parent =~ ^[a-zA-Z0-9-]+$ && $hash =~ ^[a-f0-9]{64}$ ]]
    printf '%s  %s\n' "$hash" "$train/matrix/$parent.state" | sha256sum --strict --quiet -c -
    rustc --edition 2021 -D warnings "$here/projected-state.rs" -o "$out/projected-state"
    "$out/projected-state" "$seed" "$train/matrix/$parent.state" "$out/bundle/margin.state" "$source" > "$out/projected-state.json"
    ;;
*)
    parent="$winner"
    cmp "$out/bundle/margin.state" "$fit/$source.state"
    ;;
esac
parent_source=$(awk -F '\t' -v id="$parent" '$1 == id {print $6}' "$train/matrix/factors.tsv")
[[ $parent_source =~ ^[a-zA-Z0-9-]+$ ]]
jq -ecs --arg id "$parent_source" '
    [.[]|select(.candidate==$id and (.phase=="fit" or .phase=="diversity"))] |
    if length==0 then error("M winner missing from archived TRAIN proposal schedule") else . end
' "$fit/schedule.jsonl" > "$out/margin-schedule.json"
for id in b "$parent"; do
    origin=$(awk -F '\t' -v id="$id" '$1==id {print $6}' "$train/matrix/factors.tsv")
    [[ $origin =~ ^[a-zA-Z0-9-]+$ ]]
    jq -en --arg id "$origin" --slurpfile terms "$fit/objectives.jsonl" --slurpfile score "$train/score/train/$id.costs.jsonl" \
        -f "$here/fit-agreement.jq" > "$out/$id.fit-agreement.json"
done

coordinates() { od -An -v -td2 -j "$2" -N "$3" "$1" | awk '{for (i=1;i<=NF;i++) print $i}'; }
layout_sign_bytes=6664
transport_start=6152
for sign in 1 -1; do
    offset=$((8 + (sign == -1) * layout_sign_bytes + transport_start))
    paste <(coordinates "$seed" "$offset" 512) <(coordinates "$out/bundle/margin.state" "$offset" 512) |
        awk -v sign="$sign" '$1 != $2 {printf "%d %d %d %d %d %d\n",sign,int((NR-1)/32),(NR-1)%32,$1,$2,$2-$1}'
done | jq -eRs '
    [splits("\n")|select(length>0)|split(" ")|map(tonumber)|
     {sign:.[0],slot:.[1],axis:.[2],from:.[3],to:.[4],delta:.[5]}]
    | if length>0 and all(.[]; .axis>=20) then . else error("M changed outside the transport tail") end
' > "$out/margin-delta.json"

if [[ $source != projection-* ]]; then
    jq -ecs --arg id "$source" '
        map(select(.candidate=="start" or .candidate==$id)) | group_by(.row)
        | map(select(any(.[]; .opportunity=="eligible")) |
            (map(select(.candidate=="start")) | first) as $b | (map(select(.candidate==$id)) | first) as $m |
            {sign:$b.sign,row:$b.row,selector:$b.selector,delta:($m.margin-$b.margin),
             reranked:($m.joint[1].min_sites!=$b.joint[1].min_sites),strict_before:$b.strict_win,
             strict_after:$m.strict_win,target:$m.requested,third:$m.joint[1].site,margin:$m.margin})
        | {rows:.,by_sign:(group_by(.sign)|map({sign:.[0].sign,rows:length,
            improved:(map(select(.delta>0))|length),unchanged:(map(select(.delta==0))|length),
            regressed:(map(select(.delta<0))|length),reranked:(map(select(.reranked))|length)}))}
    ' "$fit/objectives.jsonl" > "$out/margin-train-changes.json"
fi

old_count=$(od -An -tu4 -j 8 -N 4 "$seed" | awk '{print $1}')
new_count=$(od -An -tu4 -j 8 -N 4 "$out/bundle/eligibility.state" | awk '{print $1}')
anchor=$(jq -er --arg id "$eligibility" '.proposals[$id].anchor' "$out/independent/eligibility-fit.json")
[[ $((old_count+1)) == "$new_count" ]]
paste <(coordinates "$seed" "$((12 + anchor * 192))" 192) \
      <(coordinates "$out/bundle/eligibility.state" "$((12 + old_count * 192))" 192) |
    awk '$1 != $2 {printf "%d %d %d %d\n",NR-1,$1,$2,$2-$1}' |
    jq -eRs --argjson anchor "$anchor" --argjson before "$old_count" --argjson after "$new_count" '
      [splits("\n")|select(length>0)|split(" ")|map(tonumber)|{axis:.[0],from:.[1],to:.[2],delta:.[3]}] as $delta |
      {sign:1,anchor:$anchor,slot:$before,active_before:$before,active_after:$after,
       changed_coordinates:($delta|length),l1_from_anchor:($delta|map(.delta|if .<0 then -. else . end)|add),delta:$delta}
    ' > "$out/eligibility-delta.json"

if [[ -d $root/independent/physical-train-factors-v1 ]]; then
    cmp "$out/independent/margin-fit.json" "$root/independent/physical-train-factors-v1/m-fit.json"
    cmp "$out/independent/eligibility-fit.json" "$root/independent/physical-train-factors-v1/e-fit.json"
fi
jq -es --arg winner "$winner" --arg eligibility "$eligibility" '
    def describe($seed):
      {id, src, retain_admit:.t.retain_admit, exact_native_pairs:.t.source,
       fixed_rows:.t.fixed_rows, fixed_pairs:.t.fixed_pairs, fixed_ids:.t.fixed_ids,
       worst:.t.worst, debt:.t.debt,
       regression_losses:($seed.regression_ids-.regression_ids), near_losses:($seed.near_ids-.near_ids)};
    INDEX(.[];.id) as $map | $map.b as $seed
    | ((map(select(.reg<4 or .near<24)) | sort_by(-.t.fixed_pairs,-.t.fixed_rows,.t.worst,.t.debt)
      | first) // null) as $diagnostic
    | {baseline:($seed|describe($seed)),margin:($map[$winner]|describe($seed)),
       eligibility:($map[$eligibility]|describe($seed)),
       stronger_unprotected:(if $diagnostic==null then null else ($diagnostic|describe($seed)) end)}
' "$election/records.jsonl" > "$out/lineage.json"
