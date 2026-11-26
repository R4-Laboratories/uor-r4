#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mv=${1:?TRAIN version}; tv=${2:?transfer version}; gv=${3:?unused grade version}
[[ $mv =~ ^v[1-9][0-9]*$ && $tv =~ ^v[1-9][0-9]*$ && $gv =~ ^v[1-9][0-9]*$ ]]
m="$s/measurement-$mv"; t="$s/transfer-$tv"; g="$s/grade-$gv"; d="$t/held"; p="$s/panel/panel-v1/held"
verify "$t"; verify "$m"; verify "$here"
sha256sum --strict --quiet -c "$t/inputs.sha256"
for f in "$m/SHA256SUMS" "$here/SHA256SUMS" "$s/panel/panel-v1/SHA256SUMS"; do grep -Fxq "$(sha256sum "$f")" "$t/inputs.sha256"; done
manifest=$(cut -d' ' -f3- "$m/inputs.sha256"); check_release "$manifest" "$(cut -d' ' -f1 "$m/inputs.sha256")"
mkdir "$g"
args=()
for arm in native unfitted gap baseline far_only free protected correct_link exchanged_link; do args+=(--slurpfile "$arm" "$d/$arm.pairs.json"); done
jq -nc "${args[@]}" '{native:$native[0],unfitted:$unfitted[0],gap:$gap[0],baseline:$baseline[0],far_only:$far_only[0],free:$free[0],
 protected:$protected[0],correct_link:$correct_link[0],exchanged_link:$exchanged_link[0]}' > "$g/arms.json"
jq -ne --slurpfile arms "$g/arms.json" -f "$prev/grade/summary.jq" > "$g/ids.json"
jq -e 'walk(if type=="object" and has("exact") and has("earlier") then .exact|=length|.full|=length|.earlier|=length else . end)' "$g/ids.json" > "$g/scores.json"
args=()
for arm in native baseline far_only free protected correct_link exchanged_link; do args+=(--slurpfile "$arm" "$d/$arm.rows.jsonl"); done
jq -nce "${args[@]}" '{native:$native,baseline:$baseline,far_only:$far_only,free:$free,protected:$protected,correct_link:$correct_link,exchanged_link:$exchanged_link} as $all |
 range(256) as $i | $baseline[$i] as $r |
 if all($all[];.[$i].case_id==$r.case_id and .[$i].prompt_sha256==$r.prompt_sha256) then
 ($r|{case_id,pair_id,admission,family,distance,prompt_sha256,gold,gold_tokens}) + {arms:($all|map_values(.[$i].arm))}
 else error("ordinary lane source enrollment differs") end' > "$g/endpoints.jsonl"
jq -e --arg ref_sha "$(sha "$p/reference.json")" --arg endpoint_sha "$(sha "$g/endpoints.jsonl")" \
 --slurpfile train "$m/selection.json" -f "$here/grade/gate.jq" "$g/ids.json" > "$g/summary.json"
jq -ce '.distance|with_entries(.value |= (. as $cell | $cell.baseline.exact as $b |
 ["free","protected","far_only"]|map(. as $arm|$cell[$arm].exact as $w|
 {key:$arm,value:{kept:($b-($b-$w)),lost:($b-$w),gained:($w-$b)}})|from_entries))' "$g/ids.json" > "$g/movements.json"
for arm in protected free; do
 jq -ne --slurpfile parent "$d/baseline.rows.jsonl" --slurpfile matched "$d/far_only.rows.jsonl" \
  --slurpfile retained "$d/$arm.rows.jsonl" --slurpfile reference "$p/reference.json" -f "$prev/grade/witness.jq" > "$g/$arm.witness.json"
done
sha256sum "$m/SHA256SUMS" "$t/SHA256SUMS" "$here/SHA256SUMS" > "$g/inputs.sha256"; seal "$g"
