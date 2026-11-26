#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
version=${1:?measurement version}; gv=${2:?grade version}
[[ $version =~ ^v[1-9][0-9]*$ && $gv =~ ^v[1-9][0-9]*$ ]]
m="$s/measurement-$version"; g="$s/grade-$gv"; verify "$m"; jq -e '.pass' "$m/premise.json" >/dev/null
verify "$here"
sha256sum --strict --quiet -c "$m/inputs.sha256"
release=$(cut -d' ' -f3- "$m/inputs.sha256")
[[ $release == "$s"/release-v*/inputs.sha256 ]]
sha256sum --strict --quiet -c "$release"
for f in "$here/"{grade-wick.sh,common-wick.sh,SHA256SUMS}; do grep -Fxq "$(sha256sum "$f")" "$release"; done
p=$(cat "$(dirname "$release")/panel-path.txt")
mkdir "$g"
for split in held bridge; do
 d="$m/$split"
 args=()
 for arm in baseline far_only mixed_free retained last_far native unfitted gap correct_link exchanged_link; do
  args+=(--slurpfile "$arm" "$d/$arm.pairs.json")
 done
 jq -nc "${args[@]}" '{baseline:$baseline[0],far_only:$far_only[0],mixed_free:$mixed_free[0],retained:$retained[0],last_far:$last_far[0],native:$native[0],unfitted:$unfitted[0],gap:$gap[0],correct_link:$correct_link[0],exchanged_link:$exchanged_link[0]}' > "$g/$split.arms.json"
 jq -ne --slurpfile arms "$g/$split.arms.json" -f "$here/grade/summary.jq" > "$g/$split.ids.json"
 jq -e 'walk(if type=="object" and has("exact") and has("earlier") then .exact|=length|.full|=length|.earlier|=length else . end)' "$g/$split.ids.json" > "$g/$split.scores.json"
 ref="$p/held/reference.json"; if [[ $split == bridge ]]; then ref="$prev/panel/panel-v1/held/reference.json"; fi
 jq -ne --slurpfile parent "$d/baseline.rows.jsonl" --slurpfile matched "$d/far_only.rows.jsonl" --slurpfile retained "$d/retained.rows.jsonl" --slurpfile reference "$ref" -f "$here/grade/witness.jq" > "$g/$split.witness.json"
done
d="$m/held"; args=()
for arm in native baseline far_only retained correct_link exchanged_link; do args+=(--slurpfile "$arm" "$d/$arm.rows.jsonl"); done
jq -nce "${args[@]}" '{native:$native,baseline:$baseline,far_only:$far_only,retained:$retained,correct_link:$correct_link,exchanged_link:$exchanged_link} as $all |
 range(256) as $i | $baseline[$i] as $r | if all($all[];.[$i].case_id==$r.case_id and .[$i].prompt_sha256==$r.prompt_sha256)
 then ($r|{case_id,pair_id,admission,family,distance,prompt_sha256,gold,gold_tokens})+{arms:($all|map_values(.[$i].arm))}
 else error("writer lanes crossed source addresses") end' > "$g/endpoints.jsonl"
jq -e --slurpfile election "$m/train/mixed/election.json" --arg ref_sha "$(sha "$p/held/reference.json")" --arg writer_sha "$(sha "$g/endpoints.jsonl")" -f "$here/grade/gate.jq" "$g/held.ids.json" > "$g/summary.json"
jq -e '.distance.far as $f | .distance.near as $n | ($f.last_far.exact-$f.baseline.exact) as $fw |
 ($n.baseline.exact-$n.last_far.exact) as $nw | if ($fw|length)==32 and ($nw|length)==8 then
 {historical_far_gains:($fw|length),historical_near_losses:($nw|length),
 arms:(["baseline","last_far","far_only","mixed_free","retained"]|map(. as $a |{key:$a,value:{
 old_far_gains_kept:(($fw-($fw-$f[$a].exact))|length),old_near_wins_recovered:(($nw-($nw-$n[$a].exact))|length),
 missing_far:($fw-$f[$a].exact),missing_near:($nw-$n[$a].exact)}})|from_entries)}
 else error("previously observed panel identity failed") end' "$g/bridge.ids.json" > "$g/bridge-retention.json"
sha256sum "$m/SHA256SUMS" "$here/SHA256SUMS" > "$g/inputs.sha256"; seal "$g"
