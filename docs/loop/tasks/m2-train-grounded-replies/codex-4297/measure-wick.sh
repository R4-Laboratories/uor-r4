#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?run or inner}; v=${2:?new measurement version}; version "$v"
manifest=$(realpath "${3:?release manifest}"); expected=${4:?approved hash}
release_inputs "$manifest" "$expected"; m="$s/measurement-$v"
if [[ $mode == run ]]; then exec bash "$lock" "$m-lease" bash "$0" inner "$v" "$manifest" "$expected"; fi
[[ $mode == inner && -e /proc/$$/fd/9 ]]; ancestry; verify "$panel"
mkdir "$m" "$m/train" "$m/states" "$m/checks"
sha256sum "$manifest" > "$m/inputs.sha256"
for arm in strongest all ordered-all; do
    geom=endpoint; tape="$train/blind/text.bin"
    if [[ $arm == ordered-all ]]; then geom=ordered; tape="$panel/mixed/ordered/text.bin"; fi
    base="$state"; ad="$m/train/$arm"; mkdir "$ad"
    for round in 1 2; do
        batch="$ad/round-$round"; raw="$batch/device"; mkdir "$batch"
        /usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$batch/fit-time.txt" \
            "$trainer" "$arm" "$banks" "$tape" "$train/links.bin" "$base" "$raw"
        cmp "$base" "$raw/base.state"
        if [[ $geom == endpoint && $round == 1 ]]; then cmp "$baseline_raw" "$raw/base.reader.jsonl"; fi
        seal "$raw"
        /usr/bin/time -f 'wall_s=%e user_s=%U system_s=%S peak_rss_kib=%M' -o "$batch/grade-time.txt" \
            bash "$here/grade-batch-wick.sh" grade "$batch" "$arm"
        election="$batch/grade/election.json"
        id=$(jq -r '.chosen.id//"base"' "$election")
        base="$raw/$id.state"; test "$(stat -c %s "$base")" = 13336
        if [[ $round == 1 ]]; then
            cp "$base" "$ad/first.state"
            if [[ $geom == endpoint ]]; then
                jq -e --slurpfile pin "$reference" '.base.near_full==$pin[0].protected and .base.far_full==$pin[0].far' "$election" >/dev/null
            fi
        fi
    done
    cp "$base" "$m/states/$arm.state"; "$packer" pack "$geom" "$base" "$m/states/$arm.bridge"
    test "$("$packer" unpack "$m/states/$arm.bridge" "$m/checks/$arm.state")" = "$geom"
    cmp "$base" "$m/checks/$arm.state"
    jq -ne --arg sha "$(sha "$base")" --arg bundle "$(sha "$m/states/$arm.bridge")" --arg operator "$geom" \
      --slurpfile first "$ad/round-1/grade/election.json" --slurpfile second "$ad/round-2/grade/election.json" '
       $first[0] as $f | $second[0] as $s |
       if (($f.chosen|del(.id,.changed))==($s.base|del(.id,.changed))) and $f.proposals==68 and $s.proposals==68 then
        {operator:$operator,state_sha256:$sha,bundle_sha256:$bundle,qualified:$s.qualified,
         initial:$f.base,first:$f.chosen,final:$s.chosen,round_ids:[$f.chosen.id,$s.chosen.id],
         objective_before:$f.objective_before,objective_after:$s.objective_after,
         feasible_proposals:[$f.eligible_proposals,$s.eligible_proposals],proposals:136}
       else error("state and full source identities changed between rounds") end' > "$ad/selected.json"
done
jq -n --slurpfile a "$m/train/strongest/selected.json" --slurpfile b "$m/train/all/selected.json" \
    --slurpfile c "$m/train/ordered-all/selected.json" '{strongest:$a[0],all:$b[0],"ordered-all":$c[0]}' > "$m/selection.json"
jq -e 'all(.[];.proposals==136) and (.strongest.initial|del(.id,.changed))==(.all.initial|del(.id,.changed))' "$m/selection.json" >/dev/null
sha256sum "$m/states/"*.bridge "$m/states/"*.state "$m/selection.json" > "$m/selection.sha256"
seal "$m"
