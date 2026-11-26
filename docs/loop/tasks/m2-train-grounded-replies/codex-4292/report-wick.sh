#!/usr/bin/env bash
# Summarize verified raw decisions. This step has no training or generation entry point.
source "$(dirname "$0")/common-wick.sh"
mv=${1:?measurement}; tv=${2:?transfer}; gv=${3:?grade}; av=${4:?audited TRAIN version}; ov=${5:?report version}
for v in "$mv" "$tv" "$gv" "$av" "$ov"; do [[ $v =~ ^v[1-9][0-9]*$ ]]; done
m="$s/measurement-$mv"; t="$s/transfer-$tv"; g="$s/grade-$gv"; a="$s/audit/train-$av"; o="$s/report-$ov"
for d in "$m" "$t" "$g" "$a"; do verify "$d"; done
mkdir "$o"
for label in root protected free; do
 d="$m/train/$label"; [[ -d $d/device ]] || continue
 e="$d/grade/election.json"
 jq -ne --arg label "$label" --slurpfile e "$e" --slurpfile menu "$d/device/menu.jsonl" \
  --slurpfile moves "$d/grade/candidates.jsonl" --slurpfile audit "$s/audit/batch-$av-$label/readback.json" \
  --slurpfile bad "$d/grade/first_near_rejection.json" --slurpfile improving_bad "$d/grade/first_far_improving_near_rejection.json" '
 def choice($x): if $x==null then null else $x |
   {id,proposal:(if .id!=null then $menu[.id] else null end),hinge,strict_wins,strict_losses,changed,
    near:(.near_full|length),far:(.far_full|length),lost_protected,far_gained,far_lost,gained_from_round_base,lost_from_round_base} end;
 $e[0] as $v | [$moves[] | select(.id!=null and .hinge<$v.base.hinge and (.lost_protected|length)==0)] as $safe |
 {batch:$label,count:$v.counts,by_kind:$v.by_kind,base:choice($v.base),free:choice($v.free),protected:choice($v.protected),
  safe_net_far_from_base:($safe|group_by((.far_full|length)-($v.base.far_full|length))|
          map({net:((.[0].far_full|length)-($v.base.far_full|length)),candidates:length})),
  first_near_reject:{proposal:choice($v.first_near_rejection),source:$bad[0]},
  first_lower_hinge_near_reject:{proposal:choice($v.first_far_improving_near_rejection),source:$improving_bad[0]},
  audit:($audit[0]|del(.metrics))}' > "$o/$label.json"
done
for arm in free protected; do
 batch=$(cat "$m/train/$arm.batch"); d="$m/train/$batch"
 id=$(jq -r --arg a "$arm" '.[$a].id // "base"' "$d/grade/election.json")
 jq -nes --slurpfile origin "$m/train/root/device/base.edges.jsonl" --slurpfile chosen "$d/device/$id.edges.jsonl" '
 def class: if .>0 then "win" elif .==0 then "tie" else "loss" end;
 if ($origin|length)!=512 or ($chosen|length)!=512 then error("signed rows missing") else
 [range(512) as $i|$origin[$i] as $o|$chosen[$i] as $c|
  if $o.row!=$c.row or $o.sign!=$c.sign or $o.eligible_for_fit!=true or $c.eligible_for_fit!=true then error("signed enrollment changed")
  else {distance:(if $o.row<128 then "far" else "near" end),from:($o.before.margin|class),to:($c.before.margin|class)} end] |
 group_by([.distance,.from,.to])|map({distance:.[0].distance,from:.[0].from,to:.[0].to,n:length}) end' > "$o/$arm-flips.json"
done
all=(); for label in root protected free; do [[ ! -e $o/$label.json ]] || all+=("$o/$label.json"); done
jq -s --slurpfile premise "$m/premise.json" --slurpfile selection "$m/selection.json" \
 --slurpfile free_flips "$o/free-flips.json" --slurpfile protected_flips "$o/protected-flips.json" \
 --slurpfile verified "$a/verified.json" --rawfile fit_time "$m-lease/resources.txt" --rawfile test_time "$t-lease/resources.txt" '
 {initial:($premise[0]|.protected|=length|.far|=length),batches:.,verified:$verified[0],state_sha256:$selection[0].state_sha256,
  signed_edge_transitions:{free:$free_flips[0],protected:$protected_flips[0]},fit_resources:$fit_time,transfer_resources:$test_time}' "${all[@]}" > "$o/training.json"
jq --slurpfile movement "$g/movements.json" --slurpfile release "$g/summary.json" '
 def nums:map_values({exact,full,prompts,governor_prompts});
 {all:(.all|nums),distance:(.distance|map_values(nums)),
 family_distance:(.family_distance|map_values(map_values(.exact))),
 matched_pair_movements:($movement[0]|walk(if type=="object" and has("gained") then map_values(length) else . end)),
 gate:($release[0]|{pass,gate,original_near:(.original_near|length),near_lost:(.lost_near|length),
   far_only_gains:(.matched_far_gains|length),far_only_gains_lost:(.matched_far_gains_lost|length)})}' "$g/scores.json" > "$o/transfer.json"
sha256sum "$m/SHA256SUMS" "$t/SHA256SUMS" "$g/SHA256SUMS" "$a/SHA256SUMS" > "$o/inputs.sha256"; seal "$o"
