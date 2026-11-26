#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?run or inner}; v=${2:?new measurement version}; [[ $v =~ ^v[1-9][0-9]*$ ]]
manifest=$(realpath "${3:?release manifest}"); hash=${4:?approved SHA}
[[ $manifest == "$s"/release-v*/inputs.sha256 && $(sha "$manifest") == "$hash" ]]
sha256sum --strict --quiet -c "$manifest"
rel=$(dirname "$manifest"); gpu=$(cat "$rel/trainer-path.txt"); oracle=$(cat "$rel/oracle-path.txt"); frozen=$(cat "$rel/panel-path.txt")
[[ $gpu == "$s"/fit/build-*/transport-mixed-sm89 && $oracle == "$s"/audit/build-*/target/release/transport-margin-audit && $frozen == "$s"/panel/panel-v* ]]
for f in "$gpu" "$oracle" "$frozen/SHA256SUMS" "$rel/"{trainer-path.txt,oracle-path.txt,panel-path.txt,DECISION.md,FIT.md} "$here/"{measure-wick.sh,common-wick.sh,PRE-RUN.md,SHA256SUMS} \
   "$oldtrainer" "$oldoracle" "$compare" "$reader" "$joint_oracle" "$fixed" "$last_far" "$banks" "$model" "$lock"; do grep -Fxq "$(sha256sum "$f")" "$manifest"; done
verify "$here"
out="$s/measurement-$v"
if [[ $mode == run ]]; then exec bash "$lock" "$out-lease" bash "$here/measure-wick.sh" inner "$v" "$manifest" "$hash"; fi
[[ $mode == inner && -e /proc/$$/fd/9 ]]
verify "$frozen"; verify "$old"
mkdir "$out" "$out/train" "$out/train/mixed" "$out/train/far"
joint_read(){
 local split=$1 p=$2 state=$3 d=$4 stem=$5
 "$reader" learned "$banks" "$p/blind/text.bin" "$state" "$d/$stem.raw.jsonl"
 "$joint_oracle" "$p/blind/text.bin" "$banks" "$state" "$d/$stem.raw.jsonl" "$d/gap.raw.jsonl" > "$d/$stem.replay.json"
 jq -e '.pass and .learned.rows==256 and .learned.disagreements==0 and .governor_gap.disagreements==0' "$d/$stem.replay.json" >/dev/null
 arm_rows "$split" "$p" learned "$d/$stem.raw.jsonl" "$d/$stem"
}
for split in mixed far; do
 p="$frozen/$split"; d="$out/train/$split"
 cmp <(jq -r '.[]|if .distance=="far" then 0 elif .distance=="near" then 1 else error("invalid distance") end' "$p/reference.json") "$p/distance.txt"
 "$oldtrainer" diagnose "$p/blind/text.bin" "$p/links.bin" "$fixed" "$d/initial.jsonl"
 "$oldoracle" diagnose "$p/blind/text.bin" "$p/links.bin" "$fixed" "$fixed" "$fixed" "$d/initial.oracle.jsonl" "$d/initial.reconstruction.json"
 jq -es --slurpfile physical "$d/initial.jsonl" -f "$compare" "$d/initial.oracle.jsonl" > "$d/initial.checks.json"
 "$reader" governor_gap "$banks" "$p/blind/text.bin" "$fixed" "$d/gap.raw.jsonl"
 joint_read "$split" "$p" "$fixed" "$d" step-0
done
cmp <(head -n 256 "$out/train/mixed/initial.jsonl") <(head -n 256 "$out/train/far/initial.jsonl")
cmp <(head -n 128 "$out/train/mixed/step-0.raw.jsonl") <(head -n 128 "$out/train/far/step-0.raw.jsonl")
jq -ne --slurpfile m "$out/train/mixed/initial.jsonl" --slurpfile f "$out/train/far/initial.jsonl" --slurpfile p "$out/train/mixed/step-0.pairs.json" '
 {mixed_eligible:($m|map(select(.eligible_for_fit))|length),far_eligible:($f|map(select(.eligible_for_fit))|length),
 common_far_signed_losses:($m|map(select(.row<128 and .eligible_for_fit and .before.margin<0))|length),
 original_full_near:($p[0]|map(select(.distance=="near" and .full)|.pair)),
 common_far_losing_pairs:($p[0]|map(select(.block=="common" and .distance=="far" and (.exact|not))|.pair))} |
 .pass=(.mixed_eligible==512 and .far_eligible==512 and .common_far_signed_losses>0 and (.original_full_near|length)>0 and (.common_far_losing_pairs|length)>0)' > "$out/premise.json"
if ! jq -e .pass "$out/premise.json" >/dev/null; then sha256sum "$manifest" > "$out/inputs.sha256"; seal "$out"; exit 0; fi
for split in mixed far; do
 p="$frozen/$split"; d="$out/train/$split"
 for tr in first repeat; do "$gpu" "$p/blind/text.bin" "$p/links.bin" "$fixed" "$p/distance.txt" "$d/$tr"; done
 for f in step-{0..8}.state {votes,vote-summary,quality,trace}.jsonl; do cmp "$d/first/$f" "$d/repeat/$f"; done
 "$oldtrainer" train contrastive "$p/blind/text.bin" "$p/links.bin" "$fixed" "$d/old.state" "$d/old.trace" "$d/old.final"
 step=$(jq -sr 'min_by([.hinge,-.strict_wins,.step]).step' "$d/first/quality.jsonl")
 cmp "$d/first/step-$step.state" "$d/old.state"; cmp "$d/first/step-8.state" "$d/old.final"
 cmp "$d/first/trace.jsonl" "$d/old.trace"
 "$oldoracle" contrastive "$p/blind/text.bin" "$p/links.bin" "$fixed" "$d/old.state" "$d/old.final" "$d/old.oracle.jsonl" "$d/old.reconstruction.json"
 jq -es --slurpfile physical "$d/old.trace" -f "$compare" "$d/old.oracle.jsonl" > "$d/old.checks.json"
 "$oracle" "$p/blind/text.bin" "$p/distance.txt" "$fixed" "$d/old.oracle.jsonl" "$d/first" "$d/coordinate-checks.json"
 for step in {1..8}; do joint_read "$split" "$p" "$d/first/step-$step.state" "$d" "step-$step"; done
 jq -ne --slurpfile checkpoints <(cat "$d"/step-{0..8}.pairs.json) --slurpfile quality "$d/first/quality.jsonl" -f "$here/grade/elect.jq" > "$d/election.json"
done
mkdir "$out/states"
for item in far_only:far:unconstrained mixed_free:mixed:unconstrained retained:mixed:retained; do
 IFS=: read -r arm split pick <<< "$item"
 d="$out/train/$split"; step=$(jq -er --arg pick "$pick" '.selections[$pick]' "$d/election.json")
 cp "$d/first/step-$step.state" "$out/states/$arm.state"
done
# The complete training evidence including eligibility and both elections is immutable before access to held ranks.
seal "$out/train"
sha256sum "$out/states/"*.state "$out/train/SHA256SUMS" "$fixed" "$last_far" "$manifest" > "$out/selection.sha256"
for split in held bridge; do
 sha256sum --strict --quiet -c "$out/selection.sha256"
 if [[ $split == held ]]; then p="$frozen/held"; else p="$prev/panel/panel-v1/held"; fi
 d="$out/$split"; mkdir "$d"
 for arm in native unfitted; do "$reader" "$arm" "$banks" "$p/blind/text.bin" "$d/$arm.raw.jsonl"; arm_rows "$split" "$p" "$arm" "$d/$arm.raw.jsonl" "$d/$arm"; done
 "$reader" governor_gap "$banks" "$p/blind/text.bin" "$fixed" "$d/gap.raw.jsonl"
 arm_rows "$split" "$p" governor_gap "$d/gap.raw.jsonl" "$d/gap"
 for arm in baseline far_only mixed_free retained last_far; do
   case $arm in baseline) state=$fixed ;; last_far) state=$last_far ;; *) state="$out/states/$arm.state" ;; esac
   joint_read "$split" "$p" "$state" "$d" "$arm"
 done
 for arm in correct exchanged; do "$reader" "$arm" "$banks" "$p/blind/text.bin" "$p/aid.bin" "$d/$arm.raw.jsonl"; arm_rows "$split" "$p" "${arm}_link" "$d/$arm.raw.jsonl" "$d/${arm}_link"; done
done
# The earlier, independently implemented source instrument must agree on the new native and supplied choices.
p="$frozen/held/structural-links"; mkdir "$out/controls"
for arm in native governor_only nouns_only correct_link exchanged_link; do
 "$old/governor-noun-links-sm89" "$banks" "$p/geometry.bin" "$p/$arm.bin" "$out/controls/$arm.rank"
done
"$old/target/release/governor-noun-links" grade "$p" "$model" "$banks" "$out/controls"
for arm in native correct_link exchanged_link; do
 jq -ne --arg a "$arm" --slurpfile rows "$out/held/$arm.rows.jsonl" --slurpfile other "$out/controls/endpoints.jsonl" '
 def sig:[.selection,.tokens,.bank_cost,.ties,.eligible]; ($other|INDEX(.case_id)) as $idx |
 if ($rows|length)==256 and ($other|length)==256 and all($rows[];. as $r|$idx[.case_id] as $o|
 $o.prompt_sha256==$r.prompt_sha256 and ($r.arm|sig)==($o.arms[$a]|sig)) then {decisions:256,disagreements:0}
 else error("uncorroborated source control") end' > "$out/controls/$arm.checks.json"
done
# Direct comparison with historical raw source traces on the exposed panel, including tie and source coordinates.
cmp "$out/bridge/baseline.raw.jsonl" "$prev/measurement-v2/held/baseline.jsonl"
cmp "$out/bridge/last_far.raw.jsonl" "$prev/measurement-v2/held/learned.jsonl"
sha256sum "$manifest" > "$out/inputs.sha256"; seal "$out"
