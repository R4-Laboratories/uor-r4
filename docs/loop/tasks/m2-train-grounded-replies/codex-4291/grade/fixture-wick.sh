#!/usr/bin/env bash
# Exercise exact pair identity while counts stay unchanged and the TRAIN pocket remains selectable.
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-mixed-transport
test "$(hostname -s)" = wick
g=$(cd "$(dirname "$0")" && pwd)
d=${1:?new fixture root}; [[ $d == "$s"/grade-fixture-v* ]]; mkdir "$d"
jq -nc 'range(9) as $step | [range(128) as $i | ($i<64) as $far |
 {pair:($i|tostring),ids:[$i],family:("family"+(($i%4)|tostring)),distance:(if $far then "far" else "near" end),
 block:(if $far then "common" else "replacement" end),
 exact:($i<($step+1) or $i==64 or ($step==1 and $i==65)),
 full:($i<($step+1) or ($step!=1 and $i==64) or ($step==1 and $i==65))}]' > "$d/checkpoint-pairs.jsonl"
jq -nc 'range(9) as $i|{step:$i,hinge:([20,0,2,2,2,4,5,6,7][$i]),strict_wins:([10,24,19,22,22,18,13,12,11][$i])}' > "$d/quality.jsonl"
jq -ne --slurpfile checkpoints "$d/checkpoint-pairs.jsonl" --slurpfile quality "$d/quality.jsonl" -f "$g/elect.jq" > "$d/election.json"
jq -e '.original_near==["64"] and .trajectory[1].near_full==1 and .trajectory[1].lost_original_near==["64"] and
 .selections=={unconstrained:1,retained:3}' "$d/election.json" > /dev/null
for variant in pass near-substitution far-substitution correct-damaged empty-far; do
 jq -nc --arg v "$variant" '
 def arm($a):[range(128) as $i|($i<64) as $f | ($i%64) as $k |
 (if $a=="correct_link" then true elif $a=="exchanged_link" or $a=="unfitted" or $a=="gap" then false
  elif $a=="baseline" then $k<4 elif $a=="native" then $k<2
  elif $a=="far_only" then (if $f and $v!="empty-far" then $k<12 else $k<4 end)
  else if $f then (if $v=="far-substitution" then (($k<20 and $k!=7) or $k==24) else $k<20 end)
       else (if $v=="near-substitution" then ($k>0 and $k<=12) else $k<12 end) end end) as $w |
 {pair:($i|tostring),ids:[$i],distance:(if $f then "far" else "near" end),family:("family"+(($k%4)|tostring)),
 exact:$w,full:$w,earlier:($w and ($v!="correct-damaged" or $a!="correct_link" or $i!=30)),
 exact_rows:(if $w then 2 else 0 end),full_rows:(if $w then 2 else 0 end),both_governors_rows:2,neutral_rows:0}];
 ["native","unfitted","gap","correct_link","exchanged_link","baseline","far_only","retained","mixed_free","last_far"] |
 map(. as $a|{key:$a,value:arm($a)})|from_entries' > "$d/$variant.arms.json"
 jq -ne --slurpfile arms "$d/$variant.arms.json" -f "$g/summary.jq" > "$d/$variant.ids.json"
 jq -e --slurpfile election "$d/election.json" --arg ref_sha fixture --arg writer_sha fixture -f "$g/gate.jq" "$d/$variant.ids.json" > "$d/$variant.result.json"
done
jq -se '.[0].pass and (.[1].gate.retains_real_held_near_wins|not) and (.[1].pass|not) and
 (.[2].gate.retains_matched_far_gains|not) and (.[2].pass|not) and (.[3].gate.supplied_controls|not) and
 (.[3].pass|not) and .[4].pass and (.[4].far_gain_retention_informative|not)' \
 "$d/"{pass,near-substitution,far-substitution,correct-damaged,empty-far}.result.json > "$d/checks.json"
(cd "$d" && sha256sum *.json *.jsonl > SHA256SUMS)
