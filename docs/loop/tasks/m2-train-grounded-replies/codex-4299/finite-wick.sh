#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
v=${1:?new fixture version}; version "$v"; out="$s/core-fixture-$v"; mkdir "$out"
physical_pins; verify "$here"
names=$(states | jq -R . | jq -sc .)
jq -nc --argjson names "$names" '$names|to_entries[]|.key as $i|
 {candidate:.value,changed:(if $i==0 then 0 else 1 end),
  assigned_tail_l1:(if $i==1 then 0 elif $i==2 or $i==3 then 1 else 10 end),
  lost_protected:(if $i==1 then ["A"] else [] end),
  far_full:(if $i==3 then ["farA","farB"] else ["farA"] end)}|
 .assigned_l1=(.assigned_tail_l1+25)' > "$out/scores.jsonl"
elect() { jq -es --arg arm endpoint --argjson schedule "$names" -f "$here/grade/elect.jq" "$1"; }
elect "$out/scores.jsonl" > "$out/safe.json"; jq -e '.qualified and .chosen=="slot-01"' "$out/safe.json" >/dev/null
jq -c 'if .candidate=="full" then .lost_protected=[] else . end' "$out/scores.jsonl" > "$out/full.jsonl"
elect "$out/full.jsonl" | jq -e '.qualified and .chosen=="full"' >/dev/null
jq -c '.lost_protected=["A"]' "$out/scores.jsonl" > "$out/unsafe.jsonl"
elect "$out/unsafe.jsonl" | jq -e '.qualified==false and .chosen=="base"' >/dev/null
core="$parent/finite-core-v3"
for name in held near-swap control-leak; do
    jq 'walk(if type=="object" and has("baseline") and has("all") then .endpoint=.all|.ordered=.all else . end)' \
        "$core/$name.json" > "$out/$name.json"
done
for arm in endpoint ordered; do
    for name in held near-swap control-leak; do
        jq -e --arg arm "$arm" --arg ref_sha fixture --arg ep_sha fixture --slurpfile training "$core/training.json" \
            -f "$here/gate.jq" "$out/$name.json" > "$out/$arm-$name.gate.json"
    done
    jq -e '.pass' "$out/$arm-held.gate.json" >/dev/null
    jq -e '(.pass|not) and (.gate.retains_held_near|not)' "$out/$arm-near-swap.gate.json" >/dev/null
    jq -e '(.pass|not) and (.gate.supplied_controls|not)' "$out/$arm-control-leak.gate.json" >/dev/null
done
panel="$parent/panel/panel-v1/mixed"
for arm in endpoint ordered; do
    join_rows mixed "$panel" learned "$here/inputs/$arm.original.jsonl" |
        pair_rows mixed > "$out/$arm.pairs.json"
done
jq -ne --slurpfile a "$out/endpoint.pairs.json" --slurpfile b "$out/ordered.pairs.json" --slurpfile k "$here/inputs/premise.json" '
    def near($x): [$x[]|select(.distance=="near" and .full)|.pair];
    def far($x): [$x[]|select(.distance=="far" and .full)|.pair];
    select(near($a[0])==$k[0].protected and far($a[0])==$k[0].far and
            ($k[0].protected-near($b[0])|length)==8 and (far($b[0])|length)==0)|
    {pass:true,endpoint_near:24,endpoint_far:40,ordered_protected_near:16,ordered_train_far:0,
     lower_loss_identity_substitution_rejected:true,full_and_single_slot_winners:true,
     all_infeasible_fallback_tested:true,positive_releases_and_swap_refusals:2}' > "$out/checks.json"
sha256sum "$here/SHA256SUMS" "$core/SHA256SUMS" "$parent/panel/panel-v1/SHA256SUMS" > "$out/inputs.sha256"
seal "$out"; cat "$out/checks.json"

