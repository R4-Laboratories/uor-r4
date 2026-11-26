#!/usr/bin/env bash
# Compact receipts from the sealed collection and its deterministic reducers.
set -euo pipefail
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-all-outranker-interval
core=${1:?runtime methods}; [[ $core =~ ^methods-v[1-9][0-9]*$ ]]; source "$s/$core/common-wick.sh"
mv=${2:?measurement}; hv=${3:?held capture}; gv=${4:?grade}; rv=${5:?new report}; cv=${6:?client capture or refusal}
for v in "$mv" "$hv" "$gv" "$rv" "$cv"; do version "$v"; done
m="$s/measurement-$mv"; h="$s/held-$hv"; g="$s/grade-$gv"; out="$s/report-$rv"
for dir in "$m" "$h" "$g" "$m-lease" "$h-lease"; do verify "$dir"; done
listed "$g/inputs.sha256" "$m/SHA256SUMS"; listed "$g/inputs.sha256" "$h/SHA256SUMS"; mkdir "$out"
jq 'def compact: {near:(.near_full|length),protected_lost:.lost_protected,far:(.far_full|length),
       far_gained,far_lost,strongest_loss,all_loss,strictly_cheaper,equal_cost_rivals,strict_wins,strict_losses,tied_edges};
   map_values(.+{initial:(.initial|compact),first:(.first|compact),final:(.final|compact)})' "$m/selection.json" > "$out/training.json"
jq -r --slurpfile origin "$reference" '(["protected_near_pair","strongest","all","ordered-all"]|@tsv),
    ($origin[0].protected[] as $id|[$id,(.strongest.final.near_full|index($id)!=null),
        (.all.final.near_full|index($id)!=null),(.["ordered-all"].final.near_full|index($id)!=null)]|@tsv)' "$m/selection.json" > "$out/protected-near.tsv"
for domain in held contrast; do
    jq -n --slurpfile counts "$g/$domain/counts.json" --slurpfile movements "$g/$domain/movements.json" \
        '{all:$counts[0].all,distance:$counts[0].distance,family:$counts[0].family,movements:$movements[0]}' > "$out/$domain.json"
    lanes=(baseline ordered_base strongest all ordered-all correct_link exchanged_link)
    for arm in "${lanes[@]}"; do
        jq -nc --arg arm "$arm" --slurpfile witness "$g/$domain/$arm.first-source.json" '{key:$arm,value:$witness[0]}' >> "$out/$domain.witnesses.jsonl"
    done
    jq -nes --slurpfile p "$h/$domain/baseline.rows.jsonl" --slurpfile a "$h/$domain/strongest.rows.jsonl" --slurpfile b "$h/$domain/all.rows.jsonl" \
        --slurpfile c "$h/$domain/ordered-all.rows.jsonl" --slurpfile z "$h/$domain/ordered_base.rows.jsonl" '
        {baseline:$p,strongest:$a,all:$b,"ordered-all":$c,ordered_base:$z} as $x |
        def observed: {selection,tokens,governors};
        [["baseline","strongest"],["strongest","all"],["all","ordered-all"],["ordered_base","ordered-all"]]
        |map(. as [$left,$right]|$x[$left] as $u|$x[$right] as $v|
          [range($u|length) as $i|$u[$i] as $l|$v[$i] as $r |
           if $l.case_id!=$r.case_id then error("mismatched differential rows") else
             {changed:(($l.arm|observed)!=($r.arm|observed)),address_changed:(($l.arm|[.selection,.tokens])!=($r.arm|[.selection,.tokens])),
              gained_full:($r.arm.full and ($l.arm.full|not)),lost_full:($l.arm.full and ($r.arm.full|not))} end] as $diff |
          {from:$left,to:$right,rows:($u|length),joint_read_changes:($diff|map(select(.changed))|length),
           three_noun_address_changes:($diff|map(select(.address_changed))|length),
           full_prompts_gained:($diff|map(select(.gained_full))|length),full_prompts_lost:($diff|map(select(.lost_full))|length)})' > "$out/$domain-changes.json"
done
jq -nes --slurpfile a "$g/contrast/strongest.flips.json" --slurpfile b "$g/contrast/all.flips.json" \
    --slurpfile c "$g/contrast/ordered-all.flips.json" --slurpfile d "$g/contrast/correct_link.flips.json" \
    --slurpfile e "$g/contrast/exchanged_link.flips.json" --slurpfile z "$g/contrast/ordered_base.flips.json" '
    {strongest:$a[0],all:$b[0],"ordered-all":$c[0],ordered_base:$z[0],correct_link:$d[0],exchanged_link:$e[0]}|map_values(del(.records))' > "$out/contrast-flips.json"
cp "$g/summary.json" "$out/source-gate.json"
if jq -e .pass "$g/summary.json" >/dev/null; then
    client="$s/client/tokens-$cv"; verify "$client"; cp "$client/summary.json" "$out/writer.json"
else
    client="$s/client/refusal-$cv"; verify "$client"; cp "$client/receipt.json" "$out/writer.json"
fi
for arm in strongest all ordered-all; do
    for round in 1 2; do
        base="$m/train/$arm/round-$round"
        jq -nc --arg arm "$arm" --argjson round "$round" --arg gpu "$(cat "$base/fit-time.txt")" \
            --arg grade "$(cat "$base/grade-time.txt")" '{arm:$arm,round:$round,physical_training:$gpu,source_grade:$grade}' >> "$out/round-times.jsonl"
    done
done
jq -n --arg hardware "$(cat "$m-lease/device.txt")" --arg train_cost "$(cat "$m-lease/resources.txt")" \
    --arg held_cost "$(cat "$h-lease/resources.txt")" --argjson measurement "$(du -sb "$m"|cut -f1)" \
    --argjson held "$(du -sb "$h"|cut -f1)" --argjson allocated "$(du -sB1 "$s"|cut -f1)" \
    --slurpfile rounds "$out/round-times.jsonl" '{hardware:$hardware,train_lease:$train_cost,
    transfer_lease:$held_cost,measurement_bytes:$measurement,held_bytes:$held,
    zfs_allocated_before_archive:$allocated,raw_state_bytes:13336,tagged_bundle_bytes:13352,rounds:$rounds}' > "$out/resources.json"
sha256sum "$m/SHA256SUMS" "$h/SHA256SUMS" "$g/SHA256SUMS" "$client/SHA256SUMS" "$m-lease/SHA256SUMS" "$h-lease/SHA256SUMS" > "$out/inputs.sha256"; seal "$out"
