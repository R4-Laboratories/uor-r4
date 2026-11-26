#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../common-wick.sh"
v=${1:?unused fixture version}; version "$v"
out="$study/grade-fixture-$v"; mkdir "$out"
jq -ne -f "$here/fixture.jq" > "$out/input.json"
for arm in neutral winner control; do
    jq -nce --arg arm "$arm" --slurpfile panel <(jq -c '.neutral[:48][].panel' "$out/input.json") \
        --slurpfile raw <(jq -c --arg a "$arm" '.[$a][].raw' "$out/input.json") \
        --slurpfile rank <(jq -c --arg a "$arm" '.[$a][:48][].rank' "$out/input.json") \
        --slurpfile oldrank <(jq -c '.neutral[:48][].rank' "$out/input.json") \
        --slurpfile ref <(jq '[.neutral[].ref]' "$out/input.json") --slurpfile coord <(jq -c '.neutral[].coord' "$out/input.json") \
        --slurpfile joined <(jq -c --arg a "$arm" '.[$a][].joined' "$out/input.json") \
        --argjson tsv "$(jq -c '[.neutral[].tsv]' "$out/input.json")" -f "$here/rows.jq" > "$out/$arm.jsonl"
done
jq -ne --slurpfile neutral "$out/neutral.jsonl" --slurpfile winner "$out/winner.jsonl" --slurpfile control "$out/control.jsonl" \
    --slurpfile residual <(jq '{residual_pair_ids:([.neutral[:48][].panel.pair_id]|unique)}' "$out/input.json") \
    -f "$here/summary.jq" | jq 'del(.per_prompt)' > "$out/summary.json"
jq -e '.matched_pairs==23 and .recovered_matched_pairs==7 and .arm.neutral.complete_pairs==0 and
    .arm.winner.complete_pairs==8 and .arm.control.complete_pairs==0 and
    .arm.winner.gold_unique_first==16 and .arm.winner.gold_below_first==16 and .arm.winner.gold_tied_first==16' "$out/summary.json" >/dev/null
jq -es 'all(.[]|select(.matched_control!=null);.old_winner.word=="echo" and .matched_control.word=="echo" and
    .old_winner.bpe!=.matched_control.bpe) and length==48' "$out/neutral.jsonl" >/dev/null
seal "$out"
