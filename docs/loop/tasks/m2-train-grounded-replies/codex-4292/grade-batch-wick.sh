#!/usr/bin/env bash
source "$(dirname "$0")/common-wick.sh"
mode=${1:?grade, one or witness}; run=$(realpath "${2:?batch directory}")
[[ $run == "$s"/measurement-v*/train/* ]]
raw="$run/device"; out="$run/grade"
case $mode in
one)
  token=${3:?base or numeric id}
  if [[ $token == base ]]; then id=null; else [[ $token =~ ^[0-9]+$ ]]; id=$token; fi
  rows mixed "$train_panel" learned "$raw/$token.reader.jsonl" |
    jq -es -f "$prev/grade/pairs.jq" |
    jq -ce --slurpfile score <(jq -ce --argjson id "$id" 'select(.id==$id)' "$raw/scores.jsonl") \
      --slurpfile initial "$original/step-0.pairs.json" -f "$here/grade/candidate.jq" > "$out/records/$token.json"
  ;;
grade)
  [[ -e $raw/complete.json ]]; mkdir "$out" "$out/records"
  bash "$0" one "$run" base
  jq -er '.id' "$raw/menu.jsonl" | xargs -P 4 -n 1 bash "$0" one "$run"
  cp "$out/records/base.json" "$out/candidates.jsonl"
  while read -r id; do cat "$out/records/$id.json"; done < <(jq -er '.id' "$raw/menu.jsonl") >> "$out/candidates.jsonl"
  jq -es --slurpfile menu "$raw/menu.jsonl" -f "$here/grade/elect.jq" "$out/candidates.jsonl" > "$out/election.json"
  for key in first_near_rejection first_far_improving_near_rejection; do
    id=$(jq -r --arg k "$key" '.[$k].id' "$out/election.json")
    if [[ $id == null ]]; then echo null > "$out/$key.json"; continue; fi
    lost=$(jq -c --arg k "$key" '.[$k].lost_protected' "$out/election.json")
    rows mixed "$train_panel" learned "$raw/$id.reader.jsonl" |
     jq -es --argjson lost "$lost" --slurpfile reference "$train_panel/reference.json" -f "$here/grade/source.jq" \
      > "$out/$key.json"
    jq -e --argjson lost "$lost" '.!=null and (.pair as $id| $lost|index($id)!=null)' "$out/$key.json" >/dev/null
  done
  seal "$out"
  ;;
*) exit 64 ;;
esac
