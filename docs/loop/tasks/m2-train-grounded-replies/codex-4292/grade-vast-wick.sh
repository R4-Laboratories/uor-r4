#!/usr/bin/env bash
# Apply the registered TRAIN grading rules to the transferred, provider-labelled first batch.
set -euo pipefail
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-protected-transport
method=$(realpath "${1:?signed methods}"); receipt=$(realpath "${2:?restored provider receipt}"); out=${3:?new grade}
[[ $method == "$s"/methods-v* && $receipt == "$s"/fit/vast-capture-v* && $out == "$s"/fit/vast-grade-v* ]]
source "$method/common-wick.sh"
raw="$receipt/train/root/device"
if [[ $# == 4 ]]; then
 stem=$4
 if [[ $stem == base ]]; then id=null; else [[ $stem =~ ^[0-9]+$ ]]; id=$stem; fi
 rows mixed "$train_panel" learned "$raw/$stem.reader.jsonl" | jq -es -f "$prev/grade/pairs.jq" |
  jq -ce --slurpfile score <(jq -ce --argjson id "$id" 'select(.id==$id)' "$raw/scores.jsonl") \
    --slurpfile initial "$original/step-0.pairs.json" -f "$method/grade/candidate.jq" > "$out/records/$stem.json"
 exit 0
fi
[[ $# == 3 && -e $receipt/job-train.log && -e $receipt/job-finite.log ]]
grep -Eq '^# exit=0 end=' "$receipt/job-train.log"
for d in "$method" "$raw" "$receipt/train" "$receipt/finite" "$receipt/receipts/train"; do verify "$d"; done
jq -e '.scope=="first-round-train-only" and .schema=="uor-r4.codex-4292.vast-train/1"' "$receipt/receipts/train/permit.json" >/dev/null
cmp "$raw/base.state" "$parent"; cmp "$raw/step1-replayed.state" "$full_step"
cmp "$receipt/train/initial/reader.jsonl" "$original/step-0.raw.jsonl"
cmp "$receipt/train/initial/edges.jsonl" "$original/initial.jsonl"
mkdir "$out" "$out/records"; bash "$0" "$method" "$receipt" "$out" base
jq -er '.id' "$raw/menu.jsonl" | xargs -P 4 -n 1 bash "$0" "$method" "$receipt" "$out"
cp "$out/records/base.json" "$out/candidates.jsonl"
while read -r id; do cat "$out/records/$id.json"; done < <(jq -r '.id' "$raw/menu.jsonl") >> "$out/candidates.jsonl"
jq -es --slurpfile menu "$raw/menu.jsonl" -f "$method/grade/elect.jq" "$out/candidates.jsonl" > "$out/election.json"
sha256sum "$receipt/train/SHA256SUMS" "$receipt/receipts/train/SHA256SUMS" "$receipt/job-train.log" "$method/SHA256SUMS" "$0" > "$out/inputs.sha256"; seal "$out"
