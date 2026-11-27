#!/usr/bin/env bash
set -euo pipefail
code=$(cd "$(dirname "$0")" && pwd)
scores=${1:?quartet score ledger}
geometry=${2:?support accounting ledger}
grade=${3:?emission and pair ledger}
out=${4:?new receipt directory}
mkdir "$out"
for split in held train; do
  pair_files=("$geometry/$split-selector-pairs.jsonl" "$geometry/$split-post-b-pairs.jsonl")
  if [[ $split == train ]]; then pair_files+=("$geometry/train-archive-selector-pairs.jsonl"); fi
  jq -s -f "$code/thresholds.jq" "$grade/$split/pairs.jsonl" > "$out/$split-thresholds.json"
  jq -n --arg split "$split" --slurpfile pairs "$grade/$split/pairs.jsonl" \
    --slurpfile geom <(cat "${pair_files[@]}") \
    --slurpfile contexts "$geometry/$split-selector-contexts.jsonl" \
    --slurpfile zero "$scores/$split-zero.json" --slurpfile period "$scores/$split-period.json" \
    --slurpfile after "$scores/$split-post-b.json" --slurpfile full "$scores/$split-full.json" \
    -f "$code/verify-features.jq" > "$out/$split-features.json"
done
