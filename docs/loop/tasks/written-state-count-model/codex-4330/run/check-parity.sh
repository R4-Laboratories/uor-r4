#!/usr/bin/env bash
# Compare results on Wick from a cold-loaded client and the sealed CUDA score.
set -euo pipefail
export LC_ALL=C
device=${1:?CUDA decision JSONL}
client=${2:?restored product decision JSONL}
out=${3:?new output directory}
mkdir "$out"
normalize() {
  jq -ceS '
    def observation:
      {fact_bounds,query_bounds,winner_pos,best_score,best_ties,no_read_score,
       candidates:[.candidates[] | {pos,id,score,features}]} |
      if (.no_read_score|type) != "number" or (.best_ties|type) != "number" or
         (any(.candidates[]; (.pos|type)!="number" or (.features|length)!=16))
      then error("incomplete observation") else . end;
    {case_id,
     old_framed:(.old_framed|observation),
     expanded_frozen:(.expanded_frozen|observation),
     expanded_fitted:(.expanded_fitted|observation)}
  ' "$1" | sort
}
normalize "$device" > "$out/device.jsonl"
normalize "$client" > "$out/client.jsonl"
cmp "$out/device.jsonl" "$out/client.jsonl"
jq -es 'map(.case_id) as $ids | ($ids|length) == 1086 and ($ids|unique|length) == 1086' "$out/device.jsonl" >/dev/null
jq -cs '{
  cases:length,
  comparisons:(length * 3),
  candidates:([.[]|.old_framed,.expanded_frozen,.expanded_fitted|.candidates|length]|add),
  feature_words:([.[]|.old_framed,.expanded_frozen,.expanded_fitted|.candidates|length]|add * 16),
  exact:true
}' "$out/device.jsonl" > "$out/parity.json"
(cd "$out" && sha256sum device.jsonl client.jsonl parity.json > SHA256SUMS)
