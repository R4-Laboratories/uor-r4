#!/usr/bin/env bash
set -euo pipefail
root=${1:?Wick study root}; here=$(cd "$(dirname "$0")" && pwd)
bin=${JOINT_PRODUCT_BIN:-/home/callard/uor-codex/20261127-joint-product/target/release}
r=$(dirname "$root"); dest="$root/serving-cost-v1"; mkdir "$dest"
rustc --edition=2021 -O "$here/time-client.rs" -o "$dest/time-client"
lscpu > "$dest/cpu.txt"
for index in 1 2 3; do
 case $index in 1) models=(balanced candidate first-period joint);; 2) models=(joint first-period candidate balanced);; 3) models=(first-period joint balanced candidate);; esac
 for m in "${models[@]}"; do
  case "$m" in
   balanced) art="$r/20261127-codex-period-margin/cold-v1/balanced/artifact.d11";;
   candidate) art="$r/20261127-codex-scope-trace/product-v1/artifact.d11";;
   *) art="$root/product-v1/$m/artifact.d11";;
  esac
  supplied="$root/product-replay-v1/$m/held"; tmp="$dest/$m.$index"
  "$dest/time-client" "$bin/serve-native-source-choice" "$art" "$supplied/requests.jsonl" "$tmp.jsonl" > "$tmp.json"
  cmp "$tmp.jsonl" "$supplied/client.jsonl"
  steps=$(jq -er .matched_client_transitions "$supplied/cold-receipt.json")
  jq -c --arg m "$m" --argjson i "$index" --argjson steps "$steps" '.+{model:$m,repeat:$i,steps:$steps,emitted_steps_per_second:($steps*1000000000/.warm_request_ns),ms_per_request:(.warm_request_ns/1000000/.requests)}' "$tmp.json" >> "$dest/results.jsonl"
 done
done
jq -s 'group_by(.model)|map({model:.[0].model,steps:.[0].steps,requests:.[0].requests,median_ms_per_request:(map(.ms_per_request)|sort|.[1]),median_emitted_steps_per_second:(map(.emitted_steps_per_second)|sort|.[1]),runs:map(.warm_request_ns)})' "$dest/results.jsonl" > "$dest/summary.json"
