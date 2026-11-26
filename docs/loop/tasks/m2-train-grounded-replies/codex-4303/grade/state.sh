#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
panel=$1; raw=$2; probes=$3; id=$4; eid=$5; base=$6; out=$7; part=$8
mkdir "$out"
jq -enc --slurpfile rows "$panel/$part/rows.jsonl" --slurpfile meta "$panel/$part/requests.jsonl" \
    --slurpfile raw "$raw/$part/$id.costs.jsonl" --slurpfile native "$raw/$part/$id.native.jsonl" \
    -f "$here/rows.jq" > "$out/$part.rows.jsonl"
jq -enc --slurpfile meta "$panel/$part/requests.jsonl" --slurpfile raw "$out/$part.rows.jsonl" \
    --slurpfile obs "$probes/$eid.jsonl" --slurpfile base "$probes/$base.jsonl" \
    -f "$here/metrics.jq" > "$out/$part.json"
if [[ $part == train ]]; then
    jq -enc --slurpfile rows "$panel/regression/rows.jsonl" --slurpfile meta "$panel/regression/requests.jsonl" \
        --slurpfile raw "$raw/regression/$id.costs.jsonl" --slurpfile native "$raw/regression/$id.native.jsonl" \
        -f "$here/rows.jq" > "$out/regression.rows.jsonl"
    jq -ecs -f "$here/pairs.jq" "$out/regression.rows.jsonl" > "$out/regression.json"
    jq -enc --slurpfile gold "$panel/near/gold.jsonl" --slurpfile native "$raw/near/$id.native.jsonl" \
        -f "$here/near.jq" > "$out/near.json"
fi
