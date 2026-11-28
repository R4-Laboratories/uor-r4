#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 4 ]] || { echo "usage: run.sh GATE_TOOLS CAPTURE KEY NEW_OUTPUT" >&2; exit 2; }
runner=$(cd "$(dirname "$0")" && pwd)
tools=$(realpath "$1")
capture=$(realpath "$2")
key=$(realpath "$3")
output=$(realpath -m "$4")
mkdir "$output"
cd "$output"
export LD_LIBRARY_PATH="$tools/environment/usr/lib/x86_64-linux-gnu"
export SIGNED_GATE_HIGHS="$tools/environment/usr/bin/highs"
for mask in word passage passage-word supplied; do
    mkdir "$mask" "$mask/scratch"
    export SIGNED_GATE_SCRATCH="$output/$mask/scratch"
    jq -c --arg mask "$mask" --slurpfile truth "$key" -f "$runner/select.jq" "$capture" > "$mask/capture.jsonl"
    "$tools/target/release/signed-site-gate" --rank "$tools/environment/usr/bin/z3" \
        "$mask/capture.jsonl" "$key" > "$mask/verdicts.jsonl"
    "$tools/target/release/verify-signed-verdicts" \
        "$mask/capture.jsonl" "$key" "$mask/verdicts.jsonl" --rank > "$mask/verified.json"
    jq -ce --arg mask "$mask" \
        'select(.level=="panel_positive_only" and .scope=="supported_gold_vs_sites" and .arm=="actual")
        | {mask:$mask,offsets,margin,comparisons:.requirement_count,status:.verdict.status,
           box_maximum:.verdict.box_maximum,weighted_minimum:.verdict.weighted_minimum}' \
        "$mask/verdicts.jsonl" >> summaries.jsonl
done
