#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 7 ]]; then
    echo 'Usage: replay-held.sh MANIFEST MANIFEST_SHA256 CHECKPOINT GPU_ROWS GPU_ROWS_SHA256 CAP NEW_OUTPUT_DIRECTORY' >&2
    exit 2
fi
manifest=$1
manifest_sha256=$2
checkpoint=$3
gpu_rows=$4
gpu_sha256=$5
cap=$6
output=$7
ordinary=$(cd "$(dirname "$0")" && pwd)
base=$(dirname "$ordinary")
tokenizer="$base/../20261127-codex-visible-answer/panel/inputs/tokenizer.json"
held="$base/cohort/complete-reservation/final/held.jsonl"
messages="$base/cohort/complete-reservation/final/held-messages.jsonl"
binary="$ordinary/uor-chat"

sha256sum --status -c "$ordinary/binary.sha256"
printf '%s  %s\n' \
    '6ffd2228957ca42705c17fa995071a8360c5f265b719674f8d0c1990e984e4ee' "$held" \
    '4f073db55a86ecf713a5156d3f8cb76b45f90fb10f92e31a243a07f6000333d7' "$messages" \
    'a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377' "$tokenizer" \
    "$manifest_sha256" "$manifest" \
    "$gpu_sha256" "$gpu_rows" | sha256sum --status -c -

mkdir "$output"
printf '%s\n' "manifest_sha256=$manifest_sha256" "gpu_rows_sha256=$gpu_sha256" "cap=$cap" > "$output/invocation.txt"
date -Is > "$output/started"
mismatch=0
for id in 1 2 3; do
    jq -e -c -s --argjson id "$id" 'map(select(.id == $id)) | select(length == 1) | .[0].messages | select(type == "array")' \
        "$messages" > "$output/$id.messages.json"
    jq -e -c -s --argjson id "$id" 'map(select(.id == $id)) | select(length == 1) | .[0].prefix | select(type == "array")' \
        "$held" > "$output/$id.prefix.json"
    jq -e -c -s --argjson id "$id" 'map(select(.id == $id)) | select(length == 1) | .[0].native | {emitted,stop} | select((.emitted | type) == "array" and (.stop | type) == "string")' \
        "$gpu_rows" > "$output/$id.gpu.json"
    prefix_hash=$(jq -r '.[]' "$output/$id.prefix.json" |
        awk '{printf "%02x%02x", $1 % 256, int($1 / 256)}' |
        xxd -r -p | sha256sum | cut -d ' ' -f 1)
    status=0
    "$binary" --butterfly-v3 "$manifest" "$manifest_sha256" "$checkpoint" "$tokenizer" \
        "$output/$id.messages.json" "$cap" --trace "$output/$id.ordinary.json" \
        > "$output/$id.response.bytes" 2> "$output/$id.stderr" || status=$?
    jq -e --argjson status "$status" '
        if .stop == "invalid_bos" then $status == 1 else $status == 0 end
    ' "$output/$id.ordinary.json" > /dev/null
    jq -n -c --argjson id "$id" --arg prefix_hash "$prefix_hash" \
        --slurpfile prefix "$output/$id.prefix.json" \
        --slurpfile gpu "$output/$id.gpu.json" \
        --slurpfile ordinary "$output/$id.ordinary.json" '
        {
          id:$id,
          prefix_equal:($ordinary[0].prefix == $prefix[0]),
          prefix_hash_equal:($ordinary[0].prefix_u16le_sha256 == $prefix_hash),
          emitted_equal:($ordinary[0].emitted == $gpu[0].emitted),
          stop_equal:($ordinary[0].stop == $gpu[0].stop)
        }' >> "$output/compare.jsonl"
    jq -e -s 'all(.[]; .prefix_equal and .prefix_hash_equal and .emitted_equal and .stop_equal)' \
        "$output/compare.jsonl" >/dev/null || mismatch=1
done
date -Is > "$output/finished"
cat "$output/compare.jsonl"
exit "$mismatch"
