#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 5 ]]; then
  echo 'usage: replay-sample.sh WICK_RUNS TRACE_CHAT PROMPTS.jsonl TRACES.jsonl NEW_OUTPUT' >&2
  exit 2
fi
runs=$(realpath "$1")
chat=$(realpath "$2")
prompts=$(realpath "$3")
original=$(realpath "$4")
out=$(realpath -m "$5")
tokenizer="$runs/20261127-codex-paired-release/product-archive-v1/payload/tokenizer.json"
base="$runs/20261127-codex-word-ending/product-qualified-v1/artifact.d11"
balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
paired="$runs/20261127-codex-paired-release/product-v1/paired/artifact.d11"
cases='["c002","c009","c013","c021","c025","c037","c045","c047"]'

mkdir "$out"
jq -c --argjson cases "$cases" 'select(.case_id as $id | $cases | index($id) != null)' \
  "$prompts" > "$out/prompts.jsonl"
[[ $(wc -l < "$out/prompts.jsonl") -eq 8 ]]

worker() {
  local arm=$1 artifact=$2 mode=$3
  local -a options=()
  case "$mode" in
    authenticated_copy) options=(--authenticated-copy) ;;
    no_read) options=(--no-native-occurrence) ;;
  esac
  local line id prompt client
  while IFS= read -r line; do
    id=$(jq -er .case_id <<<"$line")
    prompt=$(jq -je .prompt <<<"$line"; printf '\001')
    prompt=${prompt%$'\001'}
    client=$("$chat" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 \
      --say "$prompt" --temperature 0 --max-tokens 12 --native-occurrence-trace "${options[@]}")
    jq -Sc --arg id "$id" --arg arm "$arm" --arg mode "$mode" '
      {case_id: $id, artifact_arm: $arm, decode_arm: $mode, execution: "traced", client: del(.artifact)}
    ' <<<"$client"
  done < "$out/prompts.jsonl"
}
pids=()
for arm in base balanced paired; do
  for mode in default authenticated_copy no_read; do
    (set -e; worker "$arm" "${!arm}" "$mode") > "$out/$arm.$mode.jsonl" &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ $failed == 0 ]]

for arm in base balanced paired; do
  for mode in default authenticated_copy no_read; do
    cat "$out/$arm.$mode.jsonl"
  done
done > "$out/observed.jsonl"
jq -Sc --argjson cases "$cases" '
  select(.case_id as $id | $cases | index($id) != null) | del(.client.artifact)
' "$original" > "$out/expected.jsonl"
cmp "$out/observed.jsonl" "$out/expected.jsonl"
[[ $(wc -l < "$out/observed.jsonl") -eq 72 ]]
