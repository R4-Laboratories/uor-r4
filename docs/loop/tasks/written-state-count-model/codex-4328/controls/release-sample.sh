#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 6 ]]; then
  echo 'usage: release-sample.sh WICK_RUNS RELEASE_CHAT PROMPTS.jsonl TRUTH.jsonl ARCHIVED_RELEASE NEW_OUTPUT' >&2
  exit 2
fi
runs=$(realpath "$1")
chat=$(realpath "$2")
prompts=$(realpath "$3")
truth=$(realpath "$4")
saved=$(realpath "$5")
out=$(realpath -m "$6")
tokenizer="$runs/20261127-codex-paired-release/product-archive-v1/payload/tokenizer.json"
zero="$runs/20261127-codex-release-components/product-v1/zero/artifact.d11"
rowwise="$runs/20261127-codex-paired-release/product-v1/rowwise/artifact.d11"
paired="$runs/20261127-codex-paired-release/product-v1/paired/artifact.d11"
mkdir "$out"

jq -nS --slurpfile truth "$truth" --slurpfile zero "$saved/zero.jsonl" \
  --slurpfile rowwise "$saved/rowwise.jsonl" --slurpfile paired "$saved/paired.jsonl" '
  def activates: .client.native_period_release != null;
  def verified($key):
    .client.authenticated_copy_steps[0] as $opening
    | $opening.copied and $opening.read_pos == $key.sources[0].pos and
      $opening.claim_pos == $key.sources[0].pos and $opening.committed_id == $key.sources[0].id;
  ($truth | INDEX(.case_id)) as $keys
  | ($zero + $rowwise + $paired | group_by(.case_id) | map(
      .[0].case_id as $id | $keys[$id] as $key
      | {case_id: $id, continue_request: ($key.group == "release_more"), natural: ($key.fenced | not),
         active_all: all(.[]; activates), inactive_all: all(.[]; activates | not),
         changed_output: any(.[]; .client.native_period_release as $event |
           $event != null and $event.native_id != $event.committed_id),
         verified_all: all(.[]; verified($key))}
    )) as $cases
  | ($cases | map(select(.case_id == "c002" and .active_all and .verified_all)) | first) as $stop
  | ($cases | map(select(.case_id != "c002" and (.continue_request | not) and .active_all and .verified_all and .changed_output))
      | first) as $changed_stop
  | ($cases | map(select(.natural and .inactive_all)) | first) as $inactive
  | ($cases | map(select(.continue_request and .active_all)) | sort_by(.verified_all | not) | first) as $continue
  | if $stop == null or $inactive == null then error("missing required release replay strata") else
      {stop: $stop.case_id, changed_stop: $changed_stop.case_id, inactive: $inactive.case_id, continue: $continue.case_id,
       continue_verified_source: ($continue.verified_all // false)} end
' > "$out/cases.json"
cases=$(jq -c '[.stop,.changed_stop,.inactive,.continue] | map(select(. != null))' "$out/cases.json")
jq -c --argjson cases "$cases" 'select(.case_id as $id | $cases | index($id) != null)' \
  "$prompts" > "$out/prompts.jsonl"

worker() {
  local arm=$1 artifact=$2
  local line id prompt
  while IFS= read -r line; do
    id=$(jq -er .case_id <<<"$line")
    prompt=$(jq -je .prompt <<<"$line"; printf '\001')
    "$chat" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 \
      --say "${prompt%$'\001'}" --max-tokens 12 --temperature 0 \
      --authenticated-copy --native-period-release \
      | jq -Sc --arg id "$id" --arg arm "$arm" '{case_id: $id, arm: $arm, client: del(.artifact)}'
  done < "$out/prompts.jsonl"
}
pids=()
for arm in zero rowwise paired; do
  (set -e; worker "$arm" "${!arm}") > "$out/$arm.jsonl" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ $failed == 0 ]]
cat "$out/"{zero,rowwise,paired}.jsonl > "$out/observed.jsonl"
jq -Sc --argjson cases "$cases" '
  select(.case_id as $id | $cases | index($id) != null) | del(.client.artifact)
' "$saved/"{zero,rowwise,paired}.jsonl > "$out/expected.jsonl"
cmp "$out/observed.jsonl" "$out/expected.jsonl"
[[ $(wc -l < "$out/observed.jsonl") -eq $((3 * $(wc -l < "$out/prompts.jsonl"))) ]]
