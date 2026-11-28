#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 8 ]]; then
  echo 'usage: batch.sh CHAT TOKENIZER PROMPTS PROMPTS_SHA ARM ARTIFACT ARTIFACT_SHA NEW_OUTPUT' >&2
  exit 2
fi
chat=$(realpath "$1")
tokenizer=$(realpath "$2")
prompts=$(realpath "$3")
prompts_sha=$4
arm=$5
artifact=$(realpath "$6")
artifact_sha=$7
output=$(realpath -m "$8")
[[ $arm =~ ^[a-z][a-z0-9_-]*$ ]]
[[ $output == /mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-fit/* ]]
mkdir "$output"
{
  printf '%s  %s\n' \
    a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer" \
    "$prompts_sha" "$prompts" "$artifact_sha" "$artifact"
  sha256sum "$chat" "$(realpath "$0")"
} > "$output/INPUTS.sha256"
sha256sum --quiet --strict -c "$output/INPUTS.sha256"

jq -es '
  length > 0 and (length == (map(.case_id) | unique | length))
    and all(.[]; keys == ["case_id", "prompt"] and (.case_id | type == "string" and length > 0)
         and (.prompt | type == "string" and length > 0 and (contains("\u0000") | not)))
' "$prompts" >/dev/null

UOR_WHOLE_TURN_STUDY_PROMPTS="$prompts" \
  /usr/bin/time -o "$output/batch.time" -f "%e %U %S %M" \
    "$chat" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 \
    --say "study dispatcher" --temperature 0 --max-tokens 16 \
    --authenticated-copy --native-occurrence-trace \
    > "$output/client.jsonl" 2> "$output/batch.stderr"
jq -ces --arg arm "$arm" --slurpfile prompts "$prompts" '
  if map(.case_id) == ($prompts|map(.case_id)) and all(.[]; .client.schema == "uor-r4.uor-chat-discrete-text/1")
  then .[] | .arm=$arm else error("invalid batch client output") end
' "$output/client.jsonl" > "$output/ordinary.jsonl"
sha256sum --quiet --strict -c "$output/INPUTS.sha256"
(cd "$output" && sha256sum ordinary.jsonl INPUTS.sha256 batch.time > SEALED.sha256)
