#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 9 ]]; then
  echo 'usage: ordinary.sh CHAT TOKENIZER PROMPTS PROMPTS_SHA ARM ARTIFACT ARTIFACT_SHA NEW_OUTPUT JOBS' >&2
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
jobs=$9
[[ $arm =~ ^[a-z][a-z0-9_-]*$ ]]
[[ $jobs =~ ^[1-9][0-9]*$ ]]
[[ $output == /mnt/plexpool/uor-r4-corpus/runs/20261127-codex-whole-turn-fit/* ]]
mkdir "$output"
mkdir "$output/records"
{
  printf '%s  %s\n' \
    a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer" \
    "$prompts_sha" "$prompts" "$artifact_sha" "$artifact"
  sha256sum "$chat" "$(realpath "$0")"
} > "$output/INPUTS.sha256"
sha256sum --quiet --strict -c "$output/INPUTS.sha256"

jq -es '
  length > 0 and (length == (map(.case_id) | unique | length))
    and all(.[]; keys == ["case_id", "prompt"]
         and (.case_id | type == "string" and test("^[a-zA-Z0-9_-]+$"))
         and (.prompt | type == "string" and length > 0 and (contains("\u0000") | not)))
' "$prompts" >/dev/null

# A NUL-separated name and body preserve the exact raw string, including trailing
# newlines. The scorer joins labels only after all ordinary clients return.
jq -j '.case_id, "\u0000", .prompt, "\u0000"' "$prompts" |
  xargs -0 -n 2 -P "$jobs" bash -ceu '
    chat=$1
    tokenizer=$2
    artifact=$3
    output=$4
    case_id=$5
    prompt=$6
    /usr/bin/time -o "$output/records/$case_id.time" -f "%e %U %S %M" \
      "$chat" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 \
      --say "$prompt" --temperature 0 --max-tokens 16 \
      --authenticated-copy --native-occurrence-trace \
      > "$output/records/$case_id.json" 2> "$output/records/$case_id.stderr"
  ' ordinary "$chat" "$tokenizer" "$artifact" "$output"

while IFS= read -r case_id; do
  jq -ces --arg id "$case_id" --arg arm "$arm" '
    if length == 1 and .[0].schema == "uor-r4.uor-chat-discrete-text/1" then
      {case_id:$id, arm:$arm, client:.[0]}
    else error("invalid client record for " + $id) end
  ' "$output/records/$case_id.json"
done < <(jq -er .case_id "$prompts") > "$output/ordinary.jsonl"
sha256sum --quiet --strict -c "$output/INPUTS.sha256"
(cd "$output" && sha256sum ordinary.jsonl INPUTS.sha256 > SEALED.sha256)
