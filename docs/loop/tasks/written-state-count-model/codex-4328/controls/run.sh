#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 8 ]]; then
  echo 'usage: run.sh MODEL.d11 MODEL_SHA ARM VERIFY_COPY VERIFY_OCCURRENCE TRUTH.jsonl CLI_TRACES.jsonl NEW_OUTPUT' >&2
  exit 2
fi
artifact=$(realpath "$1")
artifact_sha=$2
arm=$3
copy=$(realpath "$4")
occurrence=$(realpath "$5")
truth=$(realpath "$6")
traces=$(realpath "$7")
out=$(realpath -m "$8")
here=$(cd "$(dirname "$0")" && pwd)

mkdir "$out"
printf '%s  %s\n' "$artifact_sha" "$artifact" | sha256sum --strict --quiet -c -
sha256sum "$copy" "$occurrence" "$truth" "$traces" \
  "$here/run.sh" "$here/requests.jq" "$here/grade.jq" "$here/summary.jq" > "$out/INPUTS.sha256"
jq -nSc --arg arm "$arm" --arg artifact "$artifact_sha" --slurpfile truth "$truth" \
  --slurpfile traces "$traces" -f "$here/requests.jq" > "$out/requests.jsonl"
"$copy" "$artifact" "$out/requests.jsonl" > "$out/copy.jsonl" 2> "$out/copy.json"
"$occurrence" "$artifact" "$out/requests.jsonl" > "$out/occurrence.jsonl" 2> "$out/occurrence.json"
jq -e --arg artifact "$artifact_sha" '.artifact_sha256 == $artifact' "$out/copy.json" >/dev/null
jq -e --arg artifact "$artifact_sha" '.artifact_sha256 == $artifact' "$out/occurrence.json" >/dev/null
cmp <(jq -S .occurrence "$out/copy.json") <(jq -S .occurrence "$out/occurrence.json")
jq -nSc --arg arm "$arm" --slurpfile truth "$truth" --slurpfile traces "$traces" \
  --slurpfile copy "$out/copy.jsonl" --slurpfile assessment "$out/occurrence.jsonl" \
  -f "$here/grade.jq" > "$out/grade.jsonl"
jq -sS -f "$here/summary.jq" "$out/grade.jsonl" > "$out/summary.json"
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
(
  cd "$out"
  sha256sum INPUTS.sha256 requests.jsonl copy.jsonl copy.json occurrence.jsonl occurrence.json \
    grade.jsonl summary.json > SHA256SUMS
  sha256sum --strict --quiet -c SHA256SUMS
)
