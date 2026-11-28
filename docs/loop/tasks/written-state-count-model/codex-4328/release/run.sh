#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 4 ]]; then
  echo 'usage: run.sh WICK_RUNS CHAT PROMPTS.jsonl NEW_OUTPUT' >&2
  exit 2
fi
runs=$(realpath "$1")
chat=$(realpath "$2")
prompts=$(realpath "$3")
out=$(realpath -m "$4")
mkdir "$out"
tokenizer="$runs/20261127-codex-paired-release/product-archive-v1/payload/tokenizer.json"
zero="$runs/20261127-codex-release-components/product-v1/zero/artifact.d11"
rowwise="$runs/20261127-codex-paired-release/product-v1/rowwise/artifact.d11"
paired="$runs/20261127-codex-paired-release/product-v1/paired/artifact.d11"
{
  printf '%s  %s\n' \
    a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer" \
    5249c0c1443852a4a70850850be0f613e63f7c6efba42b9452cd4493c9c1a5d3 "$zero" \
    ca02e40096d6faed7fb826ca17d47911056c6763eab8544742f03c5753ff7d95 "$rowwise" \
    4a13e77f0320aca1fab33bf92a668e28cef14189afea6ae1f2bc387b968a13e6 "$paired" \
    475d11303e92b9f9a2e03d2499ef877be74220170d5ed75bb18d8ccdc318ff43 "$prompts"
  sha256sum "$chat" "$0"
} > "$out/INPUTS.sha256"
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
jq -se 'length==48 and (map(.case_id)|unique|length)==48 and all(.[];keys==["case_id","prompt"])' "$prompts" >/dev/null
serve() {
  local arm=$1 artifact=$2 row case_id prompt
  while IFS= read -r row; do
    case_id=$(jq -er .case_id <<< "$row")
    prompt=$(jq -je .prompt <<< "$row"; printf '\001')
    /usr/bin/time -a -o "$out/$arm.time" -f "$case_id %e %M" \
      "$chat" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 \
      --say "${prompt%$'\001'}" --max-tokens 12 --temperature 0 \
      --authenticated-copy --native-period-release 2>> "$out/$arm.stderr" \
      | jq -ce --arg id "$case_id" --arg arm "$arm" \
          '{case_id:$id,arm:$arm,client:.}' >> "$out/$arm.jsonl"
  done < "$prompts"
}
pids=()
for arm in zero rowwise paired; do
  serve "$arm" "${!arm}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]]
sha256sum --strict --quiet -c "$out/INPUTS.sha256"
(cd "$out" && sha256sum INPUTS.sha256 *.jsonl *.time *.stderr > SHA256SUMS)
