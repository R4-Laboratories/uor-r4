#!/usr/bin/env bash
# Regression on the frozen ordinary writer's observed, naturally proposed EOS.
set -euo pipefail
umask 077
r=$(realpath "${1:?Wick run root}")
new=$(realpath "${2:?corrected ordinary client}")
out=${3:?fresh output}
base=$(dirname "$r")/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
mkdir "$out"
cd "$out"
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 055b23c038e923d3b41473e5d0ed59f9b590d8893500dd316b0bfe4924115867 "$r/fit-v1/held.keys.jsonl"
pin a98c09c8a09cc897beff0406b46e720fdc57308d11c2344d71d723ee8f6172c3 "$r/client/stage-v1/FILES.sha256"
(cd "$r/client/stage-v1"; sha256sum --strict --quiet -c FILES.sha256)
row=$(jq -es 'map(select(.ordinary_id==1)) | select(length==1) | .[0].ordinal+1' "$r/fit-v1/held.keys.jsonl")
csv=$(sed -n "${row}p" "$r/client/stage-v1/held.prefix.jsonl" | jq -er '.prefix|map(tostring)|join(",")')
flags=(--protocol 2 --prompt-ids "$csv" --max-tokens 3)
"$r/client/base-chat" --discrete "$base" "${flags[@]}" > legacy.json
"$new" --discrete "$base" "${flags[@]}" --ordinary-termination-trace > traced.json
"$new" --discrete "$r/fit-v1/geometry.1.d11" "${flags[@]}" > wrapped-default.json
"$new" --discrete "$r/fit-v1/geometry.1.d11" "${flags[@]}" --ordinary-termination off > wrapped-off.json
"$new" --discrete "$r/fit-v1/geometry.1.d11" "${flags[@]}" --ordinary-termination off --ordinary-termination-trace > wrapped-trace.json
"$new" --discrete "$r/serve-v1/zero.d11" "${flags[@]}" --ordinary-termination generic > active-zero.json
jq -e '.generated_ids==[1,73,1010]' legacy.json >/dev/null
for case in traced wrapped-default wrapped-off wrapped-trace; do
    jq -e --slurpfile old legacy.json '.generated_ids==$old[0].generated_ids and .stop=="max_tokens"' "$case.json" >/dev/null
done
for case in traced wrapped-trace; do
    jq -e '.ordinary_termination.steps | map(.committed_id)==[1,73,1010]' "$case.json" >/dev/null
done
jq -e '.generated_ids==[1] and .stop=="eos"' active-zero.json >/dev/null
"$new" --discrete "$r/fit-v1/geometry.1.d11" --protocol 2 --prompt-ids "$csv" --max-tokens 1 --ordinary-termination off --ordinary-termination-trace > capped-eos.json
jq -e '.generated_ids==[1] and .stop=="max_tokens"' capped-eos.json >/dev/null
sha256sum "$new" *.json >VERIFIED.sha256
sha256sum VERIFIED.sha256
