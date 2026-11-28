#!/usr/bin/env bash
# Confirm the post-fit diagnostic-only correction on the sealed measured inputs.
set -euo pipefail
umask 077
r=$(realpath "${1:?Wick run root}")
out=${2:?fresh output}
mkdir -m 700 "$out"
cd "$out"
old=$r/client/bin-final/uor-chat
new=$(realpath "${3:-$r/client/bin-composition/uor-chat}")
stage=$r/client/stage-v1
fit=$r/fit-v1
base=$(dirname "$r")/20261127-codex-period-margin/cold-v1/balanced/artifact.d11
tokenizer=$(dirname "$r")/20261127-codex-visible-answer/panel/inputs/tokenizer.json
printf '%s  %s\n' a98c09c8a09cc897beff0406b46e720fdc57308d11c2344d71d723ee8f6172c3 "$stage/FILES.sha256" | sha256sum --strict --quiet -c -
(cd "$stage"; sha256sum --strict --quiet -c FILES.sha256)
for split in train held; do
    "$new" --discrete "$base" --ordinary-termination-capture "$stage/$split.prefix.jsonl" > "$split.keys"
    cmp "$fit/$split.keys.jsonl" "$split.keys"
done
for arm in geometry.1 geometry.2 ordered.1 ordered.2; do
    for row in 1 90; do
        text=$(sed -n "${row}p" "$stage/held.open.jsonl" | jq -jr '.user,"."')
        flags=(--discrete "$fit/$arm.d11" --protocol 2 --tokenizer "$tokenizer"
            --say "${text%.}" --max-tokens 192 --ordinary-termination generic)
        "$old" "${flags[@]}" > "frozen.$arm.$row.json"
        "$new" "${flags[@]}" > "final.$arm.$row.json"
        cmp "frozen.$arm.$row.json" "final.$arm.$row.json"
    done
done
sha256sum "$old" "$new" *.keys *.json >VERIFIED.sha256
sha256sum VERIFIED.sha256
