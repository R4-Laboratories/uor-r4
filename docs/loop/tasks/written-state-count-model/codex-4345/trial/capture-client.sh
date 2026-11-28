#!/usr/bin/env bash
# Inputs are only original causal IDs. The ordinary client never receives true targets.
set -euo pipefail
umask 077
chat=$(realpath "${1:?chat executable}")
artifact=$(realpath "${2:?artifact}")
pin=${3:?artifact sha256}
rows=$(realpath "${4:?opening or boundary rows}")
kind=${5:?open or boundary}
mode=${6:?base, off or generic}
dest=${7:?fresh output}
tokenizer=$(realpath "${8:?bound tokenizer}")
printf '%s  %s\n' a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$tokenizer" | sha256sum --strict -c - >&2
[[ $kind == open || $kind == boundary ]]
[[ $mode == base || $mode == off || $mode == generic ]]
printf '%s  %s\n' "$pin" "$artifact" | sha256sum --strict -c - >&2
mkdir -m 700 "$dest"
dest=$(realpath "$dest")
case "$kind" in open) steps=192 ;; boundary) steps=1 ;; esac
count=0
while IFS= read -r row; do
    csv=$(jq -er '.prefix | map(tostring) | join(",")' <<<"$row")
    flags=(--discrete "$artifact" --protocol 2 --max-tokens "$steps")
    if [[ $kind == open ]] && jq -e '.user|type=="string"' >/dev/null <<<"$row"; then
        # The sentinel preserves any trailing line breaks in a command substitution.
        text=$(jq -jr '.user,"."' <<<"$row")
        flags+=(--tokenizer "$tokenizer" --say "${text%.}")
    else
        flags+=(--prompt-ids "$csv")
    fi
    if [[ $mode == base ]]; then
        flags+=(--ordinary-termination-trace)
    else
        flags+=(--ordinary-termination "$mode")
        if [[ $kind == boundary ]]; then flags+=(--ordinary-termination-trace); fi
    fi
    "$chat" "${flags[@]}" > "$dest/$count.json"
    count=$((count + 1))
done < "$rows"
(( count > 0 ))
(cd "$dest"; sha256sum -- *.json > FILES.sha256)
