#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
reads=$(realpath -e "${1:?per-state grade directory}")
manifest=$(realpath -e "${2:?ordered manifest}")
out=${3:?unused election directory}
mkdir "$out"
ordinal=0
while IFS=$'\t' read -r id state; do
    [[ -n $state && $id =~ ^[a-zA-Z0-9_-]+$ ]]
    case "$id" in fit-*) kind=learned ;; recovery-*) kind=recovery ;; *) kind=initial ;; esac
    jq -enc --arg id "$id" --arg kind "$kind" --argjson ordinal "$ordinal" \
        --slurpfile t "$reads/$id/train.json" --slurpfile r "$reads/$id/regression.json" \
        --slurpfile n "$reads/$id/near.json" \
        '{id:$id,kind:$kind,ordinal:$ordinal,train:$t[0],regression:$r[0],near:$n[0]}' >> "$out/records.jsonl"
    ordinal=$((ordinal+1))
done < "$manifest"
jq -es -f "$here/elect.jq" "$out/records.jsonl" > "$out/election.json"
(cd "$out" && sha256sum records.jsonl election.json > SHA256SUMS)
