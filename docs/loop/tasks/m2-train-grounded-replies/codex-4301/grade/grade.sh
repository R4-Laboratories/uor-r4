#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd -P)
panel=$(realpath -e "${1:?labeled packet root}")
raw=$(realpath -e "${2:?raw capture root}")
id=${3:?state id}
out=${4:?unused output directory}
shift 4
[[ $id =~ ^[a-zA-Z0-9_-]+$ && $# -gt 0 ]]
mkdir "$out"
for part in "$@"; do
    data="$panel/$part"
    run="$raw/$part/$id"
    if [[ $part == near ]]; then
        jq -en --slurpfile gold "$data/gold.jsonl" --slurpfile native "$run.native.jsonl" \
            -f "$here/near.jq" > "$out/near.json"
    else
        jq -enc --slurpfile rows "$data/rows.jsonl" --slurpfile meta "$data/requests.jsonl" \
            --slurpfile raw "$run.costs.jsonl" --slurpfile native "$run.native.jsonl" \
            -f "$here/rows.jq" > "$out/$part.rows.jsonl"
        jq -ecs -f "$here/pairs.jq" "$out/$part.rows.jsonl" > "$out/$part.json"
    fi
done
