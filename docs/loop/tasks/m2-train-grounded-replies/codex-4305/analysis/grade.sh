#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
panel=$(realpath -e "$1"); physical=$(realpath -e "$2"); out=$3
here=$(cd "$(dirname "$0")" && pwd -P)
controls="$here/controls.jq"
if [[ ! -f $controls ]]; then controls="$here/../../codex-4303/panel/held-controls.jq"; fi
mkdir "$out"
out=$(realpath -e "$out")
arms=(baseline G N GN wrong-G wrong-N GN-wrong-G GN-wrong-N GN-wrong-both)
for part in regression near fresh; do
    mkdir "$out/$part"
    jq -nRc -f "$here/masks.jq" "$panel/$part/masks.tsv" > "$out/$part/masks.jsonl"
    near=/dev/null
    if [[ $part == near ]]; then near="$panel/near/gold.jsonl"; fi
    for arm in "${arms[@]}"; do
        jq -enc --arg part "$part" --slurpfile source "$panel/$part/rows.jsonl" \
            --slurpfile meta "$panel/$part/requests.jsonl" --slurpfile near "$near" \
            --slurpfile masks "$out/$part/masks.jsonl" --slurpfile reads "$physical/$part/$arm.jsonl" \
            -f "$here/rows.jq" > "$out/$part/$arm.rows.jsonl"
        jq -es --arg arm "$arm" --arg part "$part" -f "$here/metrics.jq" \
            "$out/$part/$arm.rows.jsonl" > "$out/$part/$arm.json"
    done
    for arm in "${arms[@]}"; do
        jq -c . "$out/$part/$arm.json"
    done > "$out/$part/arms.jsonl"
    jq -es -f "$here/transitions.jq" "$out/$part/arms.jsonl" > "$out/$part/transitions.json"
done
for part in regression near fresh; do
    cmp "$physical/$part/baseline.jsonl" "$physical/$part/native-reference.jsonl"
    cmp "$physical/$part/baseline.jsonl" "$physical/$part/ordinary.jsonl"
done
jq -ne --slurpfile gold "$panel/fresh/gold.jsonl" --slurpfile geometry "$panel/fresh/rows.jsonl" \
    --slurpfile correct "$physical/controls/gold.jsonl" --slurpfile swap "$physical/controls/swapped.jsonl" \
    -f "$controls" > "$out/controls.json"
(cd "$out" && find regression near fresh controls.json -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
