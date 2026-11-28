#!/usr/bin/env bash
# Regrade the retained original records, with path rebasing for a cold restore.
set -euo pipefail
umask 077

if [[ $# != 4 ]]; then
    echo 'usage: regrade.sh RETAINED_RUN DECISION_REFERENCE PUBLISHED_SHA256 FRESH_OUTPUT' >&2
    exit 2
fi
run=$(realpath "$1")
reference=$(realpath "$2")
digest=$3
out=$(realpath -m "$4")
here=$(cd "$(dirname "$0")" && pwd)
recorded=/mnt/plexpool/uor-r4-corpus/runs/20261127-codex-grounded-interval

bind_input() {
    local manifest=$1 line=$2 sha path
    read -r sha path < <(sed -n "${line}p" "$manifest")
    if [[ "$path" == "$recorded/"* ]]; then path="$run/${path#"$recorded"/}"; fi
    printf '%s  %s\n' "$sha" "$path" | sha256sum --strict --quiet -c -
    printf '%s\n' "$path"
}

mkdir "$out"
shopt -s nullglob
prior=("$run"/graded-ordinary-fitted-v2/*/*/native "$run"/graded-ordinary-fitted-v2/*/*/stop
       "$run"/causal-v1/*/grades "$run"/fresh-balanced-v1/grades "$run"/graded-train-base-v2
       "$run"/grade-train-supplied-native "$run"/grade-train-supplied-stop)
if [[ ${#prior[@]} != 24 ]]; then
    echo 'expected all 24 archived grade directories' >&2
    exit 1
fi
for previous in "${prior[@]}"; do
    manifest="$previous/INPUTS.sha256"
    gold=$(bind_input "$manifest" 1)
    truth=$(bind_input "$manifest" 2)
    records=$(bind_input "$manifest" 3)
    visible=$(bind_input "$manifest" 6)
    arm=$(jq -er .arm "$previous/grade.json")
    relative=${previous#"$run"/}
    fresh="$out/$relative"
    mkdir -p "$(dirname "$fresh")"
    bash "$here/grade.sh" "$gold" "$truth" "$records" "$arm" "$fresh" "$visible" "$reference" "$digest" \
        > "$fresh.log" 2>&1
    cmp "$previous/grade.json" "$fresh/grade.json"
    cmp "$previous/summary.json" "$fresh/summary.json"
    printf '%s\t%s\t%s\n' "$relative" "$arm" "$(jq -c .summary.cases "$fresh/grade.json")"
done | tee "$out/REGRADES.tsv"
