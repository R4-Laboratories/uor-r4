#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
[[ $# == 2 ]]
root=$(realpath -e "$1")
out=$2
mkdir "$out"
out=$(realpath -e "$out")
audit="$root/independent/build-v3/paired-audit"
expected="$root/independent/fresh-replay"
cohort="$root/panel/fresh"
codec="$root/models/tokenizer.json"
bank="$root/models/primary.words"
while IFS=$'\t' read -r state payload; do
    "$audit" replay "$cohort" "$codec" "$root/held/readback/$payload" "$bank" \
        "$root/held/score/fresh/$state.native.jsonl" "$root/held/score/fresh/$state.costs.jsonl" "$out/$state.json"
    cmp "$out/$state.json" "$expected/$state.json"
done < "$root/held/readback/states.tsv"
"$audit" physical-controls "$cohort" "$codec" \
    "$root/held/controls/gold.jsonl" "$root/held/controls/swapped.jsonl" "$out/controls.json"
cmp "$out/controls.json" "$expected/controls.json"
selected=$(jq -er .selected "$root/expected/election.json")
"$audit" compare-fresh "$out/$selected.json" "$out/original.json" "$out/median.json" "$out/intersections.json"
cmp "$out/intersections.json" "$expected/intersections.json"
diagnostic=$(jq -er .best_learned "$root/expected/election.json")
"$audit" compare-fresh "$out/$diagnostic.json" "$out/original.json" "$out/median.json" "$out/diagnostic-intersections.json"
cmp "$out/diagnostic-intersections.json" "$expected/diagnostic-intersections.json"
lineage="$root/independent/fit-lineage"
budget=$(jq -er .proposal_evaluations "$root/primary/fit-out/complete.json")
"$lineage/lineage" "$root/models/protected.state" "$root/models/ordered.state" \
    "$root/panel/train/ordered/text.bin" "$root/primary/fit-out" "$lineage" "$budget" > "$out/fit-lineage.json"
cmp "$out/fit-lineage.json" "$lineage/checks.json"
(cd "$out" && sha256sum ./*.json > SHA256SUMS)
cat "$out/SHA256SUMS"
