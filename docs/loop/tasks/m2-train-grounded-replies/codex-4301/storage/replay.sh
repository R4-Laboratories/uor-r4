#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
root=$(realpath -e "${1:?extracted packet}")
out=${2:?new verification directory}
mkdir "$out"; out=$(realpath -e "$out")
(cd "$root" && sha256sum --strict --quiet -c MEMBERS.sha256)
grade="$root/methods/grade"
bash "$grade/all.sh" "$root/panel" "$root/primary/score" "$root/primary/fit-out/candidates.tsv" \
    "$out/training" train regression near
if [[ -d "$root/recovery" ]]; then
    bash "$grade/all.sh" "$root/panel" "$root/recovery/scores" "$root/recovery/states/candidates.tsv" \
        "$out/extra" train regression near
    for d in "$out"/extra/*; do ln -s "$d" "$out/training/"; done
fi
bash "$grade/election.sh" "$out/training" "$root/expected/all-states.tsv" "$out/election"
cmp "$out/election/election.json" "$root/expected/election.json"
bash "$grade/all.sh" "$root/panel" "$root/held/score" "$root/held/readback/states.tsv" "$out/fresh" fresh
while IFS=$'\t' read -r id payload; do
    cmp "$out/fresh/$id/fresh.json" "$root/expected/fresh/$id/fresh.json"
    cmp "$out/fresh/$id/fresh.rows.jsonl" "$root/expected/fresh/$id/fresh.rows.jsonl"
    actual=$("$root/models/bundle" unpack "$root/held/readback/$id.bridge" "$out/$id.state")
    [[ $actual == ordered ]]
    cmp "$root/held/readback/$payload" "$out/$id.state"
done < "$root/held/readback/states.tsv"
jq -en --slurpfile gold "$root/panel/fresh/gold.jsonl" \
    --slurpfile pos "$root/held/controls/gold.jsonl" --slurpfile neg "$root/held/controls/swapped.jsonl" \
    -f "$grade/controls.jq" > "$out/controls.json"
cmp "$out/controls.json" "$root/expected/controls.json"
sha256sum "$out/election/election.json" "$out/controls.json"
