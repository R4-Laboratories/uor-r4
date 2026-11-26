#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
root=${1:?Wick study root}
build=${2:?named audit executable directory}
election=${3:?named TRAIN election directory}
capture=${4:?named returned held capture directory}
census=${5:?named fresh census directory}
permit=${6:?signed held fitness directory}
out=${7:?new audit output directory}
[[ $# == 7 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$root") == zfs ]]
[[ $out == "$root/audit/"* && $build == "$root/audit/build-"* ]]
runs=${root%/*}
panel="$root/panel/fresh"
held="$capture/held-raw"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
banks="$root/shipment/train-input/model/primary.words"
for input in "$capture" "$build" "$election" "$census" "$permit" "$panel"; do
    (cd "$input" && sha256sum --strict --quiet -c SHA256SUMS)
done
sha256sum --strict --quiet -c "$panel/authority.sha256"
mkdir "$out"
while IFS=$'\t' read -r id digest; do
    [[ $id =~ ^[a-z0-9-]+$ && $digest =~ ^[0-9a-f]{64}$ ]]
    state="$held/readback/$id.state"
    printf '%s  %s\n' "$digest" "$state" | sha256sum --strict --quiet -c -
    raw="$held/score/fresh/$id"
    "$build/paired-audit" replay "$panel" "$codec" "$state" "$banks" "$raw.native.jsonl" \
        "$raw.costs.jsonl" "$out/$id.json"
    printf '%s\t%s\n' "$id" "$out/$id.json" >> "$out/map.tsv"
done < "$election/states.sha.tsv"
"$build/paired-audit" physical-controls "$panel" "$codec" "$held/controls/gold.jsonl" \
    "$held/controls/swapped.jsonl" "$out/controls.json"
selected=$(jq -er .selected "$election/election.json")
learned=$(jq -er .best_learned "$election/election.json")
for id in "$selected" "$learned"; do [[ $id =~ ^[a-z0-9-]+$ ]]; done
"$build/paired-audit" compare-fresh "$out/$selected.json" "$out/original.json" "$out/median.json" "$out/intersections.json"
"$build/paired-audit" compare-fresh "$out/$learned.json" "$out/original.json" "$out/median.json" "$out/diagnostic-intersections.json"
sha256sum "$capture/SHA256SUMS" "$build/SHA256SUMS" "$election/SHA256SUMS" "$census/SHA256SUMS" \
    "$permit/SHA256SUMS" "$panel/SHA256SUMS" "$panel/authority.sha256" "$banks" "$codec" > "$out/input.sha256"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
