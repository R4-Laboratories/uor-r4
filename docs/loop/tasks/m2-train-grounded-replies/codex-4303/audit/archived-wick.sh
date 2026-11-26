#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=${1:?Wick runs root}
build=${2:?exact audit build}
finite=${3:?returned E and scorer bundle}
out=${4:?new output directory}
[[ $# == 4 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${out%/*}") == zfs ]]
previous=$runs/20261126-codex-paired-occurrence-margin
panel=$previous/panel
train=$previous/rented-train-v1
fresh=$previous/rented-fresh-v1
state=$train/fit-out/median-bilateral-3.state
codec=$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json
projection=$runs/20261125-codex-governing-source-pairs/models/fold-8/ordered-all.d11
premise=$runs/20261126-codex-fixed-assignment-median/audit/premise-v1/readback.json
banks=$previous/shipment/train-input/model/primary.words
for path in "$build" "$panel/fresh" "$finite"; do
    (cd "$path" && sha256sum --strict --quiet -c SHA256SUMS)
done
(cd "$panel" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
printf '%s  %s\n' \
    e1adbc1c8edbc0e30fa81e610d5fd47baf7ea44e178bb78ae11a44d4fe4f7977 "$state" \
    5c843187d34c3a6ca94560a161c8d82c3655d16dd65e283c36753d1e9f55e7b8 "$projection" \
    ace9145ac71f66a01e9fed62bb33357634270d4be8ec4f1604e865b134a3a663 "$banks" \
    a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377 "$codec" | sha256sum --strict --quiet -c -
legacy_score=$previous/shipment/score-code
legacy_src=$legacy_score/source
for name in score.cu switch-competition.cu textlinks-shared.cuh; do
    cmp "$legacy_src/$name" "$finite/native/scorer/source/$name"
done
mkdir "$out"
audit=$build/slot3-audit
suffix=median-bilateral-3
raw="$train/score/regression/$suffix"
"$audit" replay "$panel/regression" "$codec" "$state" "$banks" "$raw.native.jsonl" "$raw.costs.jsonl" "$out/regression.json"
raw="$train/score/near/$suffix"
"$audit" near-replay "$panel/near" "$codec" "$premise" "$panel/near/gold.jsonl" \
    "$state" "$banks" "$raw.native.jsonl" "$out/near.json"
raw="$fresh/held-raw/score/fresh/$suffix"
"$audit" replay "$panel/fresh" "$codec" "$state" "$banks" "$raw.native.jsonl" "$raw.costs.jsonl" "$out/fresh.json"
for part in regression near; do cmp "$out/$part.json" "$previous/audit/train-election-v1/$suffix/$part.json"; done
cmp "$out/fresh.json" "$previous/audit/fresh-replay-v1/$suffix.json"
jq -e '[.pairs[].follows]==[true,true,true,true]' "$out/regression.json" > /dev/null
jq -e '(.protected_pairs|length)==24 and (.lost_protected==[]) and (.members|length)==48' "$out/near.json" > /dev/null
jq -ne --slurpfile regression "$out/regression.json" --slurpfile near "$out/near.json" --slurpfile fresh "$out/fresh.json" '
    {regression_ids:[$regression[0].pairs[].id],near_ids:($near[0].protected_pairs|keys),
     matched_scalar_fields:([$regression[0].scalar_fields,$near[0].scalar_fields,$fresh[0].scalar_fields]|add),
     historical_fresh:($fresh[0].rows|group_by(.fixed_third.relation)|map({key:.[0].fixed_third.relation,value:length})|from_entries)}' \
    > "$out/summary.json"
sha256sum "$build/SHA256SUMS" "$finite/SHA256SUMS" "$state" "$projection" "$banks" "$codec" "$premise" \
    "$train/SHA256SUMS" "$fresh/SHA256SUMS" "$panel/training.sha256" "$panel/fresh/SHA256SUMS" \
    "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 > SHA256SUMS)
