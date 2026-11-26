#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
root=${1:?Wick study root}
build=${2:?named audit executable directory}
election=${3:?named completed election directory}
capture=${4:?named returned TRAIN capture directory}
out=${5:?new audit output directory}
runs=${root%/*}
here=$(cd "$(dirname "$0")" && pwd -P)
[[ $# == 5 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$root") == zfs ]]
[[ $out == "$root/audit/"* && $build == "$root/audit/build-"* && $here == "$root/audit/source-"* ]]
for input in "$build" "$election" "$capture" "$root/shipment/train-input"; do
    (cd "$input" && sha256sum --strict --quiet -c SHA256SUMS)
done
p="$root/panel"
(cd "$p" && sha256sum --strict --quiet -c training.sha256 training-authority.sha256)
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
premise="$runs/20261126-codex-fixed-assignment-median/audit/premise-v1/readback.json"
banks="$root/shipment/train-input/model/primary.words"
mkdir "$out"
ordinal=0
while IFS=$'\t' read -r id path; do
    [[ $id =~ ^[a-z0-9-]+$ && $path == "$id.state" ]]
    state="$capture/fit-out/$path"
    lane="$out/$id"
    mkdir "$lane"
    for part in train regression; do
        raw="$capture/score/$part/$id"
        "$build/paired-audit" replay "$p/$part" "$codec" "$state" "$banks" \
            "$raw.native.jsonl" "$raw.costs.jsonl" "$lane/$part.json"
    done
    "$build/paired-audit" near-replay "$p/near" "$codec" "$premise" "$p/near/gold.jsonl" \
        "$state" "$banks" "$capture/score/near/$id.native.jsonl" "$lane/near.json"
    case "$id" in fit-*) kind=learned ;; recovery-*) kind=recovery ;; *) kind=initial ;; esac
    jq -nec --arg state "$id" --argjson ordinal "$ordinal" --arg kind "$kind" \
        --slurpfile reads "$lane/train.json" --slurpfile regression "$lane/regression.json" \
        --slurpfile near "$lane/near.json" --slurpfile current "$capture/score/train/$id.costs.jsonl" \
        -f "$here/election-replay.jq" >> "$out/records.jsonl"
    ordinal=$((ordinal + 1))
done < "$election/all-states.tsv"
cmp "$election/all-states.tsv" "$capture/fit-out/candidates.tsv"
jq -nce --slurpfile computed "$out/records.jsonl" --slurpfile sealed "$election/records.jsonl" \
    --slurpfile election "$election/election.json" '
    def eligible: (.near|length)==24 and (.regression|length)==4;
    def priority:
        (if eligible then [0,0,0] else [1,-((.near|length)+(.regression|length)),-(.regression|length)] end) +
        [- (.train|length), -.local, .missing, .worst, .sum, -(.train_full|length), -.third, .ordinal];
    def admissible_group($records): [$records[] | select(eligible) | .id];
    if ($computed | length)!=$election[0].candidates or ($computed | length)!=($sealed | length) then
        error("different election population")
    else . end |
    if all(range($computed|length); . as $i | $computed[$i] as $a | $sealed[$i] as $b |
        [$a.id,$a.kind,$a.ordinal,$a.train,$a.train_full,$a.near,$a.regression,$a.local,$a.third,$a.missing,$a.worst,$a.sum] ==
        [$b.id,$b.kind,$b.ordinal,($b.train.source_ids|sort),($b.train.full_ids|sort),($b.near.wins|sort),
         ($b.regression.source_ids|sort),$b.train.local,$b.train.strict_third,$b.train.missing_margins,$b.train.worst_deficit,$b.train.sum_deficit])
    then . else error("an independent candidate metric differs") end |
    ($computed | sort_by(priority)) as $ranked |
    (admissible_group($ranked)) as $admissible |
    ([$ranked[] | select(.kind=="learned")|.id] | first) as $learned |
    ([$ranked[] | select(.kind=="initial")|.id] | first) as $initial |
    if ($ranked|map(.id)) != ($election[0].ranking|map(.id)) or $admissible != $election[0].admissible_ids
        or $learned != $election[0].best_learned or $initial != $election[0].best_initial then
        error("independent election differs")
    else {candidates:($computed|length),selected:$ranked[0].id,admissible:$admissible,best_learned:$learned,best_initial:$initial}
    end' > "$out/summary.json"
jq -se 'map(.scalar_fields)|add' "$out"/*/{train,regression,near}.json > "$out/scalar-fields.json"
sha256sum "$build/SHA256SUMS" "$capture/SHA256SUMS" "$election/SHA256SUMS" \
    "$p/training.sha256" "$codec" "$premise" "$banks" "$here/"{train-election.sh,election-replay.jq} > "$out/input.sha256"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
