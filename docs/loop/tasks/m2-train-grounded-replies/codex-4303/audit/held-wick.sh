#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
root=${1:?study root}
build=${2:?exact audit build}
capture=${3:?returned physical held root}
release=${4:?sealed fresh-input release}
out=${5:?new audit output directory}
[[ $# == 5 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${out%/*}") == zfs ]]
runs=${root%/*}
panel=$root/panel/fresh
codec=$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json
banks=$root/shipment/train-input/models/primary.words
bundle=$root/shipment/train-input/models/bundle
for path in "$build" "$capture" "$release" "$panel"; do
    (cd "$path" && sha256sum --strict --quiet -c SHA256SUMS)
done
held=$capture/held
audit=$build/slot3-audit
cmp "$panel/ordered/text.bin" "$release/ordered/text.bin"
cmp "$panel/requests.tsv" "$release/requests.tsv"
for id in gold swapped; do cmp "$panel/$id-aid.bin" "$release/$id-aid.bin"; done
mkdir "$out"
mkdir "$out/readback"
declare -A witnessed
base=$held/bundle/baseline.state
while IFS=$'\t' read -r alias id path hash; do
    [[ $alias =~ ^(baseline|margin|eligibility|combined|wrong)$ && $id =~ ^[a-z0-9-]+$ && $hash =~ ^[0-9a-f]{64}$ ]]
    [[ -z ${witnessed[$alias]+set} ]]
    witnessed[$alias]=1
    state=$held/bundle/$alias.state
    printf '%s  %s\n' "$hash" "$state" | sha256sum --strict --quiet -c -
    [[ $("$bundle" unpack "$held/bundle/$alias.bridge" "$out/readback/$alias.state") == ordered ]]
    cmp "$out/readback/$alias.state" "$state"
    raw=$held/score/fresh/$alias
    "$audit" slot3-replay "$panel" "$codec" "$state" "$banks" "$raw.native.jsonl" "$raw.costs.jsonl" \
        "$base" "$out/$alias.json"
    "$audit" physical-words "$panel/ordered/text.bin" "$state" "$held/probe/$alias.jsonl" "$out/$alias-words.json"
done < "$release/arms.tsv"
[[ ${#witnessed[@]} == 5 ]]
"$audit" factorial "$base" "$held/bundle/margin.state" "$held/bundle/eligibility.state" \
    "$held/bundle/combined.state" "$out/factorial.json"
"$audit" physical-controls "$panel" "$codec" "$held/controls/gold.jsonl" "$held/controls/swapped.jsonl" "$out/controls.json"
jq -se 'map(.scalar_fields)|add' "$out"/{baseline,margin,eligibility,combined,wrong}{,-words}.json > "$out/fields.json"
for alias in baseline margin eligibility combined wrong; do
    jq -c --arg alias "$alias" '
      {alias:$alias, state_sha256, metrics,
       target_eligibility:([.rows[].lexical.status]|group_by(.)|map({key:.[0],value:length})|from_entries),
       fixed_relations:([.rows[].fixed_third.relation]|group_by(.)|map({key:.[0],value:length})|from_entries),
       free_relations:([.rows[].free_third.correct_governor_relation]|group_by(.)|map({key:.[0],value:length})|from_entries),
       retain_status:([.rows[]|select(.affirmative_verb=="retain")|.authored_lexical[0].status]|group_by(.)|map({key:.[0],value:length})|from_entries),
       control_verbs:([.rows[]|select(.verb!="retain")|{verb,state:.lexical.status}]|group_by(.verb)|map(
         {verb:.[0].verb,status:(map(.state)|group_by(.)|map({key:.[0],value:length})|from_entries)})),
       membership_changes:.all_word_membership_changes}' "$out/$alias.json"
done | jq -es '.' > "$out/summary.json"
sha256sum "$build/SHA256SUMS" "$capture/SHA256SUMS" "$release/SHA256SUMS" "$panel/SHA256SUMS" \
    "$codec" "$banks" "$bundle" "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 readback/*.state > SHA256SUMS)
