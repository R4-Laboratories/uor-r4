#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-protected-transport"
dir="$runs/20261126-codex-mixed-transport"
panel="$dir/panel/panel-v1/mixed"
old="$dir/measurement-v1/train/mixed"
words="$runs/20261125-codex-governing-source-pairs/fit-v1/primary.words"
here=$(cd "$(dirname "$0")" && pwd)
measurement=${1:?measurement version}
build=${2:?audit build version}
version=${3:?new audit version}
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
for id in "$measurement" "$build" "$version"; do [[ $id =~ ^v[1-9][0-9]*$ ]]; done
m="$s/measurement-$measurement"
p="$m/train"
out="$s/audit/train-$version"
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
verify "$m"
verify "$s/audit/build-$build"
sha256sum --strict --quiet -c "$m/selection.sha256"
jq -e '.pass and (.protected|length)==24 and (.far|length)==40 and
    .eligible==512 and .far_eligible==256 and .far_strict_losses==104' "$m/premise.json" >/dev/null
mkdir "$out"
start="$old/first/step-0.state"
sweep="$old/first/step-1.state"
readback(){
    local label=$1 base=$2 d="$p/$1"
    verify "$d/device"; verify "$d/grade"
    "$here/run-wick.sh" batch "$version-$label" "$build" "$words" "$panel/blind/text.bin"         "$panel/links.bin" "$start" "$sweep" "$base" "$d/device" "$panel/reference.json"
}
raw_audit(){
    printf '%s/audit/batch-%s-%s/readback.json\n' "$s" "$version" "$1"
}
readback root "$start"
jq -e --slurpfile prem "$m/premise.json" '
    .metrics[0] as $b | $prem[0] as $p |
    $b.near_full==$p.protected and $b.far_full==$p.far and
    $b.score.hinge==$p.far_hinge and $b.score.strict_wins==$p.far_strict_wins and
    $b.score.strict_losses==$p.far_strict_losses
' "$(raw_audit root)" > "$out/premise.json"
election(){
    local label=$1 d="$p/$1"
    jq -e --slurpfile origin "$(raw_audit root)" --slurpfile grade "$d/grade/candidates.jsonl"         --slurpfile device "$d/grade/election.json" -f "$here/election.jq" "$(raw_audit "$label")"         > "$out/$label.election.json"
    jq -e --slurpfile near "$d/grade/first_near_rejection.json"         --slurpfile near_far "$d/grade/first_far_improving_near_rejection.json" '
        def bounds: [.start,.end];
        def sites:
          if .==null then null else map({bytes:(.bytes|bounds),tokens:(.bpe|bounds)}) end;
        def sig:
          if .==null then null else [.pair,.row.row,.row.case_id,.row.admission,
            .row.original_nouns,.row.observed.tuple.addresses,.row.observed.tuple.ties] end;
        def grade:
          if .==null then null else [.pair,.ordinal,.case_id,.admission,
            (.gold|sites),(.chosen|sites),.ties] end;
        (.first_protected_reject|sig)==($near[0]|grade) and
        (.first_far_improving_protected_reject|sig)==($near_far[0]|grade)
    ' "$out/$label.election.json" > "$out/$label.witness-check.json"
}
selected_state(){
    local label=$1 arm=$2 id
    id=$(jq -r --arg a "$arm" '.[$a].id // "base"' "$out/$label.election.json")
    printf '%s/%s/device/%s.state\n' "$p" "$label" "$id"
}
election root
for arm in protected free; do
    first=$(selected_state root "$arm")
    cmp "$first" "$m/states/$arm-first.state"
    label=$(cat "$p/$arm.batch")
    [[ $label == root || $label == protected || $label == free ]]
    cmp "$first" "$p/$label/device/base.state"
    if [[ $label != root && ! -e "$out/$label.election.json" ]]; then
        readback "$label" "$first"
        election "$label"
    fi
    cmp "$(selected_state "$label" "$arm")" "$m/states/$arm.state"
    fingerprint=$(sha256sum "$m/states/$arm.state" | cut -d' ' -f1)
    jq -e --arg a "$arm" --arg sha "$fingerprint" --slurpfile selection "$m/selection.json" \
        '.[$a]==$selection[0][$a] and $sha==$selection[0].state_sha256[$a]' \
        "$out/$label.election.json" >/dev/null
done
batches=()
receipts=()
for label in root protected free; do
    if [[ -f "$out/$label.election.json" ]]; then
        receipts+=("$s/audit/batch-$version-$label/SHA256SUMS")
        batches+=("$label")
    fi
done
printf '%s\n' "${batches[@]}" | jq -eRsc '
 split("\n")[:-1] | {pass:true,batches:.,compared_unique_batches:length}
' > "$out/verified.json"
sha256sum "$m/SHA256SUMS" "$m/selection.sha256" "$m/selection.json" "$out/verified.json" \
    "$here/verify-train-wick.sh" "$here/election.jq" "$s/audit/build-$build/SHA256SUMS" "${receipts[@]}" \
    > "$out/inputs.sha256"
(cd "$out" && sha256sum --strict --quiet -c inputs.sha256 && sha256sum -- *.json inputs.sha256 > SHA256SUMS)
