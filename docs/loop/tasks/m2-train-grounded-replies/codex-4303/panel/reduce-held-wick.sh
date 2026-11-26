#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

[[ $# == 5 ]] || {
    echo "Usage: $0 FRESH_PACKET PHYSICAL_HELD GRADE_HELD SCRIPTED_AUDITS_OR_DASH NEW_OUT" >&2
    exit 1
}
here=$(cd "$(dirname "$0")" && pwd)
packet=$1
held=$2
grade=$3
audit=$4
out=$5
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$(dirname "$out")") == zfs ]]
(cd "$packet" && sha256sum --strict --quiet -c SHA256SUMS authority.sha256)
mkdir "$out"
sha256sum "$here/"{reduce-held-wick.sh,held-row.jq,held-controls.jq,held-summary.jq} \
    "$packet/"{SHA256SUMS,authority.sha256} \
    "$held/bundle/models.tsv" "$held/controls/"{gold,swapped}.jsonl > "$out/sources.sha256"
cmp "$held/probe/baseline.jsonl" "$held/probe/margin.jsonl"
cmp "$held/probe/eligibility.jsonl" "$held/probe/combined.jsonl"

jq -enc --slurpfile geometry "$packet/rows.jsonl" --slurpfile gold "$packet/gold.jsonl" \
    --slurpfile correct "$held/controls/gold.jsonl" --slurpfile swap "$held/controls/swapped.jsonl" \
    -f "$here/held-controls.jq" > "$out/controls.json"
for arm in baseline margin eligibility combined wrong; do
    costs="$held/score/fresh/$arm.costs.jsonl"
    native="$held/score/fresh/$arm.native.jsonl"
    probe="$held/probe/$arm.jsonl"
    reduced="$out/$arm.rows.jsonl"
    summary="$grade/$arm/fresh.json"
    review=/dev/null
    if [[ $audit != - ]]; then review="$audit/$arm.json"; fi
    jq -enc --slurpfile geometry "$packet/rows.jsonl" --slurpfile meta "$packet/requests.jsonl" \
        --slurpfile gold "$packet/gold.jsonl" --slurpfile registered "$packet/eligibility.jsonl" \
        --slurpfile reads "$native" --slurpfile costs "$costs" --slurpfile probe "$probe" \
        --slurpfile baseline_probe "$held/probe/baseline.jsonl" \
        --slurpfile review "$review" -f "$here/held-row.jq" > "$reduced"
    jq -se --slurpfile existing "$summary" '
        (map(select(.source))|length) == $existing[0].source_rows and
        (group_by(.pair_id)|map(select(all(.[];.source)))|length) == $existing[0].source and
        (group_by(.pair_id)|map(select(all(.[];.source))|.[0].pair_id)) == ($existing[0].source_ids|sort) and
        (group_by(.pair_id)|map(select(all(.[];.full)))|length) == $existing[0].full and
        (group_by(.pair_id)|map(select(all(.[];.full))|.[0].pair_id)) == ($existing[0].full_ids|sort) and
        (group_by(.pair_id)|map(select(all(.[];.pinned_third.status=="win"))|.[0].pair_id))
             == ($existing[0].fixed_ids|sort)' \
        "$reduced" >/dev/null
    sha256sum "$costs" "$native" "$probe" "$summary" >> "$out/sources.sha256"
    if [[ $audit != - ]]; then sha256sum "$review" >> "$out/sources.sha256"; fi
done
jq -enc --slurpfile controls "$out/controls.json" --slurpfile baseline "$out/baseline.rows.jsonl" \
    --slurpfile margin "$out/margin.rows.jsonl" --slurpfile eligibility "$out/eligibility.rows.jsonl" \
    --slurpfile combined "$out/combined.rows.jsonl" --slurpfile wrong "$out/wrong.rows.jsonl" \
    -f "$here/held-summary.jq" > "$out/summary.json"
(cd "$out" && sha256sum --strict --quiet -c sources.sha256 && sha256sum ./*.json ./*.jsonl sources.sha256 > SHA256SUMS)
