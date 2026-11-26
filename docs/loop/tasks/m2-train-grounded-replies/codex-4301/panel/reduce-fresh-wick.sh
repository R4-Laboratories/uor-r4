#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd -P)
panel=$(realpath -e "${1:?existing sealed fresh packet}")
grade=$(realpath -e "${2:?existing per-state grades}")
raw=$(realpath -e "${3:?existing physical fresh cost JSONL directory}")
election=$(realpath -e "${4:?existing sealed election JSON}")
audit_map=${5:?audit TSV of state ID and independent replay JSON, or - for provisional}
out=${6:?new result directory}
[[ $# == 6 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$(dirname "$out")") == zfs ]]

pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
pin 34699010b8cf175a8b3544e97088d35f0b844dc226e4ecda0693fb8c10382e30 "$panel/SHA256SUMS"
(cd "$panel" && sha256sum --strict --quiet -c SHA256SUMS)
if [[ $audit_map != - ]]; then
    audit_map=$(realpath -e "$audit_map")
    awk -F '\t' 'NF != 2 || $1 !~ /^[0-9A-Za-z_-]+$/ || $2 !~ /^\// || seen[$1]++ {exit 1}' "$audit_map"
fi
mkdir "$out"

jq -er 'select(.selection=="diagnostic" or .selection=="admissible") |
    [.selected,"original","median",.best_initial,.best_learned] |
    reduce .[] as $id ([]; if $id==null or index($id)!=null then . else .+[$id] end) | .[]' \
    "$election" > "$out/order.txt"
remember() {
    local digest
    digest=$(sha256sum "$1")
    printf '%s  %s\n' "${digest%% *}" "$2" >> "$out/sources.sha256"
}
remember "$panel/SHA256SUMS" panel/SHA256SUMS
remember "$election" election/election.json
for source in reduce-fresh-wick.sh fresh-state.jq fresh-comparison.jq pairs.jq; do
    remember "$here/$source" "source/$source"
done

while IFS= read -r id; do
    [[ $id =~ ^[a-zA-Z0-9_-]+$ ]]
    base="$grade/$id/fresh"
    cmp <(jq -ecs -f "$here/pairs.jq" "$base.rows.jsonl") "$base.json"
    jq -e '.pairs==128 and .rows==256 and (.by_family|length)==4' "$base.json" >/dev/null
    remember "$base.json" "grade/$id/fresh.json"
    remember "$base.rows.jsonl" "grade/$id/fresh.rows.jsonl"
    remember "$raw/$id.costs.jsonl" "raw/$id.costs.jsonl"

    args=(--argjson observed '[]')
    if [[ $audit_map != - ]]; then
        audit=$(awk -F '\t' -v id="$id" '$1==id {print $2}' "$audit_map")
        [[ -f $audit ]]
        jq -e '(.scalar_fields>0) and (.rows|length)==256 and (.pairs|length)==128' "$audit" >/dev/null
        args=(--slurpfile observed "$audit")
        remember "$audit" "audit/$id/fresh.json"
    fi
    jq -ecn "${args[@]}" --argjson expected 256 --slurpfile raw "$raw/$id.costs.jsonl" \
        --slurpfile physical "$panel/rows.jsonl" --slurpfile grade "$base.rows.jsonl" \
        -f "$here/fresh-state.jq" > "$out/$id.rows.jsonl"
    jq -ecn --arg id "$id" --slurpfile rows "$out/$id.rows.jsonl" '{id:$id,rows:$rows}'
done < "$out/order.txt" | jq -ecs --slurpfile election "$election" -f "$here/fresh-comparison.jq" > "$out/comparison.json"

if [[ $audit_map == - ]]; then
    printf 'Provisional physical reduction. Independent tuple minima are pending.\n' > "$out/receipt.txt"
else
    printf 'Physical reduction with independent agreement and all minimizing original tuple ordinals.\n' > "$out/receipt.txt"
fi
(
    cd "$out"
    sha256sum *.rows.jsonl comparison.json order.txt sources.sha256 receipt.txt > SHA256SUMS
    sha256sum --strict --quiet -c SHA256SUMS
)
chmod -R a-w "$out"
sha256sum "$out/SHA256SUMS"
