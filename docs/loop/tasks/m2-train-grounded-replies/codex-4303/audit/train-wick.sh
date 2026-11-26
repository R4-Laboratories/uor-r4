#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
mode=${1:?finalists or all}
root=${2:?study root}
build=${3:?exact independent audit build}
capture=${4:?returned primary TRAIN capture}
matrix=${5:?election state directory with factors.tsv}
raw=${6:?native and cost captures indexed by cohort and candidate}
election=${7:?sealed completed election}
out=${8:?new audit output directory}
[[ $# == 8 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${out%/*}") == zfs ]]
[[ $mode == finalists || $mode == all ]]
for path in "$build" "$election" "$root/panel/train"; do
    (cd "$path" && sha256sum --strict --quiet -c SHA256SUMS)
done
panel=$root/panel
runs=${root%/*}
codec=$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json
premise=$runs/20261126-codex-fixed-assignment-median/audit/premise-v1/readback.json
banks=$root/shipment/train-input/models/primary.words
audit=$build/slot3-audit
base=$matrix/b.state
printf '%s  %s\n' e1adbc1c8edbc0e30fa81e610d5fd47baf7ea44e178bb78ae11a44d4fe4f7977 "$base"|sha256sum --strict --quiet -c -
mkdir "$out"
mapfile -t finalists < <(jq -er '.choice | [.m,.e,.combined] | .[]' "$election/election.json")
[[ ${#finalists[@]} == 3 ]]
for id in "${finalists[@]}"; do [[ $id =~ ^[a-z0-9-]+$ ]]; done
"$audit" factorial "$base" "$matrix/${finalists[0]}.state" "$matrix/${finalists[1]}.state" \
    "$matrix/${finalists[2]}.state" "$out/factorial.json"
"$audit" eligibility-fit "$panel/train/ordered/text.bin" "$panel/train/retain.tsv" "$base" \
    "$capture/e-out" "$out/eligibility-fit.json"
for id in b e1 e2 e3 e4 wrong; do
    "$audit" physical-words "$panel/train/ordered/text.bin" "$capture/matrix/$id.state" \
        "$capture/train-probes/$id.jsonl" "$out/$id-lexical.json"
done
if [[ $mode == all ]]; then
    "$audit" margin-capture "$panel/train/ordered/text.bin" "$panel/train/requests.tsv" "$banks" \
        "$capture/fit-out" "$out/margin-fit.json"
fi
worker() {
    local id=$1 m=$2 e=$3 mid=$4 eid=$5 src=$6 sum=$7 part path lane
    [[ $id =~ ^[a-z0-9-]+$ && $mid =~ ^[a-z0-9-]+$ && $eid =~ ^[a-z0-9-]+$ && $sum =~ ^[0-9a-f]{64}$ ]]
    path=$matrix/$id.state
    lane=$out/$id
    mkdir "$lane"
    printf '%s  %s\n' "$sum" "$path" | sha256sum --strict --quiet -c -
    if (( m > 0 && e > 0 )); then
        "$audit" factorial "$base" "$matrix/$mid.state" "$matrix/$eid.state" "$path" "$lane/byte-factors.json"
    fi
    part=train
    "$audit" slot3-replay "$panel/$part" "$codec" "$path" "$banks" \
        "$raw/$part/$id.native.jsonl" "$raw/$part/$id.costs.jsonl" "$base" "$lane/$part.json"
    part=regression
    "$audit" replay "$panel/$part" "$codec" "$path" "$banks" \
        "$raw/$part/$id.native.jsonl" "$raw/$part/$id.costs.jsonl" "$lane/$part.json"
    "$audit" near-replay "$panel/near" "$codec" "$premise" "$panel/near/gold.jsonl" \
        "$path" "$banks" "$raw/near/$id.native.jsonl" "$lane/near.json"
    jq -nce --arg id "$id" --arg sha "$sum" --arg src "$src" --argjson m "$m" --argjson e "$e" --arg mid "$mid" --arg eid "$eid" \
        --slurpfile t "$lane/train.json" --slurpfile r "$lane/regression.json" --slurpfile n "$lane/near.json" '
        {id:$id,m:$m,e:$e,mid:$mid,eid:$eid,src:$src,sha:$sha,t:$t[0].metrics,
         regression_ids:[$r[0].pairs[]|select(.follows)|.id], near_ids:[$n[0].protected_pairs|to_entries[]|select(.value)|.key],
         fields:([$t[0].scalar_fields,$r[0].scalar_fields,$n[0].scalar_fields]|add)} |
        .reg=(.regression_ids|length) | .near=(.near_ids|length)' > "$lane/record.json"
}
export audit base panel codec matrix raw out banks premise
export -f worker
if [[ $mode == all ]]; then
    tail -n +2 "$matrix/factors.tsv"
else
    awk -F '\t' -v m="${finalists[0]}" -v e="${finalists[1]}" -v c="${finalists[2]}" \
        '$1=="b" || $1==m || $1==e || $1==c' "$matrix/factors.tsv"
fi | tr '\t\n' '\000\000' | xargs -0 -r -n7 -P4 bash -euc 'set -o pipefail; worker "$@"' _
jq -cs 'sort_by(.m,.e)' "$out"/*/record.json > "$out/records.json"
jq -ne --slurpfile independent "$out/records.json" --slurpfile sealed "$election/records.jsonl" \
    --arg mode "$mode" '
    ($sealed|INDEX(.id)) as $orig | $independent[0] as $items |
    def visible: .t|{source,source_ids,full,full_ids,fixed_pairs,fixed_ids,fixed_rows,blocked,worst,debt,retain_admit,other_losses}
        | .source_ids|=sort | .full_ids|=sort | .fixed_ids|=sort;
    def ids: [.sha,(.regression_ids|sort),(.near_ids|sort),.reg,.near];
    if (($mode=="all" and ($items|length)==($sealed|length)) or ($mode=="finalists" and ($items|length)==4)) and
       all($items[]; . as $i | ($orig[$i.id]|ids)==($i|ids) and ($orig[$i.id]|visible)==($i|visible)) then
        {candidates:($items|length),fields:($items|map(.fields)|add),exact:true}
    else error("independent original identity or election metrics differ") end' > "$out/agreement.json"
if [[ $mode == all ]]; then
    jq -e --slurpfile election "$election/election.json" -f "$(dirname "$0")/verify-election.jq" \
        "$out/records.json" > "$out/election-agreement.json"
fi
sha256sum "$build/SHA256SUMS" "$panel/train/SHA256SUMS" "$panel/guards.sha256" "$matrix/factors.tsv" \
    "$election/SHA256SUMS" "$codec" "$banks" "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
