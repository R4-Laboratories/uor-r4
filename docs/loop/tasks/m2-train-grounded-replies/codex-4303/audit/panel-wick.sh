#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
build=${1:?exact independent audit build}
panel=${2:?sealed panel root}
codec=${3:?frozen tokenizer}
out=${4:?new output directory}
[[ $# == 4 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${out%/*}") == zfs ]]
for path in "$build" "$panel/train" "$panel/fresh"; do
    (cd "$path" && sha256sum --strict --quiet -c SHA256SUMS)
done
mkdir "$out"
audit=$build/slot3-audit
"$audit" split-census "$panel/train" "$panel/fresh" "$codec" "$out"
for cohort in train fresh; do
    p=$panel/$cohort
    "$audit" geometry "$p" "$codec" "$p/endpoint/text.bin" "$out/$cohort-geometry.json"
    cmp "$p/requests.tsv" <(jq -r '[.row_index,.sign,.governor_word_index,.requested_site_index,.rival_site_index]|@tsv' "$p/requests.jsonl")
    jq -se '. as $rows | all(range(0;length;2); . as $i|
        $rows[$i].row_index==$i and $rows[$i+1].row_index==($i+1) and
        $rows[$i].selector=="first" and $rows[$i+1].selector=="later" and
        $rows[$i].pair_id==$rows[$i+1].pair_id)' "$p/requests.jsonl" > "$out/$cohort-order.json"
done
sha256sum "$build/SHA256SUMS" "$panel/train/SHA256SUMS" "$panel/fresh/SHA256SUMS" "$codec" \
    "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 > SHA256SUMS)
