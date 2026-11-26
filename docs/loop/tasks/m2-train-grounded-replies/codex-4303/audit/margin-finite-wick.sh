#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
build=${1:?exact audit build}
capture=${2:?returned GPU M finite capture}
out=${3:?new output directory}
[[ $# == 3 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${out%/*}") == zfs ]]
for path in "$build" "$capture"; do
    (cd "$path" && sha256sum --strict --quiet -c SHA256SUMS)
done
mkdir "$out"
f=$capture/margin-finite
for id in first known repeat; do
    "$build/slot3-audit" synthetic-margin-capture "$f/text.bin" "$f/requests.tsv" "$f/fixture.words" \
        "$f/$id" "$out/$id.json"
done
cmp "$out/first.json" "$out/repeat.json"
diff -r "$f/first" "$f/repeat"
grep -Fq 'unmatched sign, repeated row, or unexchanged original-site ordinals' "$f/rejected-pair.log"
jq -ne --slurpfile a "$out/first.json" --slurpfile b "$out/known.json" '
    $a[0].candidates.start.terms as $base|$b[0].candidates.start.terms as $known|
    (any($a[0].candidates|to_entries[]; .key!="start" and all(.value.terms[0:4][];.strict_win))) as $learned|
    if $learned and all(range(4); . as $i|
            [$base[$i].margin,$known[$i].margin,$base[$i].joint[1].min_sites,$known[$i].joint[1].min_sites,
             $base[$i].strict_win,$known[$i].strict_win] == [-2,2,[2],[3],false,true])
       and all($base[4:6][]; .joint[1].min_sites == [2,3] and .joint[1].ties == 3 and (.strict_win|not))
       and all($base[6:8][]; .opportunity == "no_third" and .margin == null)
       and all($base[8:10][]; .opportunity == "target_non_strict_sign" and .margin == null) then
        {candidate_rows:($a[0].candidate_rows+$b[0].candidate_rows),
         independent_matched_fields:($a[0].scalar_fields+$b[0].scalar_fields),
         repeated_run:"exact",wrong_pair:"rejected",synthetic_rerank:"verified"}
    else error("M finite failed the independent strict third or absence contrast") end' > "$out/summary.json"
sha256sum "$build/SHA256SUMS" "$capture/SHA256SUMS" "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json input.sha256 > SHA256SUMS)
