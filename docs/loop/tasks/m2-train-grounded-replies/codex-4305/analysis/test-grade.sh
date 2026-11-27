#!/usr/bin/env bash
set -euo pipefail
# Finite readouts of a prior sealed GPU run check the grader before the new run.
prior=$(realpath -e "$1"); out=$2
here=$(cd "$(dirname "$0")" && pwd -P)
mkdir "$out"
panel="$prior/panel/fresh"
raw="$prior/physical/held"
jq -c '{row:.row_index,sign,g:.governor.index,wg:.opposite_governor.index,
     good:[.requested.index,.rival.index,.opposite.index,.heading.index],wrong:[]}' \
    "$panel/requests.jsonl" > "$out/masks.jsonl"
grade() {
    local in_file=$1
    jq -enc --arg part fresh --slurpfile source "$panel/rows.jsonl" --slurpfile meta "$panel/requests.jsonl" \
        --slurpfile near /dev/null --slurpfile masks "$out/masks.jsonl" --slurpfile reads "$in_file" \
        -f "$here/rows.jq"
}
for arm in baseline gold swapped; do
    cap="$raw/controls/$arm.jsonl"
    if [[ $arm == baseline ]]; then cap="$raw/score/fresh/baseline.native.jsonl"; fi
    grade "$cap" > "$out/$arm.rows.jsonl"
    jq -es --arg arm "$arm" --arg part fresh -f "$here/metrics.jq" \
        "$out/$arm.rows.jsonl" > "$out/$arm.json"
done
jq -en --slurpfile b "$out/baseline.json" --slurpfile g "$out/gold.json" --slurpfile s "$out/swapped.json" \
    '$b[0].signed==0 and $b[0].pairs==128 and $g[0].signed==128 and $g[0].full==128 and
     $s[0].signed==0 and $s[0].full==0 and $s[0].source_kinds["same-spelling rival"]==256'
jq -c 'if .row==0 then .tuple += {ties:2,ordinal:null,sites:null,addresses:null} else . end' \
    "$raw/controls/gold.jsonl" > "$out/one-tie.jsonl"
grade "$out/one-tie.jsonl" | jq -es --arg arm tie --arg part fresh -f "$here/metrics.jq" \
  | jq -e '.signed==127 and .first_later.later_only==1 and .tied==1'
jq -c 'if .row==0 then .links[0].noun.tokens[0]+=1 else . end' "$raw/controls/gold.jsonl" > "$out/wrong-address.jsonl"
if grade "$out/wrong-address.jsonl" > /dev/null 2>"$out/rejection.txt"; then exit 1; fi
grep -q 'native address disagrees with original map' "$out/rejection.txt"
printf '%s\n' 'archived fresh baseline, both supplied controls, one tie and damaged address passed'
