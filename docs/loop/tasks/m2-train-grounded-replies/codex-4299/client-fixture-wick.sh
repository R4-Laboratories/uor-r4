#!/usr/bin/env bash
# Replay captured ordinary traces under synthetic partial admissions without executing the model.
set -euo pipefail
umask 077
here=$(cd "$(dirname "$0")" && pwd); source "$here/lanes-wick.sh"
r=/mnt/plexpool/uor-r4-corpus/runs
s="$r/20261126-codex-fixed-assignment-median"; parent="$r/20261126-codex-all-outranker-interval"
known="$parent/client/fixture-v5"; old="$known/singleton"; dir=${1:?new fixture}
[[ $here == "$s"/client/methods-v* && $dir =~ ^$s/client/fixture-v[1-9][0-9]*$ && $(hostname -s) == wick ]]
donor="$r/20261125-codex-bound-instruction-state"; prior="$donor/ordinary-recipe-final-v1"; codec="$donor/custody-v1/signed/tokenizer.json"
codec_sha=a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377
model=06a7d1f197870a193699b75dd58ce6269455da3047da87f032de1d45600ccfa2
stock=2e16a8ea27b316cce01dac5744c6b02da38e0ee6c0abc3fcb610e381b1c85302
reference="$parent/client/fixture-v1/reference.json"; mkdir "$dir"
(cd "$known" && sha256sum --strict --quiet -c SHA256SUMS)
jq -c '.arms|={native,baseline,endpoint:.strongest,ordered:.["ordered-all"],correct_link,exchanged_link}' \
    "$known/device.jsonl" > "$dir/device.jsonl"
for arm in endpoint ordered both none; do
    d="$dir/$arm"; mkdir "$d"
    jq --arg arm "$arm" --arg hash "$(sha256sum "$dir/device.jsonl" | cut -d' ' -f1)" '
        . as $g | .arms={endpoint:$g.arms.strongest,ordered:$g.arms["ordered-all"]} |
        .endpoints_sha256=$hash | .arms|=with_entries(.key as $k|.value.arm=$k|.value.endpoints_sha256=$hash|
            .value.pass=($k==$arm or $arm=="both")|.value.gate|=map_values($k==$arm or $arm=="both")) |
        .passing=[.arms[]|select(.pass)|.arm] | .pass=(.passing|length>0)|.gate.at_least_one_trained_arm=.pass' \
        "$old/gate.json" > "$d/gate.json"
    read_gate "$d/gate.json" > "$d/selection.json"
    if [[ $arm == none ]]; then jq -e '.pass==false and .passing==[]' "$d/selection.json" >/dev/null; continue; fi
    plan_capture "$d" "$reference" "$dir/device.jsonl" fixture-study fixture-stock
    while IFS=$'\t' read -r lane _foldname _fold id rest; do
        source_lane=$lane
        if [[ $lane == endpoint || $lane == ordered ]]; then source_lane=ordered-all; fi
        cp "$old/$source_lane/fresh/trace/$id.json" "$d/$lane/fresh/trace/$id.json"
        cp "$old/$source_lane/fresh/trace/$id.exit" "$d/$lane/fresh/trace/$id.exit"
    done < "$d/jobs.tsv"
    collect_capture "$d" "$reference" "$dir/device.jsonl" fixture-study fixture-stock "$model" "$stock"
    mkdir "$d/tokens"; grade_tokens "$d" "$d/tokens"
    jq -e --slurpfile allowed "$d/selection.json" \
        '.source_identity_disagreements==[] and (.arms|map(.arm))==$allowed[0].lanes and
         any(.arms[]; .arm=="correct_link" and .all.exact_terminal==1 and .all.returned_correct_prefix_at_cap==1)' \
         "$d/grade/summary.json" >/dev/null
    jq -n --slurpfile selection "$d/selection.json" --slurpfile tokens "$d/tokens/tokens.jsonl" \
        '{passing:$selection[0].passing,nonstock_lanes:$selection[0].lanes,graded:($tokens|length)}' > "$d/checked.json"
done
jq '.passing=["ordered"]' "$dir/endpoint/gate.json" > "$dir/forged.json"
if read_gate "$dir/forged.json" > "$dir/forged-selection.json" 2>"$dir/forged.log"; then exit 1; fi
jq -ne --slurpfile a "$dir/endpoint/checked.json" --slurpfile b "$dir/ordered/checked.json" --slurpfile c "$dir/both/checked.json" \
    'select($a[0].graded==10 and $b[0].graded==10 and $c[0].graded==12)|
    {pass:true,synthetic_admissions:[$a[0],$b[0],$c[0]],false_gate_refused:true,forged_gate_rejected:true,
     genuine_historical_eos_and_capped_prefix_checked:true,generated:0}' > "$dir/receipt.json"
sha256sum "$here/SHA256SUMS" "$known/SHA256SUMS" "$reference" "$codec" > "$dir/inputs.sha256"
(cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

