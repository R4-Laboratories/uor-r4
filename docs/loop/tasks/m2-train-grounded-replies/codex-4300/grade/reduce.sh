#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 3 ]]
m=$(realpath -e "${1:?physical capture}")
panel=$(realpath -e "${2:?enrolled panel}")
out=${3:?new reduction directory}
grade=$(cd "$(dirname "$0")" && pwd -P)
mkdir "$out"
for cohort in historical train-original train-derived; do
    p="$panel/$cohort"; d="$m/$cohort"
    jq -nc --arg cohort "$cohort" --slurpfile meta "$p/requests.jsonl" --slurpfile rows "$p/rows.jsonl" \
        --slurpfile before "$d/before/requests.jsonl" --slurpfile after "$d/after/requests.jsonl" \
        -f "$grade/row.jq" > "$out/$cohort.rows.jsonl"
    test "$(wc -l < "$out/$cohort.rows.jsonl")" = "$(wc -l < "$p/requests.tsv")"
done
jq -ne --slurpfile historical "$out/historical.rows.jsonl" --slurpfile original "$out/train-original.rows.jsonl" \
    --slurpfile derived "$out/train-derived.rows.jsonl" \
    --argjson original_prompts "$(wc -l < "$panel/train-original/rows.jsonl")" \
    --argjson derived_prompts "$(wc -l < "$panel/train-derived/rows.jsonl")" \
    -f "$grade/summary.jq" > "$out/summary.json"
jq -sc 'map(select(.historical_lost))|sort_by(.pair_id,.selector)' "$out/historical.rows.jsonl" > "$out/four-switches.json"
jq -e 'length==8' "$out/four-switches.json" >/dev/null
jq -r '(["pair","selector","signed_role","state","governor","requested","rival","local_requested_lex_transport",
    "local_rival_lex_transport","conditioned_requested_lex_transport_bank","conditioned_rival_lex_transport_bank",
    "native_source","native_source_BPE","native_cost","native_governor","native_opposite","native_heading","classification"]|@tsv),
    (.[] as $r|["before","after"][] as $name|$r[$name] as $s|
     [$r.pair_id,$r.selector,$r.role,$name,$r.governor.text,$r.requested.text,$r.rival.text,($s.requested_local.cost|tojson),
      ($s.rival_local.cost|tojson),($s.requested_tuple.cost|tojson),($s.rival_tuple.cost|tojson),
      $s.native.signed_site.text,($s.native.signed_site.tokens|tojson),($s.native.cost|tojson),
      $s.native.signed_governors[(if $r.sign==1 then 0 else 1 end)].text,$s.native.opposite_site.text,
      $s.native.heading.text,$s.stage]|@tsv)' "$out/four-switches.json" > "$out/four-switches.tsv"
