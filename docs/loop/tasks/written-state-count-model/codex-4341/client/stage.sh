#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 5 ]] || { echo 'usage: stage.sh ROSTER ROSTER_SHA SELECTED_SOURCE SOURCE_SHA FRESH_OUT' >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/common.sh"
roster=$(realpath "$1")
source_rows=$(realpath "$3")
out=$5
pin "$2" "$roster"
pin "$4" "$source_rows"
mkdir "$out"

jq -ce --slurpfile source "$source_rows" '
    def check($ok;$why): if $ok then . else error($why) end;
    check(.schema=="uor-r4.reciprocal-transfer-validation-roster/1" and .study=="expanded4341";
          "unexpected writer roster schema or study") |
    ($source|INDEX(.left.case_id)) as $indexed |
    ($source|map(.left.case_id,.right.case_id)|to_entries|INDEX(.value)) as $order |
    [ .titles[] as $title | $title.selected[] as $pair |
      select($pair.qualification.stage=="qualified") |
      $indexed[$pair.left_id] as $s |
      check($s!=null and $s.article==$title.name and $s.family==$title.family and
        $s.passage_sha256==$pair.passage_sha256 and $s.right.case_id==$pair.right_id and
        ($pair.qualification.opening.cases|map(.id))==[$pair.left_id,$pair.right_id];
        "qualified writer pair differs from authenticated selection") |
      $pair.qualification.opening.cases[] as $case |
      (if $case.id==$pair.left_id then $s.left else $s.right end) as $side |
      check(($s.context|type)=="string" and ($side.question|type)=="string" and
        $side.prompt==($s.context+" "+$side.question) and ($side.prompt|contains("\u0000")|not);
        "prompt differs from original visible passage and question") |
      $case+{case_id:$case.id,text:$side.prompt,passage:$s.context,question:$side.question,
             passage_sha256:$pair.passage_sha256,article:$title.name,family:$title.family,
             fact:$pair.qualification.opening.fact}
    ] |
    check(length>0 and (map(.id)|length==(unique|length)) and
          ($source|length==113 and (map(.left.case_id,.right.case_id)|length==(unique|length)));
          "empty qualified study or inconsistent registered selected-source roster") |
    sort_by($order[.id].key)
' "$roster" > "$out/cases.json"

while IFS= read -r row; do
    expected=$(jq -er '.opening_sha256' <<< "$row")
    actual=$(jq -jr '.text' <<< "$row" | sha256)
    [[ ${actual%% *} == "$expected" ]] || fail "Qualified prompt digest drift"
done < <(jq -c '.[]' "$out/cases.json")

jq -c '.[] | {case_id,text}' "$out/cases.json" > "$out/blind.jsonl"
jq -c '.[] | select(.target.kind=="source") | {case_id,text}' "$out/cases.json" > "$out/positive-blind.jsonl"
jq -c '.[] | select(.target.kind=="source") | {case_id,start:.target.span.start,end:.target.span.end}' \
    "$out/cases.json" > "$out/gold.jsonl"
jq -c '.[] | select(.target.kind=="source") | {case_id,source_ids:.target.source_ids}' \
    "$out/cases.json" > "$out/truth.jsonl"
jq -n --arg roster_sha256 "$2" --arg selected_source_sha256 "$4" \
    --arg tokenizer_sha256 "$(jq -er .tokenizer_sha256 "$roster")" \
    '{$roster_sha256,$selected_source_sha256,$tokenizer_sha256}' > "$out/bindings.json"
jq '{
    qualified_pairs:length/2, ordinary_per_arm:length, ordinary_all_12_arms:(length*12),
    qualified_positives:(map(select(.target.kind=="source"))|length),
    qualified_negatives:(map(select(.target.kind=="no_read"))|length),
    each_gold_interval_diagnostic:(map(select(.target.kind=="source"))|length)
}' "$out/cases.json" > "$out/plan.json"
(cd "$out" && sha256 bindings.json cases.json blind.jsonl positive-blind.jsonl gold.jsonl truth.jsonl plan.json > STAGE.sha256)
