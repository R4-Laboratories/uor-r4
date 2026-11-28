#!/usr/bin/env bash
set -euo pipefail

[[ $# == 4 ]] || { echo 'usage: verify.sh INVENTORY INVENTORY_SHA SUPPLY SUPPLY_SHA' >&2; exit 2; }
inventory=$(realpath "$1")
supply=$(realpath "$3")
printf '%s  %s\n' "$2" "$inventory/INVENTORY.sha256" "$4" "$supply/SUPPLY.sha256" |
    sha256sum --quiet --strict -c -
(cd "$inventory" && sha256sum --quiet --strict -c INVENTORY.sha256 && sha256sum --quiet --strict -c SOURCE.sha256)
(cd "$supply" && sha256sum --quiet --strict -c SUPPLY.sha256 && sha256sum --quiet --strict -c INPUTS.sha256)

jq -ne --slurpfile train "$inventory/train-source.jsonl" --slurpfile truth "$supply/truth.jsonl" \
  --slurpfile blind "$supply/blind.jsonl" --slurpfile groups "$supply/groups.jsonl" \
  --slurpfile counts "$supply/counts.json" '
  ($train | INDEX([.article,.paragraph] | tojson)) as $paragraphs |
  ($blind | INDEX(.case_id)) as $prompts |
  ($truth | INDEX(.case_id)) as $labels |
  ($labels | keys) as $ids |
  if ($truth|length) != $counts[0].accepted_questions or ($groups|length) != $counts[0].accepted_groups or
     ($truth|length) != ($ids|length) or ($blind|length) != ($ids|length) or ($prompts|keys) != $ids or
     ([$groups[].families[][]]|sort) != $ids
  then error("source, prompt, or group membership differs") else . end |
  if all($truth[]; . as $row |
     $paragraphs[([.article,.paragraph]|tojson)] as $source |
     $row.original_char as [$start,$end] |
     [$source.qas[]|select(.id==$row.question_source_id)] as $qa |
     $source.split=="screen" and $row.split=="screen" and ($qa|length)==1 and ($qa[0].is_impossible|not) and
     ($qa[0].answers|length)>0 and all($qa[0].answers[]; .answer_start==$start and .text==$row.span_text) and
     $row.case_id==$row.question_source_id and $qa[0].question==$row.question and
     $source.context[$start:$end]==$row.span_text and
     [($source.context[0:$start]|utf8bytelength),($source.context[0:$end]|utf8bytelength)]==$row.original_bytes and
     ($source.context|utf8bytelength)==$row.passage_bytes and
     ($prompts[$row.case_id]|keys)==["case_id","text"] and $prompts[$row.case_id].text==($source.context+" "+$row.question))
  then . else error("official original address or blind prompt differs") end |
  if all($groups[]; . as $group |
     ($group.families|map(length))==({quad:[2,2],triple:[2,1],switch:[1,1],paraphrase:[2]}[$group.stratum]) and
     all($group.families[]; (map($labels[.].original_bytes)|unique|length)==1) and
     all($group.families[][]; $labels[.] as $row |
       $row.article==$group.article and $row.paragraph==$group.paragraph and $row.passage_sha256==$group.passage_sha256))
  then {verified_cases:($truth|length),verified_groups:($groups|length)}
  else error("original address family or stratum differs") end'
