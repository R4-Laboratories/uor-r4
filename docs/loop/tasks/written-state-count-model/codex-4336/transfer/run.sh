#!/usr/bin/env bash
set -euo pipefail
umask 077
here=$(cd "$(dirname "$0")" && pwd)
runs=$(realpath "${1:?Wick runs}")
chat=$(realpath "${2:?pinned client}")
out=$(realpath -m "${3:?fresh output}")
previous="$runs/20261127-codex-edge-source/archive-v1"
prompts="$previous/parent-cold/4333/payload/panel-v1/blind/prompts.jsonl"
scores="$previous/cold-readback/client/scores.jsonl"
balanced="$runs/20261127-codex-period-margin/cold-v1/balanced/artifact.d11"
tokenizer="$runs/20261127-codex-whole-turn-fit/archive-readback-v1/tools/tokenizer.json"
mkdir "$out"
mkdir "$out/rows"
(cd "$previous/cold-readback/client" && sha256sum --strict --quiet -c SHA256SUMS)
(cd "$(dirname "$prompts")" && sha256sum --strict --quiet -c SEALED.sha256)
sha256sum --strict --quiet -c <<SUMS
5c98e60db9011890d3dbce42f197c8703cfb75865e85ee61db2c364a8624c2aa  $balanced
a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377  $tokenizer
SUMS
jq -ces --slurpfile prompts "$prompts" --slurpfile tapes "$previous/cold-readback/client/client/balanced.jsonl" '
  ($prompts | INDEX(.case_id)) as $texts | ($tapes | INDEX(.case_id)) as $tapes_by_id |
  [ .[] | select(.arm == "balanced") | . as $s | ($tapes_by_id[.case_id].client.prompt_ids) as $prefix | if (.prefix_ids != ($prefix|length)) then error("prefix length drift") else . end |
    (.grade.expected_ids_through_eos // []) as $ids |
    (if .grade.target == null then null else
      {start:.grade.target.pos, end:(.grade.target.pos + ($ids|length) - 1), ids:$ids[:-1]} end) as $span |
    if $span != null and (($ids[-1] != 1) or ($prefix[$span.start:$span.end] != $span.ids)) then error("source or EOS drift") else . end |
    {case_id, text:$texts[.case_id].text, prompt_ids:$prefix, expected:$ids, span:$span,
     next_id:(if $span == null then null else $prefix[$span.end] end)} ]
  | if length == 18 and ([.[] | select(.span != null)] | length) == 13 then .[] else error("unexpected panel") end
' "$scores" > "$out/inputs.jsonl"
jq -j '.case_id,"\u0000",.text,"\u0000"' "$out/inputs.jsonl" |
  xargs -0 -r -n2 -P3 bash -ceu '
    set -o pipefail
    out=$1; chat=$2; model=$3; tok=$4; id=$5; text=$6
    for action in native stop; do
      request="$out/rows/$id-$action-claim.json"
      jq -c --arg id "$id" --arg action "$action" \
        "select(.case_id==\$id) | {authority:{prompt_ids,first:.span},claim:{first:.span,first_action:\$action}}" \
        "$out/inputs.jsonl" > "$request"
      "$chat" --discrete "$model" --tokenizer "$tok" --say "$text" --protocol 2 --seed 0 \
         --temperature 0 --max-tokens 16 --authenticated-copy --native-period-release \
         --native-occurrence-trace --verified-span-release "$request" |
         jq -ceS --arg id "$id" --arg arm "$action" "{id:\$id,arm:\$arm,client:del(.artifact)}" > "$out/rows/$id-$action.json"
    done
  ' study "$out" "$chat" "$balanced" "$tokenizer"
cat "$out"/rows/*-native.json "$out"/rows/*-stop.json > "$out/records.jsonl"
jq -ces --slurpfile inputs "$out/inputs.jsonl" -f "$here/grade.jq" "$out/records.jsonl" > "$out/summary.json"
(cd "$out" && sha256sum inputs.jsonl records.jsonl summary.json > SHA256SUMS)
