#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 5 ]] || { echo 'usage: visibility.sh STAGE STAGE_SHA VERIFIED TRACE_SHA FRESH_OUT' >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/common.sh"
stage=$(realpath "$1")
verified=$(realpath "$3")
out=$5
sealed "$stage" STAGE.sha256 "$2" cases.json
sealed "$verified" TRACE.sha256 "$4" trace.json ordinary.json
jq -e --arg digest "$2" '.stage_sha256==$digest' "$verified/trace.json" >/dev/null
mkdir "$out"

jq -ce --slurpfile cases "$stage/cases.json" '
    ($cases[0]|INDEX(.id)) as $c |
    map(select(.negative_group!=null) | . as $output | $c[.case_id] as $source |
      {case_id,grade:.deterministic_visible,
       display:{passage:$source.passage,question:$source.question,reply:$output.text}}) |
    group_by(.display)|.[] |
    if (map(.grade)|unique|length)!=1 then error("visible recognizer disagrees on identical bytes")
    else {display:.[0].display,case_ids:map(.case_id),grade:.[0].grade} end
' "$verified/ordinary.json" > "$out/grouped.jsonl"
while IFS= read -r row; do
    key=$(jq -cS '.display' <<< "$row" | sha256)
    jq -cS --arg key "${key%% *}" '.+{key:$key}' <<< "$row"
done < "$out/grouped.jsonl" > "$out/visibility.jsonl"
jq -cS 'select(.grade=="requires_semantic_review") | .display+{key}' \
    "$out/visibility.jsonl" > "$out/blind-review.jsonl"
jq -n --arg trace_sha256 "$4" '{$trace_sha256}' > "$out/bindings.json"
(cd "$out" && sha256 bindings.json visibility.jsonl blind-review.jsonl > VISIBLE.sha256)
