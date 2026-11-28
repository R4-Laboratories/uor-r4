#!/usr/bin/env bash
set -euo pipefail
umask 077
client=$(realpath "${1:?final client}")
model=$(realpath "${2:?balanced writer}")
tok=$(realpath "${3:?bound tokenizer}")
blind=$(realpath "${4:?blind inputs}")
gold=$(realpath "${5:?gold intervals and reciprocal pair IDs}")
truth=$(realpath "${6:?token truth and passage identity}")
native=$(realpath "${7:?frozen ordinary learned-reader records}")
mode=${8:?gold-native, gold-stop or swapped-read}
out=$(realpath -m "${9:?fresh output}")
case "$mode" in gold-native|gold-stop|swapped-read) ;; *) exit 2;; esac
mkdir "$out"
jq -cens --arg mode "$mode" --slurpfile blind "$blind" --slurpfile gold "$gold" \
    --slurpfile truth "$truth" --slurpfile native "$native" '
    [$blind,$gold,$truth,$native] | map(map(.case_id)|sort) as $ids |
    if ($ids|unique|length)!=1 or ($ids[0]|unique|length)!=($ids[0]|length)
    then error("control case roster differs") else . end |
    ($truth|INDEX(.case_id)) as $ts | ($gold|INDEX(.case_id)) as $gs | ($native|INDEX(.case_id)) as $rs |
    $blind[] | . as $row | $gs[.case_id] as $g | $ts[.case_id] as $t |
    $rs[.case_id].client.prompt_ids as $prefix | $g.swapped_case_id as $peer |
    if $mode=="swapped-read" then select($peer!=null) |
      if $gs[$peer].swapped_case_id!=$row.case_id or $t.passage_sha256==null or
         $ts[$peer].passage_sha256!=$t.passage_sha256 or $rs[$peer].client.native_interval.selection==null
      then error("swap fails matched factual passage or reader identity") else . end
    else select($g.start!=null) end |
    (if $mode=="swapped-read" then $rs[$peer].client.native_interval.selection.span
     else {start:$g.start,end:$g.end} end) as $span |
    (if $span==null then null else $prefix[$span.start:$span.end] end) as $ids |
    if $span!=null and (if $mode=="swapped-read" then
          $rs[$peer].client.prompt_ids[$span.start:$span.end]!=$ids
       else $t.source_ids!=$ids end)
    then error("selected interval changes original IDs") else . end |
    {case_id,text,peer:$peer,control:(if $span==null then null else $span+{ids:$ids,prompt_ids:$prefix} end)}
    ' > "$out/actions.jsonl"
while IFS= read -r row; do
    id=$(jq -er '.case_id' <<< "$row")
    text=$(jq -jr '.text,"."' <<< "$row")
    args=(--native-occurrence-trace)
    if jq -e '.control!=null' <<< "$row" > /dev/null; then
        jq -c '.control' <<< "$row" > "$out/selected.json"
        args+=(--interval-supplied-source "$out/selected.json")
        if [[ $mode == gold-stop ]]; then args+=(--interval-supplied-stop); fi
    else
        args+=(--no-native-occurrence)
    fi
    "$client" --discrete "$model" --tokenizer "$tok" --say "${text%.}" --protocol 2 \
       --seed 0 --temperature 0 --max-tokens 16 "${args[@]}" |
       jq -ceS --arg id "$id" '{case_id:$id,client:del(.artifact)}' >> "$out/records.jsonl"
done < "$out/actions.jsonl"
jq -ens --slurpfile actions "$out/actions.jsonl" --slurpfile records "$out/records.jsonl" \
    '($actions|map(.case_id))==($records|map(.case_id)) and ($records|length)>0' > /dev/null
(cd "$out" && sha256sum actions.jsonl records.jsonl > REPLAY.sha256)
