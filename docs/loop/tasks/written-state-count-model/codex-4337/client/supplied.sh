#!/usr/bin/env bash
set -euo pipefail
umask 077
client=$(realpath "${1:?pinned client}")
model=$(realpath "${2:?balanced artifact}")
tokenizer=$(realpath "${3:?bound tokenizer}")
blind=$(realpath "${4:?blind JSONL}")
gold=$(realpath "${5:?original address JSONL}")
truth=$(realpath "${6:?independent token truth JSONL}")
baseline=$(realpath "${7:?ordinary same-panel capture}")
out=$(realpath -m "${8:?fresh root}")
mkdir "$out"
mkdir "$out/native" "$out/stop"
# This control reproduces the existing immediate-entry writer. A delayed leading
# space has an independent token in the normal client, so exclude those rows here.
jq -cens --slurpfile blind "$blind" --slurpfile gold "$gold" --slurpfile truth "$truth" \
    --slurpfile records "$baseline" '
    if [($blind|map(.case_id)),($gold|map(.case_id)),($truth|map(.case_id)),($records|map(.case_id))]
        | map(sort) | unique | length != 1 then error("case identities drifted") else . end |
    ($records|INDEX(.case_id)) as $r | ($gold|INDEX(.case_id)) as $g | ($truth|INDEX(.case_id)) as $t |
    $blind[] | . as $prompt | $g[.case_id] as $target | $t[.case_id] as $answer |
    select($target.start!=null and $answer.expected_ids==($answer.source_ids+[1])) |
    ($r[.case_id].client.prompt_ids) as $ids |
    if (($target.end-$target.start) != ($answer.source_ids|length)) or
        ($ids[$target.start:$target.end] != $answer.source_ids)
    then error("gold original interval fails authentication") else . end |
    {case_id,text,ids:$ids,span:{start:$target.start,end:$target.end,ids:$answer.source_ids}}
    ' > "$out/supported-immediate.jsonl"
while IFS= read -r row; do
    id=$(jq -er '.case_id' <<< "$row")
    text=$(jq -jr '.text,"."' <<< "$row")
    for action in native stop; do
        jq -c --arg action "$action" \
            '{authority:{prompt_ids:.ids,first:.span},claim:{first:.span,first_action:$action}}' \
            <<< "$row" > "$out/request.json"
        "$client" --discrete "$model" --tokenizer "$tokenizer" --say "${text%.}" \
            --protocol 2 --seed 0 --temperature 0 --max-tokens 16 --authenticated-copy \
            --native-period-release --native-occurrence-trace --verified-span-release "$out/request.json" |
            jq -ceS --arg id "$id" '{case_id:$id,client:del(.artifact)}' >> "$out/$action/records.jsonl"
    done
done < "$out/supported-immediate.jsonl"
for action in native stop; do
    jq -ens --slurpfile inputs "$out/supported-immediate.jsonl" --slurpfile records "$out/$action/records.jsonl" \
        '($inputs|map(.case_id))==($records|map(.case_id)) and
         all($records[]; (.client.verified_span_release.steps|map(.committed_id))==.client.generated_ids)' > /dev/null
done
(cd "$out" && sha256sum supported-immediate.jsonl native/records.jsonl stop/records.jsonl > REPLAY.sha256)
