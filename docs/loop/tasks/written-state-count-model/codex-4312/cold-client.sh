#!/usr/bin/env bash
# Independent ordinary product runs. One fresh process for each retained dialogue or prose anchor.
set -euo pipefail
umask 077
key=$1
run=$2
prompts=$3
tokenizer=$4
evaluator=$5
client=$6
out=$7
mkdir "$out"
for arm in reasoning-first answer-first; do
    code=$(jq -er --arg arm "$arm" '.[$arm]' "$key")
    artifact="$run/train/$arm/artifact.d11"
    source="$run/generate-$code"
    mkdir "$out/$arm"
    "$evaluator" replay "$artifact" "$source/witnesses.jsonl" "$prompts" "$source/replies.jsonl" > "$out/$arm/readback.json"
    jq -rcs 'group_by(.domain)[]|sort_by(.id)|[first,last][]|[.id,(.prefix|join(","))]|@tsv' "$prompts" |
        while IFS=$'\t' read -r id prefix; do
            expected=$(jq -rc --arg id "$id" 'select(.id==$id)|.tokens' "$source/replies.jsonl")
            length=$(jq -er 'length' <<< "$expected")
            "$client" --discrete "$artifact" --prompt-ids "$prefix" --max-tokens "$length" > "$out/$arm/$id.json"
            jq -e --argjson ids "$expected" '.generated_ids == $ids' "$out/$arm/$id.json" >/dev/null
        done
    # The text entry point binds the token vocabulary and stops at the model's natural EOS.
    "$client" --discrete "$artifact" --tokenizer "$tokenizer" --protocol 2 --say 'Explain how a seed grows into a tree.' --max-tokens 32 > "$out/$arm/text-mode.json"
    jq -e '(.generated_ids|length)>0 and ((.generated_ids|index(1)) as $e |
        if $e==null then (.generated_ids|length)==32 else $e==((.generated_ids|length)-1) end)' "$out/$arm/text-mode.json" >/dev/null
done
