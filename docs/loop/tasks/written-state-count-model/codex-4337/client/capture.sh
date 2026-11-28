#!/usr/bin/env bash
set -euo pipefail
umask 077

client=$(realpath "${1:?client binary}")
model=$(realpath "${2:?D11 artifact}")
tokenizer=$(realpath "${3:?tokenizer}")
input=$(realpath "${4:?blind case_id/text JSONL}")
out=$(realpath -m "${5:?fresh capture directory}")
shift 5
mkdir "$out"
# A blind capture may only consume the client text and an opaque case identity.
# Process substitutions can mask a producer's failure, so validate the full input first.
jq -es 'length > 0 and (map(.case_id) | length == (unique | length))
    and all(.[]; keys == ["case_id","text"] and
        (.case_id | type == "string" and length > 0) and
        (.text | type == "string" and length > 0 and (contains("\u0000")|not)))' \
    "$input" > /dev/null
while IFS= read -r row || [[ -n $row ]]; do
    case_id=$(jq -er '.case_id' <<< "$row")
    # The suffix prevents command substitution from stripping meaningful trailing newlines.
    text=$(jq -jr '.text, "."' <<< "$row")
    "$client" --discrete "$model" --tokenizer "$tokenizer" --say "${text%.}" \
        --protocol 2 --seed 0 --temperature 0 --max-tokens 16 "$@" |
        jq -ceS --arg id "$case_id" \
            'select((.prompt_ids|type)=="array" and (.generated_ids|type)=="array" and
                    (.stop|type)=="string") | {case_id:$id,client:del(.artifact)}' \
            >> "$out/records.jsonl"
done < "$input"
jq -ens --slurpfile cases "$input" --slurpfile records "$out/records.jsonl" \
    '($cases|map(.case_id)) == ($records|map(.case_id))' > /dev/null
sha256sum "$client" "$model" "$tokenizer" "$input" > "$out/INPUTS.sha256"
(cd "$out" && sha256sum records.jsonl > REPLAY.sha256)
