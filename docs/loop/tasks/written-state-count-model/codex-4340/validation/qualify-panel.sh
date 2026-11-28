#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 9 ]]; then
    echo 'usage: qualify-panel.sh EXTRACTION REVIEW_A REVIEW_B PIN_FILE CLIENT BINDER BASE TOKENIZER FRESH_OUT' >&2
    exit 2
fi
extraction=$(realpath "$1")
a=$(realpath "$2")
b=$(realpath "$3")
pins=$(realpath "$4")
chat=$(realpath "$5")
binder=$(realpath "$6")
model=$(realpath "$7")
tokenizer=$(realpath "$8")
out=$(realpath -m "$9")
here=$(cd "$(dirname "$0")" && pwd)
prev=$here/../../codex-4338/data
cd "$extraction"
sha256sum --strict --quiet -c EXTRACTION.sha256
(cd selection && sha256sum --strict --quiet -c SELECTION.sha256)
(cd selected-semantic && sha256sum --strict --quiet -c SEMANTIC.sha256)
src=$extraction/selected-semantic/source.jsonl
mkdir "$out"
cd "$out"
# Each role's digest is checked against the registration-time or grading-time
# receipt, before any model can be opened for this validation.
for role in extraction review_a review_b client binder model tokenizer; do
    case "$role" in
        extraction) file=$extraction/EXTRACTION.sha256 ;;
        review_a) file=$a ;;
        review_b) file=$b ;;
        client) file=$chat ;;
        binder) file=$binder ;;
        model) file=$model ;;
        tokenizer) file=$tokenizer ;;
    esac
    printf '%s  %s\n' "$(jq -er --arg k "$role" '.[$k]' "$pins")" "$file" | sha256sum --quiet --strict -c -
done
if cmp -s "$a" "$b"; then
    echo 'the two independent review ledgers are byte-identical' >&2
    exit 1
fi
jq -nc --slurpfile source "$src" --slurpfile first "$a" --slurpfile second "$b" \
    -f "$here/consensus.jq" > consensus.jsonl
jq -c --slurpfile grades consensus.jsonl '
    . as $row | ($grades|INDEX(.pair_id))[$row.pair_id] as $decision | select($decision.accepted) |
    ["left","right"][] as $s | (if $s=="left" then "right" else "left" end) as $mate |
    $row[$s] as $q | $q.answer as $span | $span.first_position as $start |
    {input:{case_id:$q.case_id,text:$q.prompt},gold:{case_id:$q.case_id,start:$start,
      end:(if $start!=null then $start+($span.source_ids|length) else null end),group:$row.pair_id,
      swapped_case_id:$row[$mate].case_id},
     truth:{case_id:$q.case_id,tape:$row.family,article:$row.article,pair_id:$row.pair_id,
       passage_sha256:$row.passage_sha256,source_ids:$span.source_ids,original_bytes:$span.bytes,
       expected_ids:(if $span!=null then $span.response_ids+[1] else null end),
       response_lexical_offset:$span.response_lexical_offset,negative_group:$decision.negative_group}}' \
    "$src" > semantic-cases.jsonl
for part in input gold truth; do jq -c --arg part "$part" '.[$part]' semantic-cases.jsonl > "$part.jsonl"; done
if [[ -s input.jsonl ]]; then
    "$chat" --discrete "$model" --tokenizer "$tokenizer" --interval-features input.jsonl > features.jsonl
    "$binder" "$tokenizer" input.jsonl > openings.jsonl
    jq -en --slurpfile capture features.jsonl '
      all($capture[]; (.no_read_keys|type)=="array" and (.no_read_keys|length)==16)
    ' >/dev/null
    jq -nc --slurpfile train input.jsonl --slurpfile gold gold.jsonl --slurpfile truth truth.jsonl \
        --slurpfile opening openings.jsonl --slurpfile capture features.jsonl -f "$prev/native.jq" > native.jsonl
else
    : > native.jsonl
fi
jq -sc --slurpfile native native.jsonl '
    ($native|INDEX(.case_id)) as $n | .[] | . as $r |
    .native_reasons=([.left_case_id,.right_case_id]|map($n[.].reasons[]?)|unique) |
    .qualified=(.accepted and $n[.left_case_id].retained==true and $n[.right_case_id].retained==true)
' consensus.jsonl > pairs.jsonl
jq -s '
    group_by([.family,.article]) |
    map({family:.[0].family,article:.[0].article,selected:length,semantic:map(select(.accepted))|length,
         qualified:map(select(.qualified))|length}) as $titles |
    ($titles|map(select(.family=="two_answer"))) as $reciprocal |
    ($titles|map(select(.family=="answerability"))) as $absence |
    {titles:$titles,two_answer_total:($reciprocal|map(.qualified)|add),
     answerability_total:($absence|map(.qualified)|add),
     coverage_pass:(($reciprocal|length)==7 and ($absence|length)==11 and
       all($reciprocal[]; .qualified>=1) and ($reciprocal|map(select(.qualified>=2))|length)>=5 and
       all($absence[]; .qualified>=2)),model_fit:"NOT_RUN"}
' pairs.jsonl > evidence.json
sha256sum ./*.jsonl evidence.json > SEALED.sha256
