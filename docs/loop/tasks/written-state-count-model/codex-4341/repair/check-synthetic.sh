#!/usr/bin/env bash
# Every row, review, annotation and protocol token in this test is invented.
set -euo pipefail
umask 077
[[ $# == 2 ]] || { echo 'usage: check-synthetic.sh REPOSITORY FRESH_TEST_DIRECTORY' >&2; exit 2; }
repo=$(cd "$1" && pwd -P)
here=$(cd "$(dirname "$0")" && pwd -P)
mkdir "$2"
cd "$2"
prior=docs/loop/tasks/written-state-count-model/codex-4340/validation
mkdir -p "$prior"
cp "$repo/$prior/"{check-consensus.sh,consensus.jq,qualify-panel.sh,extract-selected.jq} "$prior/"
bash "$prior/check-consensus.sh" legacy
jq -n -f "$here/test-fixtures.jq" > manifest.json
jq -c '
    to_entries[] | select(.key == "two_answer" or .key == "answerability") |
    .key as $family | .value[].candidates[] | select(.disposition == "review") |
    .pair + {family:$family,rank,pair_id:($family + "-" + .pair.passage_sha256)}
' manifest.json > old.jsonl
select_extra() {
    jq -c --slurpfile old old.jsonl -f "$here/expand.jq" "$1"
}
select_extra manifest.json > extra.jsonl
select_extra manifest.json | cmp - extra.jsonl
jq -se '(map(.article)|group_by(.)|map(length))==[10,13,23] and all(.[];.rank>4)' extra.jsonl >/dev/null
cat old.jsonl extra.jsonl > expanded.jsonl
jq -c '.left={case_id:.left} | .right={case_id:.right}' expanded.jsonl > source.jsonl
jq -c '{pair_id,passage_sha256,left_case_id:.left.case_id,right_case_id:.right.case_id,
         left_grade:"supported",right_grade:(if .family=="two_answer" then "supported" else "unanswerable" end),
         left_reason:"synthetic",right_reason:"synthetic",distinct_reason:"synthetic",distinct:true}' source.jsonl > a.jsonl
jq -c '.distinct_reason="independent synthetic"' a.jsonl > b.jsonl
jq -c --slurpfile extra extra.jsonl '
  if .pair_id==$extra[0].pair_id then .right_grade="false_premise" else . end
' b.jsonl > b-disagree.jsonl

reject() {
    if "$@" > rejected.stdout 2> rejected.stderr; then
        echo "Corrupted synthetic input was accepted: $*" >&2
        exit 1
    fi
}
consensus() {
    jq -nc --slurpfile source source.jsonl --slurpfile first a.jsonl --slurpfile second "$1" \
        --slurpfile registered "$2" -f "$prior/consensus.jq"
}
consensus b-disagree.jsonl expanded.jsonl > consensus.jsonl
jq -se 'length==113 and ([.[]|select(.accepted)]|length)==112 and
  any(.[]; .reasons==["second_question_grade_disagreement"])' consensus.jsonl >/dev/null
jq -c 'if .rank==5 then .left="altered" else . end' expanded.jsonl > wrong.jsonl
reject consensus b.jsonl wrong.jsonl
sed '1d' b.jsonl > missing.jsonl
reject consensus missing.jsonl expanded.jsonl
jq -c '.answerability[0].candidates[4].disposition="review"' manifest.json > mutated.json
reject select_extra mutated.json
jq -c '.answerability[1].candidates[4].pair.left=.answerability[1].candidates[0].pair.left' manifest.json > mutated.json
reject select_extra mutated.json

# The deterministic report keeps the published 67 baseline and the full new
# denominator. Native failures here are invented outcome flags.
jq -c '. + {qualified:(.accepted and .negative_group!="false_premise"),native_reasons:[]}' \
    consensus.jsonl > pairs.jsonl
title_counts='group_by([.family,.article]) | map({family:.[0].family,article:.[0].article,
  selected:length,semantic:map(select(.accepted))|length,qualified:map(select(.qualified))|length})'
head -n 67 pairs.jsonl | jq -s "{titles:($title_counts),coverage_pass:false,model_fit:\"NOT_RUN\"}" > old-evidence.json
jq -s "{titles:($title_counts),coverage_pass:true,model_fit:\"NOT_RUN\"}" pairs.jsonl > expanded-evidence.json
jq -n --slurpfile old old-evidence.json --slurpfile expanded expanded-evidence.json --slurpfile pairs pairs.jsonl \
    --slurpfile selection manifest.json -f "$here/coverage.jq" > coverage.json
jq -e '.original_selected==67 and .expanded_selected==113 and .coverage_pass==true and
       ([.titles[].additional_selected]|add)==46 and
       ([.answerability_negative_groups[].concordant_or_unresolved]|add)==90' coverage.json >/dev/null

# Verify the exact prior extraction filter succeeds and detects label drift.
cat > dataset.json <<'JSON'
{"version":"v2.0","data":[{"title":"Fixture","paragraphs":[{"context":"A token. B rival.","qas":[{"id":"positive","is_impossible":false,"question":"Which letter?","answers":[{"text":"A","answer_start":0},{"text":"A","answer_start":0}]},{"id":"negative","is_impossible":true,"question":"Which object?","answers":[],"plausible_answers":[{"text":"rival","answer_start":11}]}]}]}]}
JSON
cat > pick.jsonl <<'JSON'
{"article":"Fixture","passage_sha256":"invented","family":"answerability","rank":5,"pair_id":"answerability-invented","left":"positive","right":"negative"}
JSON
cat > roster.jsonl <<'JSON'
{"split":"validation","article":"Fixture","passage_sha256":"invented","paragraph":0,"window_bytes":[0,17],"shared_passage":"A token. B rival.","shared_passage_ids":[5,6,7,8],"full_paragraph":true,"positive":{"published_id":"positive","published_impossible":false,"question":"Which letter?","prompt":"A token. B rival. Which letter?","prefix":[1,5,6,7,8,9,1],"user_bounds":[1,6],"original_annotation_bytes":[0,1],"original_annotation_char":0,"published_annotation_agreement":2,"label":{"text":"A","char_start":0,"bytes":[0,1],"whole_span_ids":true,"source_ids":[5],"first_id":5,"first_position":1,"response_ids":[16,5],"response_lexical_offset":1}},"negative":{"published_id":"negative","published_impossible":true,"question":"Which object?","prompt":"A token. B rival. Which object?","prefix":[1,5,6,7,8,10,1],"user_bounds":[1,6],"original_annotation_bytes":[11,16],"original_annotation_char":11,"label":{"text":"rival","char_start":11,"bytes":[11,16]}}}
JSON
extract() {
    jq -c --arg family answerability --slurpfile picks pick.jsonl --slurpfile source "$1" \
        -f "$prior/extract-selected.jq" dataset.json
}
extract roster.jsonl | jq -e '.left.answer.source_ids==[5] and .right.answer==null and .right.distractor.bytes==[11,16]' >/dev/null
jq -c '.positive.original_annotation_bytes=[0,2]' roster.jsonl > wrong-annotation.jsonl
reject extract wrong-annotation.jsonl
jq -c '.split="locked"' roster.jsonl > wrong-split.jsonl
reject extract wrong-split.jsonl

# Exercise the real shared qualifier around invented client and binder output.
# The original Wick driver expects GNU realpath and sha256sum.
mkdir bin
for command_name in realpath sha256sum; do
    if command -v "g$command_name" >/dev/null 2>&1; then
        ln -s "$(command -v "g$command_name")" "bin/$command_name"
    fi
done
export PATH="$PWD/bin:$PATH"
mkdir -p "$prior/../../codex-4338/data"
cp "$repo/$prior/../../codex-4338/data/native.jq" "$prior/../../codex-4338/data/"
cat > bin/fixture-client <<'SH'
#!/usr/bin/env bash
set -euo pipefail
jq -c '.case_id as $id | ["geometry","count"][] |
   {case_id:$id,arm:.,user:[1,4],fact:[1,4],question:[4,5],
    no_read_keys:[range(16)|0],candidates:[{start:1,end:2},{start:2,end:3}]}' "${!#}"
SH
cat > bin/fixture-binder <<'SH'
#!/usr/bin/env bash
set -euo pipefail
jq -c '{case_id,user:[1,4],ids:[0,5,6,7],byte_bounds:[[1,2],[2,3],[3,4]]}' "$2"
SH
chmod +x bin/fixture-client bin/fixture-binder
touch fixture-model fixture-tokenizer
mkdir -p synthetic-extraction/{selection,selected-semantic}
jq -c '
    .left += {prompt:"invented",answer:{first_position:1,source_ids:[5],bytes:[0,1],
               response_ids:[16,5],response_lexical_offset:1}} |
    .right += {prompt:"invented",answer:(if .family=="two_answer" then
      {first_position:2,source_ids:[6],bytes:[1,2],response_ids:[16,6],response_lexical_offset:1}
      else null end)}
' source.jsonl > semantic-source.jsonl
qualify_fixture() {
    local source=$1 roster=$2 a=$3 b=$4 out=$5
    shift 5
    cp "$source" synthetic-extraction/selected-semantic/source.jsonl
    cp "$roster" synthetic-extraction/selection/review-identities.jsonl
    (
        cd synthetic-extraction
        (cd selection && sha256sum review-identities.jsonl > SELECTION.sha256)
        (cd selected-semantic && sha256sum source.jsonl > SEMANTIC.sha256)
        sha256sum selection/SELECTION.sha256 selected-semantic/SEMANTIC.sha256 > EXTRACTION.sha256
    )
    digest() { sha256sum < "$1" | cut -d ' ' -f 1; }
    jq -n --arg extraction "$(digest synthetic-extraction/EXTRACTION.sha256)" \
        --arg review_a "$(digest "$a")" --arg review_b "$(digest "$b")" \
        --arg client "$(digest bin/fixture-client)" --arg binder "$(digest bin/fixture-binder)" \
        --arg model "$(digest fixture-model)" --arg tokenizer "$(digest fixture-tokenizer)" \
        '{ $extraction,$review_a,$review_b,$client,$binder,$model,$tokenizer }' > fixture-pins.json
    bash "$prior/qualify-panel.sh" synthetic-extraction "$a" "$b" fixture-pins.json \
        bin/fixture-client bin/fixture-binder fixture-model fixture-tokenizer "$out" "$@"
}
qualify_fixture semantic-source.jsonl expanded.jsonl a.jsonl b-disagree.jsonl qualify-113 expanded.jsonl
jq -e '.coverage_pass==true and .two_answer_total==23 and .answerability_total==89' \
    qualify-113/evidence.json >/dev/null
qualify_fixture semantic-source.jsonl expanded.jsonl a.jsonl b-disagree.jsonl qualify-replay expanded.jsonl
diff -rq qualify-113 qualify-replay
head -n 67 semantic-source.jsonl > prior-semantic.jsonl
head -n 67 a.jsonl > prior-a.jsonl
head -n 67 b.jsonl > prior-b.jsonl
qualify_fixture prior-semantic.jsonl old.jsonl prior-a.jsonl prior-b.jsonl qualify-67
for name in consensus pairs native; do
    head -n "$(wc -l < "qualify-67/$name.jsonl")" "qualify-113/$name.jsonl" |
        cmp - "qualify-67/$name.jsonl"
done
printf 'Synthetic selector, extraction, consensus, native fixture qualification and coverage checks passed\n'
