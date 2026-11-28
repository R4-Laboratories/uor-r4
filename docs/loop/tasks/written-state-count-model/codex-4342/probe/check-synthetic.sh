#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 6 ]]; then
    echo 'usage: check-synthetic.sh BASE TOKENIZER OLD_CLIENT NEW_CLIENT PROBE FRESH_OUTPUT' >&2
    exit 2
fi
base=$(realpath "$1")
tokenizer=$(realpath "$2")
old_client=$(realpath "$3")
new_client=$(realpath "$4")
probe=$(realpath "$5")
out=$(realpath -m "$6")
here=$(cd "$(dirname "$0")" && pwd)
digest() { sha256sum "$1" | cut -d' ' -f1; }
mkdir "$out" "$out/stage"
stage=$out/stage

# Authored geometry and intake fixtures only. No source or model evaluation.
text_a='A canal moved amber toward Rome and silver toward Athens.'
text_b='A ship carried cargo toward Dover for winter.'
sha_a=$(printf '%s' "$text_a" | sha256sum | cut -d' ' -f1)
sha_b=$(printf '%s' "$text_b" | sha256sum | cut -d' ' -f1)
text_c=$(printf 'A rope. %.0s' {1..300})'A rope held cargo.'
sha_c=$(printf '%s' "$text_c" | sha256sum | cut -d' ' -f1)
jq -n --arg text_a "$text_a" --arg text_b "$text_b" --arg text_c "$text_c" \
    --arg sha_a "$sha_a" --arg sha_b "$sha_b" --arg sha_c "$sha_c" '
    def case($id; $passage; $sha; $question; $surface; $paragraph):
        {case_id:$id, text:($passage+" "+$question), split:"screen", article:"synthetic",
         paragraph:$paragraph, passage_sha256:$sha, passage_bytes:($passage|length),
         question_source_id:("source-"+$id), family:("family-"+$id), question:$question,
         span_text:$surface,
         original_bytes:(if $surface==null then null else ($passage|index($surface)) as $start
             | [$start,($start+($surface|length))] end)}
        | .original_char=.original_bytes;
    [
        case("a1";$text_a;$sha_a;"What did the canal move toward Rome in the passage?";"amber";0),
        case("a2";$text_a;$sha_a;"Near Rome what did a canal move then?";"amber";0),
        case("b1";$text_a;$sha_a;"What did the canal move toward Athens in the passage?";"silver";0),
        case("b2";$text_a;$sha_a;"Near Athens what did a canal move then?";"silver";0),
        case("n1";$text_b;$sha_b;"What did the ship carry toward Dover in winter?";"cargo";1),
        case("n2";$text_b;$sha_b;"What did the ship avoid near Dover in winter?";null;1),
        case("l1";$text_c;$sha_c;"What did the rope hold?";"cargo";2),
        case("l2";$text_c;$sha_c;"Which material was held by a rope?";"cargo";2)
    ]' > "$out/fixture.json"
jq -c '.[]|{case_id,text}' "$out/fixture.json" > "$stage/blind.jsonl"
jq -c '.[]|del(.text)' "$out/fixture.json" > "$stage/truth.jsonl"
jq -nc --arg sha_a "$sha_a" --arg sha_b "$sha_b" --arg sha_c "$sha_c" '
    {group_id:"q",split:"screen",stratum:"quad",article:"synthetic",paragraph:0,
     passage_sha256:$sha_a,families:[["a1","a2"],["b1","b2"]],contrast:"target-switch"},
    {group_id:"n",split:"screen",stratum:"switch",article:"synthetic",paragraph:1,
     passage_sha256:$sha_b,families:[["n1"],["n2"]],contrast:"answerability"},
    {group_id:"l",split:"screen",stratum:"paraphrase",article:"synthetic",paragraph:2,
     passage_sha256:$sha_c,families:[["l1","l2"]],contrast:"paraphrase"}
    ' > "$stage/groups.jsonl"

bash "$here/capture.sh" "$stage" "$(digest "$stage/blind.jsonl")" \
    "$(digest "$stage/truth.jsonl")" "$(digest "$stage/groups.jsonl")" "$base" \
    "$tokenizer" "$(digest "$tokenizer")" "$old_client" "$(digest "$old_client")" \
    "$new_client" "$(digest "$new_client")" "$probe" "$out/valid"
jq -e '
    .source_qualified_cases == 8 and .unadmitted_client_cases == 2 and
    .verified_client_cases == 6 and .verified_legacy_openings == 12 and .verified_relation_openings == 30 and
    .registered_group_pass_count == 0 and .qualified_group_rows[0].truth_cases == 4 and
    .qualified_group_rows[0].original_target_unreachable == 0 and
    .qualified_group_rows[0].original_mate_unreachable == 0 and
    .qualified_group_rows[0].arms.directed_geometry.bank_10.reverse_comparisons == 4 and
    .qualified_group_rows[1].labeled_no_read == 1 and
    .qualified_group_rows[1].arms.directed_geometry.bank_10.reverse_comparisons == 1 and
    (.qualified_group_rows[2].arms | length) == 0 and
    ([ .qualified_group_rows[2].opening_admissions[].refusal ] |
        . == ["prefix-exceeds-context", "prefix-exceeds-context"])
    ' "$out/valid/probe.json" > /dev/null

reject_tape() {
    local label=$1 mutation=$2 error=$3
    local altered=$out/$label.jsonl
    jq -sc "$mutation | .[]" "$out/valid/new.jsonl" > "$altered"
    if "$probe" "$stage" "$(digest "$stage/blind.jsonl")" "$(digest "$stage/truth.jsonl")" \
        "$(digest "$stage/groups.jsonl")" "$base" "$tokenizer" "$(digest "$tokenizer")" \
        "$out/valid/old.jsonl" "$(digest "$out/valid/old.jsonl")" "$altered" "$(digest "$altered")" \
        > "$out/$label.stdout" 2> "$out/$label.stderr"; then
        echo "Expected rejection of $label" >&2
        exit 1
    fi
    rg -q --fixed-strings "$error" "$out/$label.stderr"
}

reject_tape old-key '.[0].candidates[0].keys[10] |= .+1' 'pre-change legacy opening or keys differ'
reject_tape foreign-bank '.[2].candidates[0].keys[7] |= .+1' 'a bank other than 10 changed'
reject_tape candidate-order '.[2].candidates[0:2] |= reverse' 'candidate endpoints or order changed'
reject_tape endpoint '.[2].relation_endpoints[0].source[0]=65535' 'original lexical neighbor, anchor or clause differs'
reject_tape root '.[2].relation_tokens[0].root |= (. + 1) % 16' 'original token IDs, bytes or roots malformed'
reject_tape roles '.[2].fact[0] |= .+1' 'original user, fact, or question role bounds differ'
reject_tape no-read '.[2].no_read_keys[10] |= (2560 + 224 + ((. + 1) % 32))' 'NoRead keys changed'
reject_tape directed-order '
    .[2].candidates[0].keys[10] |= (2560 + ((. - 2560 + 1) % 65))
    ' 'bank 10 address differs from independent endpoint replay'
reject_tape equal-arity '.[3].candidates[0].keys[10] |= (2560 + ((. - 2560 + 1) % 65))' \
    'bank 10 address differs from independent endpoint replay'
reject_tape extra-score '.[2].scores={}' 'unknown field'
reject_tape missing-case 'del(.[0])' 'missing or trailing capture cases'

reject_truth() {
    local label=$1 mutation=$2 error=$3
    local altered=$out/$label
    mkdir "$altered"
    cp "$stage/blind.jsonl" "$stage/groups.jsonl" "$altered/"
    jq -sc "$mutation | .[]" "$stage/truth.jsonl" > "$altered/truth.jsonl"
    if "$probe" "$altered" "$(digest "$altered/blind.jsonl")" "$(digest "$altered/truth.jsonl")" \
        "$(digest "$altered/groups.jsonl")" "$base" "$tokenizer" "$(digest "$tokenizer")" \
        "$out/valid/old.jsonl" "$(digest "$out/valid/old.jsonl")" \
        "$out/valid/new.jsonl" "$(digest "$out/valid/new.jsonl")" \
        > "$out/$label.stdout" 2> "$out/$label.stderr"; then
        echo "Expected rejection of $label" >&2
        exit 1
    fi
    rg -q --fixed-strings "$error" "$out/$label.stderr"
}

reject_truth wrong-passage '.[0].passage_bytes += 1' 'original passage digest differs'
reject_truth wrong-question '.[0].question += "x"' 'original question bytes differ'
reject_truth wrong-chars '.[0].original_char[0] += 1' 'original answer code-point positions differ'
