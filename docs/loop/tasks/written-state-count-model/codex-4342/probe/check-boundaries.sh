#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 6 ]]; then
    echo 'usage: check-boundaries.sh BASE TOKENIZER OLD_CLIENT NEW_CLIENT PROBE FRESH_OUTPUT' >&2
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

# Authored words reproduce the exact 32-token fallback boundary from review.
passage='Amber Keeper sent Falcon toward Green Heron supercalifragilisticexpialidocious'
passage_sha=$(printf '%s' "$passage" | sha256sum | cut -d' ' -f1)
jq -nc --arg passage "$passage" --arg sha "$passage_sha" '
    [
       ["clipped-cyan", ("Keeper" + (" cyan" * 11) + "?")],
       ["clipped-utf8", ("𐐷 Keeper" + (" cyan" * 8) + "?")]
    ][] |
    . as [$case, $question] |
    ($passage | index("Falcon")) as $start |
    {case_id:$case, text:($passage+" "+$question), split:"screen", article:"synthetic-boundaries",
     paragraph:0, passage_sha256:$sha, passage_bytes:($passage|length), question_source_id:("fixture-"+$case),
     family:$case, question:$question, span_text:"Falcon",
     original_bytes:[$start, ($start+6)], original_char:[$start, ($start+6)]}
    ' > "$out/fixture.jsonl"
jq -c '{case_id,text}' "$out/fixture.jsonl" > "$stage/blind.jsonl"
jq -c 'del(.text)' "$out/fixture.jsonl" > "$stage/truth.jsonl"
jq -nc --arg sha "$passage_sha" '
    {group_id:"clip",split:"screen",stratum:"paraphrase",article:"synthetic-boundaries",paragraph:0,
     passage_sha256:$sha,families:[["clipped-cyan","clipped-utf8"]],contrast:"paraphrase"}
    ' > "$stage/groups.jsonl"

bash "$here/capture.sh" "$stage" "$(digest "$stage/blind.jsonl")" \
    "$(digest "$stage/truth.jsonl")" "$(digest "$stage/groups.jsonl")" "$base" \
    "$tokenizer" "$(digest "$tokenizer")" "$old_client" "$(digest "$old_client")" \
    "$new_client" "$(digest "$new_client")" "$probe" "$out/verified"
jq -e '.verified_client_cases == 2 and .verified_legacy_openings == 4 and .verified_relation_openings == 10' \
    "$out/verified/probe.json" > /dev/null

jq -es '
    def test_cyan:
      .question == [44,76] and
      .relation_tokens[43].bytes == [121] and .relation_tokens[44].bytes == [97,110] and
      (.relation_endpoints as $endpoints | [
         .candidates | to_entries[] | select(.value.start == 7 and .value.end == 11) |
         .value.keys[10] == 2560 and $endpoints[.key].query == [null,48] and
         $endpoints[.key].query_anchor == [45,48] and $endpoints[.key].source == [6,11]
      ] == [true]);
    def test_utf8:
      .question == [40,72] and
      ([.relation_tokens[39:43][].bytes[]] == [240,144,144,183]) and
      all(.relation_endpoints[]; .query_anchor == null and .query == [null,null]) and
      all(.candidates[]; .keys[10] == 2560);
    ([.[] | select(.case_id == "clipped-cyan" and .relation_tokens != null) | test_cyan] == [true,true,true]) and
    ([.[] | select(.case_id == "clipped-utf8" and .relation_tokens != null) | test_utf8] == [true,true,true])
    ' "$out/verified/new.jsonl" > /dev/null

# Restore the exact faulty witness from the review, with a fresh tape digest.
# The independent verifier must reject it during endpoint replay.
jq -c '
    if .case_id == "clipped-cyan" and .arm == "directed_geometry"
    then (.candidates | to_entries[] | select(.value.start == 7 and .value.end == 11).key) as $index |
         .relation_endpoints[$index].query[0] = 44 | .candidates[$index].keys[10] = 2606
    else . end' "$out/verified/new.jsonl" > "$out/archived-counterexample.jsonl"
if "$probe" "$stage" "$(digest "$stage/blind.jsonl")" "$(digest "$stage/truth.jsonl")" \
    "$(digest "$stage/groups.jsonl")" "$base" "$tokenizer" "$(digest "$tokenizer")" \
    "$out/verified/old.jsonl" "$(digest "$out/verified/old.jsonl")" \
    "$out/archived-counterexample.jsonl" "$(digest "$out/archived-counterexample.jsonl")" \
    > "$out/counterexample.stdout" 2> "$out/counterexample.stderr"; then
    echo 'Expected rejection of clipped lexical neighbor' >&2
    exit 1
fi
rg -qx 'clipped-cyan: original lexical neighbor, anchor or clause differs' "$out/counterexample.stderr"
(
    cd "$out"
    sha256sum ./fixture.jsonl ./verified/PROBE.sha256 ./archived-counterexample.jsonl \
        ./counterexample.stdout ./counterexample.stderr > BOUNDARIES.sha256
)
