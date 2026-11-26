#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
build=${1:?exact independent audit build}
capture=${2:?returned E and scorer physical finite capture}
out=${3:?new output directory}
[[ $# == 3 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "${out%/*}") == zfs ]]
for path in "$build" "$capture"; do
    (cd "$path" && sha256sum --strict --quiet -c SHA256SUMS)
done
mkdir "$out"
audit=$build/slot3-audit
lex=$capture/lexical-finite
f=$capture/native/fixture
score=$capture/native/scorer/fixture
"$audit" synthetic-eligibility-fit "$f/text.bin" "$lex/labels.tsv" "$lex/blocked.state" "$lex/learn" "$out/e-fit.json"
diff -r "$lex/learn" "$lex/repeat"
for state in blocked learn/e1 learn/e2 learn/e3 learn/e4 learn/wrong; do
    id=${state##*/}
    "$audit" synthetic-view "$lex/$state.state" "$out/$id.view"
    "$audit" physical-words "$f/text.bin" "$out/$id.view" "$lex/$id.jsonl" "$out/$id-words.json"
    "$audit" fixture "$f/text.bin" "$out/$id.view" "$f/fixture.words" "$f/requests.tsv" \
        "$lex/observed/$id.native.jsonl" "$lex/observed/$id.costs.jsonl" "$out/$id-scorer.json"
done
for stream in native costs; do
    cmp "$capture/scorer-finite/a/probe.$stream.jsonl" "$capture/scorer-finite/b/probe.$stream.jsonl"
done
"$audit" fixture "$score/text.bin" "$score/fixture.state" "$score/fixture.words" "$score/requests.tsv" \
    "$capture/scorer-finite/a/probe.native.jsonl" "$capture/scorer-finite/a/probe.costs.jsonl" "$out/score-finite.json"
cmp "$lex/observed/e4.native.jsonl" "$f/original.native.jsonl"
grep -Fq 'factors overlap' "$lex/bad.log"
grep -Fq 'invalid or repeated labeled row' "$lex/invalid.log"
jq -ne --slurpfile before "$lex/blocked.jsonl" --slurpfile after "$lex/e4.jsonl" --slurpfile wrong "$lex/wrong.jsonl" \
    --slurpfile initial "$lex/observed/blocked.costs.jsonl" --slurpfile repaired "$lex/observed/e4.costs.jsonl" \
    --slurpfile invalid "$lex/observed/wrong.costs.jsonl" '
    if all(range(4); . as $i| [$before[$i].words[0][:2],$after[$i].words[0][:2],$wrong[$i].words[0][:2]] == [[32,32],[0,32],[32,0]])
        and all(range($initial|length); . as $i| $initial[$i] as $a| $repaired[$i] as $b| $invalid[$i] as $c|
             if $a.sign == 1 then ($a.sites[$a.requested].local==null and $b.sites[$b.requested].local!=null and $c.sites[$c.requested].local==null)
             else true end) then {strict_tie:"rejected",right_sign:"recovers",wrong_sign:"rejected"}
    else error("E does not physically distinguish tie, repaired, and wrong sign") end' > "$out/opportunity.json"
sha256sum "$build/SHA256SUMS" "$capture/SHA256SUMS" "$(realpath -e "$0")" > "$out/input.sha256"
(cd "$out" && sha256sum *.json *.view input.sha256 > SHA256SUMS)
