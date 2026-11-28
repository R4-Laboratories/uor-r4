#!/usr/bin/env bash
set -euo pipefail
umask 077
grading=$(realpath "${1:?grading source directory}")
gold=$(realpath "${2:?gold}")
truth=$(realpath "${3:?truth}")
records=$(realpath "${4:?actual fitted native records}")
stop_records=$(realpath "${5:?actual fitted supplied-stop records}")
reference=$(realpath "${6:?frozen decision reference}")
reference_sha=${7:?published reference SHA-256}
out=$(realpath -m "${8:?fresh negative-control root}")
mkdir "$out"

reject() {
    local name=$1 filter=$2 diagnostic=$3 arm=${4:-geometry.actual.native}
    local source=${5:-$records}
    jq -sce "$filter | .[]" "$source" > "$out/$name.jsonl"
    if bash "$grading/grade.sh" "$gold" "$truth" "$out/$name.jsonl" "$arm" \
            "$out/$name-grade" /dev/null "$reference" "$reference_sha" > "$out/$name.log" 2>&1; then
        echo "accepted defective control $name" >&2
        return 1
    fi
    grep -Fq "$diagnostic" "$out/$name.log"
}

at_copy='first(to_entries[]|.key as $row|.value.client.native_interval.steps|to_entries[] |
    select(.value.commit=="copy")|[$row,.key]) as [$row,$step]'
at_release='first(to_entries[]|.key as $row|.value.client.native_interval.steps|to_entries[] |
    select(.value.event=="verified_end" and .value.ordinary_id!=1)|[$row,.key]) as [$row,$step]'
# Preserve the previous-ID chain and every reported final commit. The writer
# proposal must still attest to an ordinary EOS on the same release step.
eos_stream='
    .stop="eos" |
    .generated_ids=(.generated_ids[:$step]+[1]) |
    .native_interval.steps=(.native_interval.steps[:$step+1]) |
    .native_occurrence_steps=(.native_occurrence_steps[:$step+1]) |
    .native_interval.steps[$step].committed_id=1 |
    .native_occurrence_steps[$step].committed_id=1'

reject wrong-prefix '.[0].client.native_occurrence_steps[0].previous_id += 1' 'committed prefix identity'
reject false-eos 'first(to_entries[]|select(.value.client.stop!="eos")|.key) as $i |
    .[$i].client.stop="eos"' 'invalid EOS stream'
reject outside-source "$at_copy"' |
    .[$row].client.native_interval.steps[$step].source_pos=(.[$row].client.prompt_ids|length)' \
    'invalid copy address'
reject forged-native-eos "$at_release | .[\$row].client |= ($eos_stream)" \
    'ordinary commit differs from writer proposal'
reject forged-ordinary-eos "$at_release | .[\$row].client |= ($eos_stream"' |
    .native_interval.steps[$step].ordinary_id=1)' 'span ordinary ID differs from writer proposal'
reject noread-with-copy "$at_copy"' |
    .[$row].client.native_interval |= (.selection.span=null | .steps[$step].cursor=0 | .steps[$step].start=0)' \
    'NoRead cannot authorize a span trace'
reject missing-selection "$at_copy"' | del(.[$row].client.native_interval.selection)' \
    'missing or conflicting interval selection'
reject missing-interval "$at_copy"' | del(.[$row].client.native_interval)' \
    'missing interval trace for declared decoder'
reject wrong-cursor "$at_copy"' | .[$row].client.native_interval.steps[$step].cursor+=1' \
    'invalid selected interval cursor or release'
reject early-forced-stop "$at_copy | .[\$row].client |= ($eos_stream"' |
    .native_interval.steps[$step] |=
    (.event="verified_end" | .action="stop" | .commit="supplied_stop" | .cursor=.end |
     .source_pos=null | .source_id=null))' \
    'invalid selected interval cursor or release' geometry.actual.supplied-stop "$stop_records"
reject unflagged-forced-stop "$at_copy"' | .[$row].client.native_interval.supplied_stop=false' \
    'interval stop flag does not match release control' geometry.actual.supplied-stop "$stop_records"
reject supplied-source-on-learned-stop "$at_copy"' |
    .[$row].client.native_interval |= (.supplied_source=.selection.span | .selection=null)' \
    'learned arm cannot carry a supplied source' geometry.actual.supplied-stop "$stop_records"
reject partial-learned-stop '.[:-1]' 'unexpected case roster or ordering' \
    geometry.actual.supplied-stop "$stop_records"
reject reordering 'reverse' 'unexpected case roster or ordering'
reject unregistered-arm '.' 'unregistered grading arm' fabricated.supplied-stop
reject tie-as-read "$at_copy"' | .[$row].client.native_interval.selection |=
    (.best_candidate_score=.no_read_score)' 'invalid learned score or NoRead tie'
reject legal-but-wrong-score "$at_copy"' | .[$row].client.native_interval.selection.best_candidate_score+=1' \
    'learned decision differs from frozen cold reference'
reject wrong-weight '.[0].client.native_interval.identity.state_sha256=("0"*64)' \
    'learned reader artifact, arm or weight identity mismatch'
reject disguised-legacy '.[0].client.model_schema="uor-r4.lut-discrete-native-source-choice/1"' \
    'learned reader artifact, arm or weight identity mismatch'
# Redirect a well-formed copy to another original occurrence of the exact same
# IDs. Update every active coordinate and preserve all writer commits/scores.
reject legal-but-wrong-winner '
    first(to_entries[] | .key as $row | .value.client as $client |
      $client.native_interval.selection.span as $source | select($source!=null) |
      ($source.end-$source.start) as $n | range(4;($client.prompt_ids|length)-$n-3) as $position |
      select($position!=$source.start and
        ($client.prompt_ids[$position:$position+$n]|tojson)==($client.prompt_ids[$source.start:$source.end]|tojson)) |
      [$row,$position-$source.start]) as [$row,$delta] |
    .[$row].client.native_interval |=
      (.selection.span |= (.start+=$delta | .end+=$delta) |
       .steps |= map(if .phase=="first" then
         .start+=$delta | .end+=$delta | .cursor+=$delta |
         (if .source_pos==null then . else .source_pos+=$delta end)
         else . end))' 'learned decision differs from frozen cold reference'

jq -sc 'first(to_entries[]|select(.value.swapped_case_id!=null)|.key) as $i |
    .[$i].swapped_case_id=.[$i].case_id | .[]' "$gold" > "$out/unmatched-gold.jsonl"
if bash "$grading/grade.sh" "$out/unmatched-gold.jsonl" "$truth" "$records" geometry.actual.native \
        "$out/unmatched-grade" /dev/null "$reference" "$reference_sha" \
        > "$out/unmatched.log" 2>&1; then exit 1; fi
grep -Fq 'gold and truth do not match a sealed panel' "$out/unmatched.log"
jq '.models["geometry.actual"].panels.held[0].selection.no_read_score+=1' "$reference" > "$out/forged-reference.json"
forged_sha=$(sha256sum "$out/forged-reference.json" | cut -d ' ' -f1)
if bash "$grading/grade.sh" "$gold" "$truth" "$records" geometry.actual.native \
        "$out/forged-reference-grade" /dev/null "$out/forged-reference.json" "$forged_sha" \
        > "$out/forged-reference.log" 2>&1; then exit 1; fi
grep -Fq 'unrecognized published decision reference' "$out/forged-reference.log"
(cd "$out" && sha256sum *.log > REJECTIONS.sha256)
