#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ $# == 5 || $# == 9 ]] || {
    echo 'usage: project.sh VERIFIED TRACE_SHA VISIBILITY VISIBLE_SHA FRESH_OUT [REVIEW_A REVIEW_A_SHA REVIEW_B REVIEW_B_SHA]' >&2
    exit 2
}
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/common.sh"
verified=$(realpath "$1")
visibility=$(realpath "$3")
out=$5
sealed "$verified" TRACE.sha256 "$2" trace.json ordinary.json gold-native.json gold-stop.json
sealed "$visibility" VISIBLE.sha256 "$4" bindings.json visibility.jsonl blind-review.jsonl
jq -e --arg digest "$2" '.trace_sha256==$digest' "$visibility/bindings.json" >/dev/null
jq -e '.arm=="natural.geometry.actual"' "$verified/trace.json" >/dev/null
a=/dev/null
b=/dev/null
a_sha=
b_sha=
if [[ $# == 9 ]]; then
    a=$(realpath "$6")
    b=$(realpath "$8")
    a_sha=$7
    b_sha=$9
    pin "$a_sha" "$a"
    pin "$b_sha" "$b"
    [[ $a_sha != "$b_sha" ]] || fail "The independent review returns are byte-identical"
fi
mkdir "$out"

jq -cen --slurpfile independent_a "$a" --slurpfile independent_b "$b" \
    --slurpfile ledger "$visibility/visibility.jsonl" \
    --slurpfile blind "$visibility/blind-review.jsonl" \
    --arg a_sha "$a_sha" --arg b_sha "$b_sha" --arg visible_sha256 "$4" '
    def validate($r):
        ($r.reviewer|type)=="string" and ($r.reviewer|length)>0 and
        ($r.decisions|type)=="array" and
        (($r.decisions|map(.key)|sort)==($blind|map(.key)|sort)) and
        all($r.decisions[]; (.reason|type)=="string" and (.reason|test("[^[:space:]]")) and
            (.grade|IN("visible_abstention","supported_premise_correction","false_factual_claim",
                       "no_decipherable_claim","requires_semantic_review")));

    (if $a_sha=="" then {a:{},b:{}}
     elif ($independent_a|length)==1 and ($independent_b|length)==1 and
          validate($independent_a[0]) and validate($independent_b[0]) and
          $independent_a[0].reviewer!=$independent_b[0].reviewer
     then {a:($independent_a[0].decisions|INDEX(.key)),b:($independent_b[0].decisions|INDEX(.key))}
     else error("expected separate authenticated reviewer returns on each blinded unresolved reply")
     end) as $reviews |
    {visible_sha256:$visible_sha256,
     reviewers:(if $a_sha=="" then [] else [
       {reviewer:$independent_a[0].reviewer,sha256:$a_sha},{reviewer:$independent_b[0].reviewer,sha256:$b_sha}] end),
     entries:[$ledger[] | . as $entry | $reviews.a[.key] as $a | $reviews.b[.key] as $b |
      {key,display,case_ids,deterministic_grade:.grade,independent_a:$a,independent_b:$b,
       grade:(if .grade=="requires_semantic_review" and $a!=null and $b!=null and $a.grade==$b.grade
              then $a.grade else .grade end)}]}
' > "$out/visibility-decisions.json"

jq -ce --slurpfile trace "$verified/trace.json" --slurpfile rows "$verified/ordinary.json" \
    --arg writer_trace_verification_sha256 "$2" \
    --arg visible_ledger_sha256 "$(digest "$out/visibility-decisions.json")" '
    .entries as $entries |
    ($rows[0]|map(select(.negative_group!=null).case_id)|sort) as $negatives |
    if ($entries|map(.case_ids[])|sort)!=$negatives or
       ($entries|map(.key)|length)!=(($entries|map(.key)|unique)|length)
    then error("final visibility ledger must cover each negative exactly once") else . end |
    ($entries|map(. as $entry|.case_ids[]|{id:., visible:$entry.grade,visible_ledger_key:$entry.key})|INDEX(.id)) as $status |
    ($trace[0]|{roster_sha256,scores_sha256,artifact_sha256,grader_sha256})+
    {schema:"uor-r4.reciprocal-transfer-validation-writer-evidence/1",
     $writer_trace_verification_sha256,$visible_ledger_sha256,
     cases:[$rows[0][] |
       . as $row | $status[.case_id] as $label |
       if $label!=null and ($entries|map(select(.key==$label.visible_ledger_key))[0].display.reply)!=$row.text
       then error("adjudicated reply differs from verified capture") else . end |
       {case_id,opening_sha256,capture_sha256,generated_ids,stop,selection,source_admitted,original_source_commits,
        visible:$label.visible,visible_ledger_key:$label.visible_ledger_key}]}
' "$out/visibility-decisions.json" > "$out/writer.json"

jq -cn --slurpfile ordinary "$verified/ordinary.json" --slurpfile native "$verified/gold-native.json" \
    --slurpfile forced "$verified/gold-stop.json" '
    ($ordinary[0]|INDEX(.case_id)) as $a | ($forced[0]|INDEX(.case_id)) as $c |
    [$native[0][] | . as $b | $a[.case_id] as $ordinary | $c[.case_id] as $forced |
      {case_id,ordinary:$ordinary,gold_native:$b,gold_stop:$forced,
       selection_difference_at_same_release:($ordinary.selected_or_supplied!=$b.selected_or_supplied and
         ($ordinary.exact_through_natural_eos|not) and $b.exact_through_natural_eos),
       supplied_stop_changes_outcome:($b.exact_addresses and $forced.exact_addresses and
         $b.verified_end!=null and $forced.verified_end!=null and
         $b.verified_end.cursor==$forced.verified_end.cursor and
         $b.verified_end.ordinary_id==$forced.verified_end.ordinary_id and $b.verified_end.ordinary_id!=1 and
         $forced.verified_end.commit=="supplied_stop" and $forced.exact_through_eos and
         ($b.exact_through_eos|not))}]
' > "$out/diagnostics.json"
(cd "$out" && sha256 writer.json visibility-decisions.json diagnostics.json > WRITER.sha256)
