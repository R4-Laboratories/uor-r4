#!/usr/bin/env bash
# Finite fabricated wire fixtures. No tokenizer, fitted artifact or client process.
set -euo pipefail
umask 077
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/common.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
mkdir -p stage capture/{ordinary,gold-native,gold-stop}
jq -n '
    def t($offset): {kind:"source",span:{start:1,end:2},source_ids:[33],
        expected_ids:(if $offset==1 then [223,33,1] else [33,1] end),
        response_lexical_offset:$offset};
    def row($id;$target): {id:$id,case_id:$id,text:null,opening_sha256:("0"*64),prompt_ids:[24,33,5,33,206],
        fact:[1,4],passage:"The car has four wheels.",question:"What is its color?",target:$target};
    [row("exact";t(0)),row("wrong";t(1)),row("extra";t(0)),
      row("abstain";{kind:"no_read",group:"unanswered"}),
      row("assert-a";{kind:"no_read",group:"unanswered"}),
      row("assert-b";{kind:"no_read",group:"unanswered"})]
' > stage/cases.json
jq -n '{roster_sha256:("0"*64),tokenizer_sha256:("1"*64)}' > stage/bindings.json
(cd stage && sha256 cases.json bindings.json > STAGE.sha256)
stage_sha=$(digest stage/STAGE.sha256)

jq -n --arg schema "$schema" '{
    schema:$schema,artifact_sha256:("2"*64),interval:{base_sha256:("3"*64),arm:"geometry",
        state_sha256:("4"*64),state_bytes:8192,fit_sha256:("5"*64)}
}' > pack.json
jq -n --slurpfile cases stage/cases.json --slurpfile pack pack.json '
    def row($c): {case_id:$c.id,scores:{candidates:[(if $c.target.kind=="source" then 1 else -1 end)],
        selection:{span:$c.target.span,no_read_score:0,best_candidate_score:(if $c.target.kind=="source" then 1 else -1 end)}}};
    [$cases[0][] as $c | row($c)] as $device |
    {roster_sha256:("0"*64),tokenizer_sha256:("1"*64),base_writer_sha256:("3"*64),
     models:[{arm:{mixture:"natural",key:"geometry",fit:"actual"},artifact_sha256:("2"*64),state_sha256:("4"*64),
       validation:{reference:$device,capture:[$cases[0][] as $c |
         {request_id:$c.id,opening_sha256:$c.opening_sha256,features:[row($c)+{arm:"geometry"}]}]}}]}
' > score.json
score_sha=$(digest score.json)
pack_sha=$(digest pack.json)

for mode in ordinary gold-native gold-stop; do
    jq -cn --arg mode "$mode" --slurpfile cases stage/cases.json \
        --slurpfile score score.json --slurpfile pack pack.json '
      def spanstep($span;$event;$cursor;$proposed;$committed;$stop):
        {event:$event,cursor:$cursor,ordinary_id:$proposed,committed_id:$committed,
         start:(if $cursor==null then null else $span.start end),
         end:(if $cursor==null then null else $span.end end),
         phase:(if $cursor==null then null else "first" end),
         second_event:null,prompt_end:5,
         source_pos:(if $event=="copy" then $cursor else null end),
         source_id:(if $event=="copy" then 33 else null end),
         action:(if $event=="verified_end" then (if $stop then "stop" else "native" end) else null end),
         commit:(if $event=="copy" then "copy" elif $stop then "supplied_stop" else "ordinary" end)};

      ($score[0].models[0].validation.reference|INDEX(.case_id)) as $ref |
      $cases[0][] as $c | $c.id as $id | $c.target.span as $span |
      select($mode=="ordinary" or $span!=null) |
      ($mode=="gold-stop") as $forced |
      (if $span==null then []
       elif $id=="wrong" then [
         spanstep($span;"opening_mismatch";1;22;22;false),
         spanstep($span;"ordinary";null;1;1;false)]
       else [
         spanstep($span;"copy";1;74;33;false),
         spanstep($span;"verified_end";2;(if $id=="extra" then 7 else 1 end);
                  (if $id=="extra" and ($forced|not) then 7 else 1 end);$forced),
         (if $id=="extra" and ($forced|not) then spanstep($span;"ordinary";null;1;1;false) else empty end)]
       end) as $steps |
      (if $span!=null then ($steps|map(.committed_id)) elif $id=="abstain" then [9,1] else [17,1] end) as $ids |
      {case_id:$id,client:{artifact_sha256:$pack[0].artifact_sha256,model_schema:$pack[0].schema,
        schema:"uor-r4.uor-chat-discrete-text/1",protocol:"uor-r4.literal-role-dialogue/2",
        tokenizer_sha256:("1"*64),context:512,decoding:$pack[0].schema, prompt_ids:$c.prompt_ids,
        generated_ids:$ids,stop:"eos",
        text:(if $id=="abstain" then "Cannot be determined." elif $span==null then "The car is blue." else "Fixture" end),
        native_interval:{identity:$pack[0].interval,supplied_stop:$forced,steps:$steps,
          selection:(if $mode=="ordinary" then $ref[$id].scores.selection else null end),
          supplied_source:(if $mode=="ordinary" then null else $span end),
          leading_space:($id=="wrong"),leading_space_reached:(if $id=="wrong" then false else null end)},
        native_occurrence_steps:[$ids|to_entries[] | .key as $i |
          {committed_id:.value,native_proposal:($steps[$i].ordinary_id//.value),decision:"disabled",
           copied:false,source_admitted:false,prefix_start:0,prefix_len:(5+$i),
           previous_id:(if $i==0 then 206 else $ids[$i-1] end)}]}}
    ' > "capture/$mode/records.jsonl"
done

jq -c '.[] | select(.target.kind=="source") |
    {case_id:.id,text:null,peer:null,control:(.target.span+{ids:.target.source_ids,prompt_ids})}
' stage/cases.json > capture/gold-native/actions.jsonl
cp capture/gold-native/actions.jsonl capture/gold-stop/actions.jsonl
: > capture/ordinary/INPUTS.sha256
jq -n --arg stage_sha256 "$stage_sha" --arg scores_sha256 "$score_sha" --arg pack_sha256 "$pack_sha" '
    {$stage_sha256,$scores_sha256,$pack_sha256,arm:"natural.geometry.actual"}
' > capture/bindings.json
seal_capture() {
    (cd capture && sha256 bindings.json ordinary/{INPUTS.sha256,records.jsonl} \
        gold-native/{actions,records}.jsonl gold-stop/{actions,records}.jsonl > CAPTURE.sha256)
}
verify() {
    seal_capture
    bash "$here/verify.sh" stage "$stage_sha" score.json "$score_sha" pack.json "$pack_sha" \
        natural.geometry.actual capture "$(digest capture/CAPTURE.sha256)" "$1"
}
verify verified
trace_sha=$(digest verified/TRACE.sha256)
bash "$here/visibility.sh" stage "$stage_sha" verified "$trace_sha" visibility
visible_sha=$(digest visibility/VISIBLE.sha256)
bash "$here/project.sh" verified "$trace_sha" visibility "$visible_sha" unresolved
jq -e '
    INDEX(.case_id) |
    .exact.exact_through_natural_eos and
    .wrong.canonical_initial_space_missed and .wrong.first_wrong=={offset:0,expected:223,actual:22} and
    .extra.first_wrong=={offset:1,expected:1,actual:7}
' verified/ordinary.json >/dev/null
jq -e '.cases|INDEX(.case_id)|
    .abstain.visible=="visible_abstention" and
    .["assert-a"].visible=="requires_semantic_review" and
    .["assert-a"].visible_ledger_key==.["assert-b"].visible_ledger_key
' unresolved/writer.json >/dev/null
jq -e 'INDEX(.case_id)|
    .extra.supplied_stop_changes_outcome and .extra.gold_stop.exact_through_eos and
    (.extra.gold_stop.exact_through_natural_eos|not)
' unresolved/diagnostics.json >/dev/null

for reviewer in a b; do
    jq -s --arg reviewer "$reviewer" '
        {reviewer:$reviewer,decisions:[.[]|{key,grade:"false_factual_claim",
         reason:"The passage does not state a color."}]}
    ' visibility/blind-review.jsonl > "review-$reviewer.json"
done
bash "$here/project.sh" verified "$trace_sha" visibility "$visible_sha" adjudicated \
    review-a.json "$(digest review-a.json)" review-b.json "$(digest review-b.json)"
jq -e '.cases|INDEX(.case_id)|.abstain.visible=="visible_abstention" and
    .["assert-a"].visible=="false_factual_claim" and .["assert-b"].visible=="false_factual_claim"
' adjudicated/writer.json >/dev/null
jq '.decisions[0].grade="no_decipherable_claim"' review-b.json > review-disagree.json
bash "$here/project.sh" verified "$trace_sha" visibility "$visible_sha" disagreed \
    review-a.json "$(digest review-a.json)" review-disagree.json "$(digest review-disagree.json)"
jq -e '.cases|INDEX(.case_id)|.["assert-a"].visible=="requires_semantic_review"' disagreed/writer.json >/dev/null

cp capture/ordinary/records.jsonl ordinary.jsonl
reject() {
    local label=$1 edit=$2 diagnostic=$3
    jq -cs "$edit | .[]" ordinary.jsonl > capture/ordinary/records.jsonl
    if verify "$label" > "$label.log" 2>&1; then fail "Accepted defective fixture $label"; fi
    grep -Fq "$diagnostic" "$label.log" || { cat "$label.log"; return 1; }
}

reject forged-filled-eos '
    .[2].client |= (.generated_ids=[33,1] |
      .native_interval.steps |= (.[0:2]|.[1].committed_id=1) |
      .native_occurrence_steps |= (.[0:2]|.[1].committed_id=1))
' 'ordinary commit differs from writer proposal'
reject forged-ordinary-id '
    .[2].client |= (.generated_ids=[33,1] |
      .native_interval.steps |= (.[0:2]|.[1].committed_id=1|.[1].ordinary_id=1) |
      .native_occurrence_steps |= (.[0:2]|.[1].committed_id=1))
' 'span ordinary ID differs from writer proposal'
reject missing-case '.[:-1]' 'writer case roster omission'
reject missing-trace '.[0].client.native_occurrence_steps |= .[:-1]' 'missing trace or committed prefix'
reject missing-eos '.[0].client |= (.generated_ids=[33]|.stop="max_tokens")' 'invalid or omitted EOS stream'
reject wrong-address '.[0].client.native_interval.steps[0].source_pos=3' 'invalid selected interval cursor'
reject different-cold-source '.[0].client.native_interval.selection.span={start:3,end:4}' 'learned decision differs'
reject forged-prefix '.[0].client.native_occurrence_steps[1].previous_id=5' 'committed prefix'
printf 'PASS ordinary writer trace, source, EOS, omission, committed ID, deduplication and visibility fixtures\n'
