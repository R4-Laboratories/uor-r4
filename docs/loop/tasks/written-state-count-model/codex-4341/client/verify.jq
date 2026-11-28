# Appended to the hash-pinned #4337 definitions by verify.sh.
def insist($condition;$why): if $condition then . else error($why) end;

def verify_trace($case;$pack;$device;$mode;$tokenizer):
  . as $capt | .client as $r | $case.id as $id |
  ($case.target.kind=="source") as $positive |
  ($mode=="ordinary") as $ordinary |
  ($mode=="gold-stop") as $forced |
  $r.native_interval as $native |
  insist($r.schema=="uor-r4.uor-chat-discrete-text/1" and
         $r.protocol=="uor-r4.literal-role-dialogue/2" and
         $r.tokenizer_sha256==$tokenizer and $r.artifact_sha256==$pack.artifact_sha256 and
         $r.model_schema==interval_schema and $r.decoding==interval_schema and
         $native.identity==$pack.interval and $r.verified_span_release==null and
         $r.authenticated_copy_steps==null and $r.native_period_release==null and
         ($native.steps|type)=="array" and
         ($r.native_occurrence_steps|type)=="array" and
         $r.prompt_ids==$case.prompt_ids and ($r.text|type)=="string" and
         natural($r.context) and $r.context>=($case.prompt_ids|length);
         "ordinary writer artifact, opening or trace identity mismatch: "+$id) |
  (if $ordinary then
     learned_provenance($r;$id;{};{artifact_sha256:$pack.artifact_sha256,identity:$pack.interval};$device)
   else
     insist($positive and $native.selection==null and $native.supplied_source==$case.target.span;
            "supplied source differs from original registered address: "+$id)
   end) |
  $r.generated_ids as $ids |
  insist(($ids|type)=="array" and ($ids|length)>0 and ($ids|length)<=16 and
         all($ids[]; natural(.) and .<=4294967295) and
         ($ids[:-1]|index(1))==null and
         (if $r.stop=="eos" then $ids[-1]==1
          elif $r.stop=="max_tokens" then ($ids|length)==16 and $ids[-1]!=1 else false end);
         "invalid or omitted EOS stream: "+$id) |
  events($r) as $ev | ordinary_events($r) as $occ |
  insist(($occ|map(.committed_id))==$ids and
         (($ev|length)==0 or ($ev|map(.committed_id))==$ids) and
         all($occ|to_entries[]; .key as $i |
             (($r.prompt_ids|length)+$i) as $prefix_total |
             .value.prefix_len==([$prefix_total,$r.context]|min) and
             .value.prefix_start==($prefix_total-.value.prefix_len) and
             .value.previous_id==(if $i==0 then $r.prompt_ids[-1] else $ids[$i-1] end));
         "missing trace or committed prefix identity mismatch: "+$id) |
  ordinary_bound($r;$ev;$occ;$id) |
  selected_interval($r;{user_bounds:$case.fact};$ev;$id;{stop:$forced}) |
  copy_events($r) as $copies |
  insist(all($copies[]; . as $copy |
             $copy.kind=="span" and $copy.pos<($r.prompt_ids|length));
         "copy event lacks an original source address: "+$id) |
  (if $positive then $case.target.expected_ids else null end) as $expected |
  (if $positive then ([$case.target.span.start,$case.target.span.end] as [$start,$end] |
       [$copies[]|[.offset,.pos]]==[range($start;$end) as $i |
           [$i-$start+$case.target.response_lexical_offset,$i]]) else false end) as $addresses |
  {case_id:$id,opening_sha256:$case.opening_sha256,capture_sha256:$capt.capture_sha256,
   generated_ids:$ids,stop:$r.stop,selection:$native.selection,
   source_admitted:(selection($r)!=null),
   original_source_commits:($copies|map(del(.kind))),
   text:$r.text, selected_or_supplied:selection($r), verified_end:native_release($ev),
   first_wrong:first_wrong($expected;$ids), exact_addresses:$addresses,
   exact_through_eos:($positive and $ids==$expected and $r.stop=="eos"),
   exact_through_natural_eos:($positive and $ids==$expected and $r.stop=="eos" and
                             ($forced|not) and $addresses),
   canonical_initial_space_missed:($positive and $case.target.response_lexical_offset==1 and $ids[0]!=223),
   deterministic_visible:(if $positive then null else $r.text|status_by_text end),
   negative_group:($case.target.group // null)};

$cases[0] as $all |
(if $mode=="ordinary" then $all else $all|map(select(.target.kind=="source")) end) as $want |
insist(map(.case_id)==($want|map(.id));"writer case roster omission, repetition or reordering") |
$model[0].validation as $validation |
insist(($validation.reference|map(.case_id)|sort)==($all|map(.id)|sort) and
       ($validation.capture|map(.request_id)|sort)==($all|map(.id)|sort);
       "independent validation score roster differs from writer stage") |
($validation.reference|INDEX(.case_id)) as $device |
($validation.capture|INDEX(.request_id)) as $cold |
[to_entries[] | .key as $i | $want[$i] as $case |
 ($cold[$case.id].features|map(select(.arm==$key))) as $right_features |
 insist($cold[$case.id].opening_sha256==$case.opening_sha256 and ($right_features|length)==1 and
        $right_features[0].case_id==$case.id and $right_features[0].scores==$device[$case.id].scores;
        "independent validation score and cold capture differ for "+$case.id) |
 .value | verify_trace($case;$pack[0];$device[$case.id].scores;$mode;$tokenizer)]
