def first_wrong($expected;$actual):
  if $expected==null then null
  else ([range(0;([($expected|length),($actual|length)]|max)) as $i
       | select($expected[$i]!=$actual[$i])
       | {offset:$i,expected:$expected[$i],actual:$actual[$i]}]|first // null)
  end;
def status_by_text:
  ascii_downcase | gsub("[[:space:]]+";" ") | sub("^ +";"") | sub(" +$";"") | sub("[.!?]+$";"")
  | if . == "" then "empty_reply"
    elif IN("unknown","unspecified","not specified","not stated","not given","undetermined","cannot be determined",
      "the passage does not say","the passage does not specify","the passage does not provide that information",
      "the text does not say","the text does not specify","it is not specified","it is not stated",
      "there is not enough information","i don't know","i do not know")
    then "visible_abstention"
    else "requires_semantic_review" end;
def events($r): ($r.native_interval.steps // $r.verified_span_release.steps // []);
def ordinary_events($r): ($r.native_occurrence_steps // []);
def copy_events($r):
  ((events($r)|to_entries|map(select(.value.commit=="copy")
       | {offset:.key,pos:.value.source_pos,id:.value.source_id,kind:"span"})) +
   (ordinary_events($r)|to_entries|map(select(.value.copied==true)
       | {offset:.key,pos:(.value.source_pos+.value.prefix_start),id:.value.source_id,kind:"occurrence"})))
  | sort_by(.offset);
def selection($r):
  $r.native_interval.selection.span // $r.native_interval.supplied_source // null;
def registration:
  [
    {name:"balanced-native",kind:"balanced",panels:["held"],stop:false},
    {name:"balanced",kind:"balanced",panels:["gate"],stop:false},
    {name:"supplied-native",kind:"legacy",panels:["gate"],stop:false},
    {name:"supplied-stop",kind:"legacy",panels:["gate"],stop:true},
    {name:"supplied-control-gold-native",kind:"gold",panels:["held"],stop:false},
    {name:"supplied-control-gold-stop",kind:"gold",panels:["held"],stop:true},
    {name:"supplied-control-geometry-swap",kind:"peer",panels:["held"],stop:false,model:"geometry.actual"},
    {name:"supplied-control-count-swap",kind:"peer",panels:["held"],stop:false,model:"count.actual"},
    ("geometry","count") as $reader |
    ("actual","permuted","zero") as $fit |
    ("native", (if $fit=="actual" then "supplied-stop" else empty end)) as $release |
    {name:([$reader,$fit,$release]|join(".")),kind:"learned",panels:["held","protected"],
     stop:($release=="supplied-stop"),model:([$reader,$fit]|join("."))}
  ] | map(select(.name==$arm)) |
  if length!=1 then error("unregistered grading arm: "+$arm) else .[0] end;
def natural($value): ($value|type)=="number" and $value>=0 and $value==($value|floor);
def score($value): ($value|type)=="number" and $value==($value|floor) and $value>=-524288 and $value<=524272;
def interval_schema: "uor-r4.lut-discrete-native-interval/1";
def legacy_schema: "uor-r4.verified-span-release/1";
def disabled_decoding: "greedy native occurrence disabled; frozen marker and trunk";
def reference:
  if ($decisions|length)!=1 or $decisions[0].schema!="uor-r4.ordinary-interval-decision-reference/1"
  then error("invalid frozen decision reference") else $decisions[0] end;
def registered_panel($ref;$reg):
  [$ref.panels|to_entries[]|select(.value.gold_sha256==$gold_sha and .value.truth_sha256==$truth_sha)]
  | if length!=1 then error("gold and truth do not match a sealed panel")
    elif (.[0].key as $name | $reg.panels|index($name))==null then error("arm is not registered on this panel")
    else .[0].key end;
def expected_roster($reg;$ts):
  [$gold[]| . as $g | $ts[.case_id] as $t |
    select(if $reg.kind=="gold" then .start!=null
      elif $reg.kind=="legacy" then .start!=null and $t.expected_ids==($t.source_ids+[1])
      elif $reg.kind=="peer" then .swapped_case_id!=null
      else true end) | $g.case_id];
def learned_provenance($r;$id;$reg;$model;$frozen):
  $r.native_interval as $native |
  if $r.model_schema!=interval_schema or $r.decoding!=interval_schema or $r.verified_span_release!=null or
       $r.artifact_sha256!=$model.artifact_sha256 or $native.identity!=$model.identity or $frozen==null
  then error("learned reader artifact, arm or weight identity mismatch: "+$id)
  elif $native.supplied_source!=null
  then error("learned arm cannot carry a supplied source: "+$id)
  elif ($native.selection|type)!="object" or
       ($native.selection|keys)!=["best_candidate_score","no_read_score","span"]
  then error("missing or malformed learned selection: "+$id)
  else
    $native.selection as $read |
    if (score($read.no_read_score)|not) or
       ($read.best_candidate_score!=null and (score($read.best_candidate_score)|not)) or
       (($read.best_candidate_score!=null and $read.best_candidate_score>$read.no_read_score)!=($read.span!=null))
    then error("invalid learned score or NoRead tie: "+$id)
    elif $read!=$frozen.selection
    then error("learned decision differs from frozen cold reference: "+$id)
    else . end
  end;
def other_provenance($r;$id;$reg;$balanced;$source):
  if $r.artifact_sha256!=$balanced.artifact_sha256 or $r.model_schema!=$balanced.model_schema
  then error("balanced writer artifact identity mismatch: "+$id)
  elif $reg.kind=="legacy" then
    if $r.native_interval!=null or $r.decoding!=legacy_schema or
       ($r.verified_span_release.steps[0]|{start,end})!=$source
    then error("legacy source differs from authenticated gold interval: "+$id) else . end
  elif ($reg.kind|IN("gold","peer")) and $source!=null then
    if $r.decoding!=interval_schema or $r.verified_span_release!=null or
       $r.native_interval.identity!=null or $r.native_interval.selection!=null or
       $r.native_interval.supplied_source!=$source
    then error("supplied source differs from sealed intervention: "+$id) else . end
  elif $r.native_interval!=null or $r.verified_span_release!=null or
       $r.decoding!=(if $reg.kind=="balanced" then "authenticated opening with native period release" else disabled_decoding end)
  then error("unexpected interval trace or writer mode: "+$id) else . end;
def ordinary_bound($r;$ev;$occ;$id):
  if $r.native_period_release!=null or
     any($occ[]; (if $r.native_interval!=null then .decision!="disabled"
                  else (.decision|IN("disabled","framing_rejected")|not) end) or
                  .copied!=false or .source_admitted!=false or (natural(.native_proposal)|not))
  then error("unbound ordinary writer proposal: "+$id)
  elif ($ev|length)==0 then
    if any($occ[]; .committed_id!=.native_proposal)
    then error("ordinary commit differs from writer proposal: "+$id) else . end
  elif any($ev|to_entries[]; . as $step | .value.ordinary_id!=$occ[$step.key].native_proposal)
  then error("span ordinary ID differs from writer proposal: "+$id)
  elif any($ev[]; .commit=="ordinary" and .committed_id!=.ordinary_id)
  then error("ordinary commit differs from writer proposal: "+$id)
  else . end;
def selected_interval($r;$t;$ev;$id;$reg):
  $r.native_interval as $n | selection($r) as $span | ($r.prompt_ids|length) as $end
  | if ($n.selection==null)==($n.supplied_source==null) or
       ($n.selection!=null and ($n.selection|type!="object" or (has("span")|not)))
    then error("missing or conflicting interval selection: "+$id)
    elif $n.supplied_stop!=$reg.stop
    then error("interval stop flag does not match release control: "+$id)
    elif $span==null then
      if ($ev|length)>0 or $n.leading_space!=false or $n.leading_space_reached!=null
      then error("NoRead cannot authorize a span trace: "+$id) else . end
    elif (natural($span.start)|not) or (natural($span.end)|not) or
         $span.start>= $span.end or $span.end>$end or $span.end-$span.start>8 or
         ($t.user_bounds!=null and ($span.start<$t.user_bounds[0] or $span.end>$t.user_bounds[1])) or
         any($r.prompt_ids[$span.start:$span.end][]; IN(0,1,16,201))
    then error("invalid selected original interval: "+$id)
    elif ($ev|length)!=($r.generated_ids|length) or ($n.leading_space|type)!="boolean"
    then error("incomplete selected interval trace: "+$id)
    else
      reduce $ev[] as $step ({cursor:$span.start,leading:$n.leading_space};
        .cursor as $cursor
        | (if $cursor==null then "ordinary"
           elif .leading then (if $step.ordinary_id==223 then "native_space" else "opening_mismatch" end)
           elif $cursor==$span.end then "verified_end"
           else "copy" end) as $event
        | (if $event=="verified_end" then (if $n.supplied_stop then "stop" else "native" end)
           else null end) as $action
        | (if $event=="copy" then "copy"
           elif $action=="stop" then "supplied_stop" else "ordinary" end) as $commit
        | (if $event=="copy" then $r.prompt_ids[$cursor] else null end) as $source_id
        | if $step.event!=$event or $step.action!=$action or $step.commit!=$commit or
             $step.second_event!=null or $step.prompt_end!=$end or
             $step.cursor!=$cursor or
             (if $cursor==null then [$step.start,$step.end,$step.phase]!=[null,null,null]
              else [$step.start,$step.end,$step.phase]!=[$span.start,$span.end,"first"] end) or
             $step.source_pos!=(if $event=="copy" then $cursor else null end) or
             $step.source_id!=$source_id or
             ($event=="copy" and $step.committed_id!=$source_id) or
             ($commit=="supplied_stop" and $step.committed_id!=1)
          then error("invalid selected interval cursor or release: "+$id)
          else .leading=false |
            .cursor=(if $event=="copy" then $cursor+1
                     elif $event=="native_space" then $cursor else null end)
          end)
      | if $n.leading_space_reached!=(if $n.leading_space and ($ev|length)>0
                                      then $ev[0].event=="native_space" else null end)
        then error("leading-space trace differs from opening event: "+$id) else . end
    end;
def legacy_release($r;$ev;$id;$reg):
  if any($ev|to_entries[]; .key as $i | .value as $s |
      ($s.commit=="supplied_stop" and ($s.event!="verified_end" or $s.action!="stop" or
                                     $s.committed_id!=1 or ($reg.stop|not))) or
      ($s.event=="verified_end" and
        ($s.action!=(if $reg.stop then "stop" else "native" end) or
         ($s.commit=="supplied_stop")!=$reg.stop or
         (natural($s.start)|not) or (natural($s.end)|not) or $s.start>=$s.end or
         $s.end>($r.prompt_ids|length) or $s.cursor!=$s.end or $i<($s.end-$s.start) or
         ($ev[$i-($s.end-$s.start):$i] | map(select(.commit=="copy" and
           ((.phase==$s.phase and .second_event==null) or
            ($s.phase=="second" and .second_event=="copy"))) | .source_pos))
         !=[range($s.start;$s.end)])))
  then error("unverified legacy span release: "+$id) else . end;
def native_release($events):
  [$events[] | select(.event=="verified_end") | {action,commit,ordinary_id,committed_id,cursor,end}][0];
def counts:
  def n(p): map(select(p))|length;
  {cases:length, supported:n(.supported),unsupported:n(.supported|not),
   original_interval_exact:n(.selection_exact==true),unsupported_interval_selected:n(.false_selected_source==true),
   unsupported_actual_source_commit:n(.false_committed_source==true),
   unsupported_explicit_noread:n(.explicit_noread==true and (.supported|not)),
   first_original_address_exact:n(.first_original_exact==true),
   full_original_addresses_exact:n(.copy_addresses_exact==true),
   false_same_value_address:n(.wrong_same_value_address==true),
   exact_reply_through_eos:n(.exact_through_eos==true),
   exact_reply_and_addresses:n(.exact_both==true),
   copy_then_verified_end:n(.verified_end!=null),
   release_eos_by_native:n(.verified_end.action=="native" and .verified_end.committed_id==1),
   release_eos_supplied:n(.verified_end.commit=="supplied_stop"),
   delayed_openings:n(.delayed_lexical_opening==true),
   delayed_emitted_space_first:n(.delayed_lexical_opening==true and .actual_first_id==223),
   delayed_actual_copy_after_space:n(.delayed_lexical_opening==true and .copied_after_printed_space==true),
   visible_unsupported_grades:(map(select(.supported|not)|.visible_unsupported_grade)|group_by(.)|map({case:.[0],n:length})),
   first_wrong_offsets:(map(select(.first_wrong!=null)|.first_wrong.offset)|group_by(.)|map({offset:.[0],n:length}))};
def grades($records):
  registration as $reg | reference as $ref | registered_panel($ref;$reg) as $panel
  | $ref.models[$reg.model // ""] as $model
  | ($model.panels[$panel] // [] | INDEX(.case_id)) as $frozen
  | ($gold|INDEX(.case_id)) as $gs | ($truth|INDEX(.case_id)) as $ts
  | ($visible|INDEX(.case_id)) as $vs
  | if ($gold|map(.case_id))!=$ref.panels[$panel].roster or ($truth|map(.case_id))!=$ref.panels[$panel].roster or
       ($records|map(.case_id))!=expected_roster($reg;$ts) or
       (($visible|map(.case_id))-($records|map(.case_id))|length)>0
    then error("unexpected case roster or ordering for "+$arm) else . end
  | if any($gold[] | select(.swapped_case_id!=null); . as $g |
       $g.case_id==$g.swapped_case_id or $gs[$g.swapped_case_id].swapped_case_id!=$g.case_id or
       $gs[$g.swapped_case_id].group!=$g.group)
    then error("unmatched reciprocal case pairing") else . end
  | [$records[] | .case_id as $id | .client as $r | $gs[$id] as $g | $ts[$id] as $t
    | if $g==null or $t==null then error("unknown case") else . end
    | if ($t.canonical_prefix!=null and $t.canonical_prefix!=$r.prompt_ids) then error("canonical prompt drift: "+$id) else . end
    | ($g.start!=null) as $supported
    | ($t.response_lexical_offset // 0) as $offset
    | if ($supported and ($t.expected_ids[-1]!=1 or ($t.expected_ids[:-1]|index(1))!=null)) or
         (($r.stop=="eos")!=($r.generated_ids[-1]==1)) or ($r.generated_ids[:-1]|index(1))!=null
       then error("invalid EOS stream: "+$id) else . end
    | if $supported and $r.prompt_ids[$g.start:$g.end]!=$t.source_ids
       then error("independent source ID mismatch: "+$id) else . end
    | (if ($reg.kind|IN("legacy","gold")) then {start:$g.start,end:$g.end}
       elif $reg.kind=="peer" then
          $t.passage_sha256 as $passage | $ts[$g.swapped_case_id] as $peer |
          $frozen[$g.swapped_case_id] as $expected |
          if $expected==null or $passage==null or $passage!=$peer.passage_sha256 or
             $t.canonical_prefix==null or $peer.canonical_prefix==null
          then error("missing frozen peer or original passage identity: "+$id)
          else $expected.selection.span as $span |
            if $span!=null and
               $r.prompt_ids[$span.start:$span.end]!=$peer.canonical_prefix[$span.start:$span.end]
            then error("peer span differs in original IDs: "+$id) else $span end end
       else null end) as $source
    | events($r) as $ev
    | if ($r.decoding==interval_schema and $r.native_interval==null) or
         ($r.decoding==legacy_schema and $r.verified_span_release==null)
      then error("missing interval trace for declared decoder: "+$id) else . end
    | if ($ev|length)>0 and ($ev|map(.committed_id))!=$r.generated_ids
       then error("span trace differs from committed stream: "+$id) else . end
    | if ($reg.stop|not) and any($ev[]; .commit=="supplied_stop")
       then error("supplied EOS on primary arm: "+$id) else . end
    | ordinary_events($r) as $occ
    | if ($r.native_interval!=null or ($ev|length)>0) and ($occ|length)!=($r.generated_ids|length)
       then error("missing ordinary self-feed trace: "+$id) else . end
    | if ($occ|length)>0 and (($occ|map(.committed_id))!=$r.generated_ids or
        any($occ|to_entries[];
          .key as $i| .value.previous_id != (if $i==0 then $r.prompt_ids[-1] else $r.generated_ids[$i-1] end)))
      then error("ordinary writer trace violates committed prefix identity: "+$id) else . end
    | selection($r) as $selection
    | copy_events($r) as $all_copies
    | if ($all_copies|map(.offset)|length != (unique|length)) or
         any($all_copies[]; (.pos|type)!="number" or .pos<0 or .pos!= (.pos|floor) or
           (.kind=="span" and .pos>=($r.prompt_ids|length)) or .pos>=(($r.prompt_ids|length)+.offset) or
           .id!=($r.prompt_ids+$r.generated_ids)[.pos] or .id!=$r.generated_ids[.offset])
       then error("invalid copy address, token or duplicate commit: "+$id) else . end
    | if $r.native_interval!=null then
        ordinary_bound($r;$ev;$occ;$id) | selected_interval($r;$t;$ev;$id;$reg)
      elif $r.verified_span_release!=null then
        ordinary_bound($r;$ev;$occ;$id) | legacy_release($r;$ev;$id;$reg)
      elif $reg.kind=="peer" and $source==null then ordinary_bound($r;[];$occ;$id)
      else . end
    | if $reg.kind=="learned" then learned_provenance($r;$id;$reg;$model;$frozen[$id])
      else other_provenance($r;$id;$reg;$ref.balanced;$source) end
    | ($all_copies|map(select(.pos<($r.prompt_ids|length)))) as $copies
    | ($copies|map(.pos)) as $positions
    | (if $supported then [range($g.start;$g.end)] else [] end) as $expected_positions
    | ($supported and $positions==$expected_positions
       and ($copies|map(.offset))==[range($offset;$offset+($expected_positions|length))]) as $right_address_sequence
    | ($supported and $t.expected_ids==$r.generated_ids and $r.stop=="eos") as $right_reply
    | (if $supported then null
       elif $vs[$id]!=null then $vs[$id].grade
       else $r.text|status_by_text end) as $visible_result
    | {case_id:$id,tape:$t.tape,group:$g.group,swapped_case_id:$g.swapped_case_id,supported:$supported,
       first_wrong:first_wrong($t.expected_ids;$r.generated_ids),
       actual_first_id:$r.generated_ids[0],actual_stop:$r.stop,
       selected:$selection,selection_exact:(if $selection!=null then ($supported and $selection=={start:$g.start,end:$g.end}) else null end),
       false_selected_source:(($supported|not) and ($selection!=null or any($occ[]; .source_admitted==true))),
       false_committed_source:(($supported|not) and ($copies|length)>0),
       visible_unsupported_grade:$visible_result,
       explicit_noread:($r.native_interval.selection!=null and $r.native_interval.selection.span==null),
       generated_history_copies:($all_copies|map(select(.pos>=($r.prompt_ids|length)))),
       first_original_exact:($supported and ($copies[0].pos==$g.start and $copies[0].offset==$offset)),
       wrong_same_value_address:($supported and ($copies|length)>0 and $copies[0].pos!=$g.start and $copies[0].id==$t.source_ids[0]),
       copy_addresses_exact:$right_address_sequence,exact_through_eos:$right_reply,exact_both:($right_reply and $right_address_sequence),
       actual_source_commits:$copies,verified_end:native_release($ev),
       delayed_lexical_opening:($offset>0),copied_after_printed_space:($offset==1 and $r.generated_ids[0]==223 and $copies[0].offset==1),
       phase_trace:$r.native_interval.leading_space_reached}]
;
grades(.) as $rows
| {arm:$arm,summary:($rows|counts),by_tape:($rows|group_by(.tape)|map({tape:.[0].tape,results:counts})),
   matched_pairs:($rows|map(select(.swapped_case_id!=null))|group_by([.case_id,.swapped_case_id]|sort)|map(select(length==2)
     | {group:.[0].group,strict_route:all(.[]; if .supported then .selection_exact==true else (.false_selected_source|not) end),
        strict_commit:all(.[];if .supported then .exact_both else
          (.false_committed_source|not) and .visible_unsupported_grade=="visible_abstention" end)})), rows:$rows}
