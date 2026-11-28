def need($p;$m): if $p then . else error($m) end;
def index_cases: map({key:.case_id,value:.}) | from_entries;
def first_wrong($actual;$expected):
  first(range(0;([$actual,$expected]|map(length)|max)) as $i
    | select($actual[$i]!=$expected[$i]) | {token:($i+1),expected:$expected[$i],actual:$actual[$i]}) // null;
def count_if(f): map(select(f))|length;
def counts:
  {rows:length, exact:count_if(.exact), eos:count_if(.eos), cap:count_if(.eos|not),
   eligible:count_if(.release!=null), right_source_and_period:count_if(.verified_boundary),
   selected_eos:count_if(.release.choice=="eos"), unresolved_copy:count_if(.release.unresolved_copy==true),
   kept_native_on_tie:count_if(.release.choice=="native"),
   changed_third_id:count_if(.release!=null and .release.committed_id!=.release.native_id),
   correct_third_on_verified:count_if(.verified_boundary and .expected[2]==.actual[2]),
   exact_on_verified:count_if(.verified_boundary and .exact),
   first_wrong: (group_by(.first_wrong.token)|map({key:(.[0].first_wrong.token|tostring),value:length})|from_entries)};
($truth|index_cases) as $gold
| ($ordinary|map(select(.artifact_arm=="base" and .decode_arm=="authenticated_copy"))|index_cases) as $base
| need(length==144 and ($gold|length)==48 and ($base|keys)==($gold|keys)
       and (map(.arm)|unique)==["paired","rowwise","zero"]
       and all(group_by(.arm)[]; (map(.case_id)|sort)==($gold|keys));"case census differs")
| map(. as $row | .case_id as $id | .client as $c | $base[$id].client as $b | $gold[$id] as $t
    | $b.authenticated_copy_steps as $native | $c.authenticated_copy_steps as $steps
    | $c.native_period_release as $r | $t.sources[0] as $source
    | ($native[0].copied and ($native[1].copied==false) and $native[1].committed_id==16) as $eligible
    | ($source!=null and $native[0].copied and $native[0].read_pos==$source.pos and
       $native[0].claim_pos==$source.pos and $native[0].source_id==$source.id and
       $native[0].gate_id==$source.id and $native[0].committed_id==$source.id and
       $b.prompt_ids[$source.pos]==$source.id) as $verified
    | need($c.prompt_ids==$t.prompt_ids and $c.prompt_ids==$b.prompt_ids and
           ($steps|map(.committed_id))==$c.generated_ids and $steps[:2]==$native[:2]
           and ($r!=null)==$eligible;"opening, period or causal trigger differs: "+$id)
    | need(if $eligible then
             $r.native_id==$native[2].committed_id and $r.committed_id==$steps[2].committed_id and
             ($steps[2]|del(.committed_id))==($native[2]|del(.committed_id)) and
             ($r.choice==(if $r.score<0 then "eos" elif $r.score>0 then "copy" else "native" end)) and
             ($r.unresolved_copy==($r.choice=="copy")) and
             ($r.committed_id==(if $r.choice=="eos" then 1 else $r.native_id end))
           else true end;"release action or frozen read diverged: "+$id)
    | need(if $r==null or $r.native_id==$r.committed_id then $steps==$native else
             $c.generated_ids==($b.generated_ids[:2]+[1]) end;"unexpected transcript difference: "+$id)
    | need($c.stop==(if $c.generated_ids[-1]==1 then "eos" else "max_tokens" end) and
           ($c.generated_ids[:-1]|index(1))==null and
           ($c.generated_ids[-1]==1 or ($c.generated_ids|length)==12);"termination mismatch: "+$id)
    | {case_id:$id,arm:$row.arm,group:$t.group,supported:$t.supported,release:$r,
       verified_boundary:($verified and $eligible),expected:$t.expected_ids,actual:$c.generated_ids,
       source_pos:$native[0].read_pos,expected_pos:$source.pos,
       exact:($c.generated_ids==$t.expected_ids),eos:($c.stop=="eos"),
       first_wrong:first_wrong($c.generated_ids;$t.expected_ids)})
| {summary:(group_by(.arm)|map({key:.[0].arm,value:counts})|from_entries),
   verified:(map(select(.verified_boundary))|group_by([.arm,.group])
       |map({arm:.[0].arm,group:.[0].group,counts:counts})),
   rows:.}
