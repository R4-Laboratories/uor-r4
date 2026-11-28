# jq -nc --slurpfile gold panel-v1/gold.jsonl -f source-grade.jq decisions.jsonl
def count_if(f): map(select(f))|length;
def single($arm;$d;$t):
  $d[$arm].winner_pos as $win |
  ([$d[$arm].candidates[] | select(.pos==$win)]|first) as $c |
  {arm:$arm,id:$t.case_id,cohort:$t.cohort,split:$t.split,group:$t.contrast_group,
   delay:$t.source_delay,supported:$t.supported,target:$t.source.pos,winner:$win,token:$c.id,
   correct:(if $t.supported then $win==$t.source.pos else $win==null end),
   wrong_same_id:($t.supported and $win!=null and $win!=$t.source.pos and $c.id==$t.source.id)};
def matches:
  group_by([.arm,.cohort,.split,.group]) |
  map(select(length==2) |
    {arm:.[0].arm, cohort:.[0].cohort,split:.[0].split,group:.[0].group,
     delay:(map(select(.supported)|.delay)|max),
     complete:all(.[];.correct),
     support_flip:(map(.supported)|unique|length==2),
     different_target:(map(.target)|unique|length==2),
     pos_wins:count_if(.supported and .correct), false_admissions:count_if((.supported|not) and .winner!=null)});
($gold|INDEX(.case_id)) as $truth |
([inputs as $d | if $truth[$d.case_id]==null then error("unmapped ID") else
  ["old_framed","expanded_frozen","expanded_fitted"][] as $a | single($a;$d;$truth[$d.case_id]) end]) as $r |
if ($r|length)!=3*($gold|length) or ($r|map([.arm,.id])|unique|length)!=($r|length) then error("incomplete join") else
{rows:($r|group_by([.arm,.cohort,.split])|map({arm:.[0].arm,cohort:.[0].cohort,split:.[0].split,
  n:length,positive:count_if(.supported),address_wins:count_if(.supported and .correct),
  no_read:count_if((.supported|not) and .correct),false_admissions:count_if((.supported|not) and .winner!=null),
  wrong_same_id:count_if(.wrong_same_id)})),
 natural_pairs:($r|map(select(.cohort=="squad"))|matches|group_by([.arm,.split,.delay])|
   map(if all(.[];.support_flip) then
     {arm:.[0].arm,split:.[0].split,delay:.[0].delay,pairs:length,strict:count_if(.complete),
      positives:(map(.pos_wins)|add),false_admissions:(map(.false_admissions)|add)} else error("broken paired gold") end)),
 authored_pairs:($r|map(select(.cohort=="authored"))|matches|group_by([.arm,.split,.support_flip,.different_target])|
   map({arm:.[0].arm,split:.[0].split,support_flip:.[0].support_flip,different_target:.[0].different_target,pairs:length,strict:count_if(.complete)})),
 protected_four:[$r[] | select(.id|test("^protected-c02[1-4]$"))]}
end
