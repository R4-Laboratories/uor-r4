# Read the emitted integer words from the ordinary client's raw v4 record.
def stop_for:
  if .[-1]==1 then "eos" elif length==48 then "max_tokens" else "reader_exhausted" end;
def first_error($expected;$actual):
  first(range(0;([$actual,$expected]|map(length)|max)) as $i |
       select($actual[$i]!=$expected[$i]) |
       {position:($i+1),expected:$expected[$i],actual:$actual[$i]}) // null;
def inspect($ids;$want):
  {wrong:first_error($want;$ids), exact:($ids==$want and ($ids|stop_for)=="eos"),
   cap_prefix:(($ids|stop_for)=="max_tokens" and $ids==$want[:48])};
($ref[0]|INDEX(.case_id)) as $refs | ($stock|INDEX(.id)) as $stocks |
inputs as $row | $refs[$row.id] as $r | $stocks[$row.id] as $s |
$row.record.turns[1] as $t | $r.effective_grounded.nominal as $nominal |
(if $r.admission=="take" then $nominal.expected else $s.ids end) as $wanted |
(if $r.admission=="take" then $nominal.expected_text else $s.text end) as $wanted_text |
($t.question_reader_trace) as $q |
(if $q==null then [$t.role_action_trace.steps[].words[3]] else [$q.first[].words[22]] end) as $initial |
(if $q.retry==null then $initial else [$q.retry[].words[22]] end) as $returned |
if $row.exit!=0 or $r==null or $s==null or $t.prompt_ids!=$r.prompt or
   $row.record.turns[0].reply_ids!=$r.structural_history.ids or $row.record.turns[0].stop!="eos" or
   $returned!=$t.reply_ids or ($returned|stop_for)!=$t.stop or
   ($returned==$wanted and $t.reply_text!=$wanted_text)
then error("raw reply or source history differs: "+$row.id)
else {id:$row.id,arm:$arm,admission:$r.admission,distance:$r.distance,pair:$r.pair_id,
      first:inspect($initial;$wanted),returned:inspect($returned;$wanted),retry:($q.retry!=null)}
end
