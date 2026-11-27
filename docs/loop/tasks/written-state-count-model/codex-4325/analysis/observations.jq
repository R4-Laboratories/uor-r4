def id: [.pair_id,.side]|@json;
def need(p;$msg): if p then . else error($msg) end;
def trajectory($r; $arm):
  .arms[$arm] as $t | $t.steps as $s | [$s[].committed_id] as $ids | $r.expected as $e |
  ([range(0;([$ids|length,$e|length]|max)) as $i|select($ids[$i]!=$e[$i])|$i+1][0]) as $wrong |
  need($wrong==$t.first_mismatch and $t.terminal==($ids[-1]==1);"bad EOS / mismatch accounting") |
  (($e|length)>=4) as $br |
  ($br and $ids[:3]==$e[:3] and $s[2].selected_source=="verified_copy" and $s[2].refusal==0 and
   $s[2].requested_pos==$r.positions[1] and $s[2].requested_id==$r.ids[1]) as $b |
  {exact:($t.terminal and $wrong==null),wrong:$wrong,ids:$ids,b_required:$br,b_reached:$b,
   b_right:($b and $ids[3]==$e[3]),terminal:$t.terminal,
   scores:[$s[]|select(.stage!=null)|.action_score],
   addresses:[$s[]|select(.selected_source=="verified_copy")|.requested_pos]};
([$rows[]|id]|unique|length) as $n |
($keys|INDEX(.[];id)) as $key | ($panel|INDEX(.[];id)) as $p |
need($n>0 and ($rows|length)==$n and ($key|length)==$n and ($p|length)==$n;"bad key cardinality") |
$rows[]|. as $r|$key[id] as $k | $p[id] as $ref |
need($k!=null and $ref!=null and .prefix==$ref.prefix and .positions==$ref.positions and .ids==$ref.ids and .expected==$ref.expected;"row identity mismatch") |
{pair_id,side,contrast,family,context:$k.context,question:$k.question,group:$k.context_id,
 selector:$k.operative_clause_ordinal,one:$k.one_word_clause_ordinal,axis:($k.pair_axis//"direct-count-control"),
 cohort:($k.cohort//$k.partition//$k.origin//null),length:(.prefix|length),expected,positions,ids,
 directive:$k.directive,reply:$k.reply,learned:trajectory($r;"learned"),after_supplied_first:trajectory($r;"first"),ceiling:trajectory($r;"both")}
