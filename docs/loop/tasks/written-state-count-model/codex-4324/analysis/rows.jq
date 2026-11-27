def id: [.pair_id,.side]|@json;
def check(v; msg): if v then . else error(msg) end;
def bare: .steps |= map(del(.release));
def obs($t;$expected):
  $t.steps as $s | [$s[].committed_id] as $out |
  ([range(0;([$out|length,$expected|length]|max)) as $i | select($out[$i]!=$expected[$i])|$i+1][0]) as $wrong |
  check($wrong==$t.first_mismatch and $t.terminal==($out[-1]==1); "independent mismatch or EOS") |
  {exact:($wrong==null and $t.terminal), wrong:$wrong, ids:$out, terminal:$t.terminal,
   b_required:(($expected|length)>=4),
   reached_b:(($expected|length)>=4 and $out[:3]==$expected[:3] and $s[2].selected_source=="verified_copy" and $s[2].refusal==0 and $s[2].requested_pos==.positions[1] and $s[2].requested_id==.ids[1]),
   choices:[$s[]|select(.stage!=null)|{stage,action_score,selected_source,committed_id,requested_id,requested_pos,release}]};
($keys|INDEX(.[];id)) as $km | ($panel|INDEX(.[];id)) as $pm | ($old|INDEX(.[];id)) as $om |
check(($km|length)==($rows|length) and ($pm|length)==($rows|length) and ($rows|map(id)|unique|length)==($rows|length) and (($old|length)==0 or ($om|length)==($rows|length));"identity cardinality") |
$rows[] | . as $r | $km[id] as $key | $pm[id] as $p | $om[id] as $prior |
check($key!=null and $p!=null and .prefix==$p.prefix and .positions==$p.positions and .ids==$p.ids and .expected==$p.expected;"panel mismatch") |
check(($old|length)==0 or ($prior!=null and ($prior.arms.learned|bare)==(.arms.learned|bare));"historical trajectory mismatch") |
{pair_id,side,contrast,family:(.family//$key.family),group:($key.quartet_id//$key.context_id),selector:$key.operative_clause_ordinal,one:$key.one_word_clause_ordinal,
 axis:($key.pair_axis//("contrast-"+(.contrast|tostring))),directive:$key.directive,context:$key.context,question:$key.question,reply:$key.reply,
 expected,positions,ids,arms:(. as $r | .arms|to_entries|map(. as $a | {key:$a.key,value:($r|obs($a.value;$r.expected))})|from_entries)}
