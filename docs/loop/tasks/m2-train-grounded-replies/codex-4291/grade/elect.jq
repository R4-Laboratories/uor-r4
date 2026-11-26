def need($ok;$why): if $ok then . else error($why) end;
def won(f): map(select(f)|.pair)|sort;
def election: sort_by([.hinge,-.strict_wins,.step])|first.step;
need(($checkpoints|length)==9 and ($quality|map(.step))==[range(9)] and all($checkpoints[];length==128);
     "nine binding TRAIN checkpoint observations required") |
($checkpoints[0]|map({pair,ids,family,distance,block})) as $key |
need(all($checkpoints[];map({pair,ids,family,distance,block})==$key);"TRAIN pair membership changed during selection") |
($checkpoints[0]|won(.distance=="near" and .full)) as $near |
($checkpoints[0]|won(.block=="common" and .distance=="far" and .full)) as $far |
[range(9) as $i | $checkpoints[$i] as $p | ($p|won(.distance=="near" and .full)) as $n |
 ($p|won(.block=="common" and .distance=="far" and .full)) as $f |
 {step:$i,hinge:$quality[$i].hinge,strict_wins:$quality[$i].strict_wins,exact_pairs:($p|won(.exact)|length),
  full_pairs:($p|won(.full)|length),near_full:($n|length),common_far_full:($f|length),
  lost_original_near:($near-$n),gained_common_far:($f-$far),lost_common_far:($far-$f),feasible:(($near-$n)|length==0)}] as $steps |
{original_near:$near,original_common_far:$far,trajectory:$steps,selections:{
 unconstrained:($steps|election),retained:($steps|map(select(.feasible))|election)}}
