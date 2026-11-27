def key: [.pair_id,.side]|@json;
def unit_id($v):
 if $v=="selector_edge" then [.group,"clause",.one]|@json
 elif $v=="clause_edge" then [.group,"selector",.selector]|@json
 else .pair_id end;
def wins($v): group_by(unit_id($v))|map(select(length==2 and all(.[];.learned.exact))|(.[0]|unit_id($v)))|sort;
def view($v): if $v=="first_direct" then map(select(.one==null and .contrast==0))
 elif $v=="post_b_direct" then map(select(.one==null and .contrast==1)) elif $v=="scope" or $v=="clause_edge" or $v=="selector_edge" then map(select(.one!=null)) else . end;
[$balanced,$linked,$first,$joint]as $rows | ["balanced","linked","first-period","joint"] as $models |
($balanced|map(key)|sort)as $base |
if all($rows[];(map(key)|sort)==$base) then . else error("paired membership mismatch") end |
{pair_identities:[range(0;2) as $i|range(2;4) as $j|(["all","scope","first_direct","post_b_direct"] + (if $split=="quartet" or $split=="held" then ["clause_edge","selector_edge"] else [] end))[] as $v|
 ($rows[$i]|view($v)|wins($v)) as $a|($rows[$j]|view($v)|wins($v)) as $b |
 {from:$models[$i],to:$models[$j],cohort:$v,previous:($a|length),retained:(($a|length)-(($a-$b)|length)),
  lost:($a-$b),gained:($b-$a)}],
 changes:[range(0;3) as $i|($rows[$i]|INDEX(.[];key)) as $m|
 {from:$models[$i],to:"joint",shifts:([$joint[]|. as $new|$m[key] as $old|
 {from:$old.learned.wrong,to:$new.learned.wrong,scope:(.one!=null)}]|group_by([.scope,.from,.to])|map(.[0]+{n:length}))}],
 examples:([($first|INDEX(.[];key)) as $f|($linked|INDEX(.[];key)) as $l|$joint[]|. as $new|
 {pair_id,side,family,scope:(.one!=null),directive,reply,positions,expected,old:$l[key].learned,first:$f[key].learned,joint:.learned}]
 |group_by([.scope,.old.wrong,.first.wrong,.joint.wrong])|map(.[0]+{n:length}))}
