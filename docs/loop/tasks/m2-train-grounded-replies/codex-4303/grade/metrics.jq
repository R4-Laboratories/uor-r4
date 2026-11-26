def counts: group_by(.)|map({key:(.[0]|tostring),value:length})|from_entries;
def measure($a;$b):
 if $a==null then {relation:"missing-target",debt:0,blocked:0}
 elif $b==null then {relation:"no-third",debt:0,blocked:0}
 else {relation:(if $a<$b then "win" elif $a>$b then "loss" else "tie" end),
       key:(if $a[0]!=$b[0] then "lexical" elif $a[1]!=$b[1] then "transport" elif $a[2]!=$b[2] then "bank" else "tie" end),
       blocked:(if $a[0]>$b[0] then 1 else 0 end),
       debt:(if $a[0]==$b[0] then [0,1+$a[1]-$b[1]]|max else 0 end)} end;
def admits($w;$s): if $s==1 then $w[0]<$w[1] else $w[1]<$w[0] end;
def pairset: sort_by(.id,.sel)|group_by(.id)|map(
 if length!=2 or .[0].sel!="first" or .[1].sel!="later" or .[0].retain!=.[1].retain then error("factorial pair identity")
 else {id:.[0].id,family:.[0].family,retain:.[0].retain,retain_focus:.[0].retain_focus,sign:.[0].sign,
    source:all(.[];.source),full:all(.[];.full),local:all(.[];.local=="win"),
    fixed:all(.[];.fixed.relation=="win"),free:all(.[];.free.relation=="win"),
    both_free:all(.[];.both_free.relation=="win"),eligible:all(.[];.focus_eligible),
    retain_admit:all(.[];.retain_admit),worst:([.[].fixed.debt]|max)} end);
def ids($list;$field): [$list[]|select(.[$field])|.id];
def grain: {pairs:length,source:([.[]|select(.source)]|length),full:([.[]|select(.full)]|length),
             fixed:([.[]|select(.fixed)]|length),free:([.[]|select(.free)]|length),eligible:([.[]|select(.eligible)]|length)};
if ($base|length)!=($meta|length) or ($obs|length)!=($meta|length) or ($raw|length)!=($meta|length)
then error("incomplete lexical or physical observations") else . end |
[range($meta|length) as $i|$meta[$i] as $m|$raw[$i] as $x|$obs[$i] as $y|$base[$i] as $b|
 if [$m.row_index,$m.pair_id,$m.selector]!=[$x.row,$x.pair_id,$x.selector] or $i!=$y.row or $i!=$b.row or
    ($y.words|length)!=($b.words|length) then error("observer and original row mismatch") else . end|
 (if $m.sign==1 then [$m.governor,$m.opposite_governor] else [$m.opposite_governor,$m.governor] end) as $w|
 (if $w[0].text=="retain" then true else false end) as $rt|
 (admits($y.words[$m.governor.index];$m.sign)) as $focus|
 if $x.local.requested!=null and ($focus|not) then error("eligible native link with closed sign gate") else . end|
 {id:$m.pair_id,sel:$m.selector,sign:$m.sign,family:$m.family,retain:$rt,retain_focus:($rt and $m.sign==1),
 source:$x.source,full:$x.full,local:$x.local.relation,choice:$x.native.choice,
 fixed:measure($x.joint.requested;$x.third_fixed.cost),free:measure($x.joint.requested;$x.third_free.cost),
 both_free:measure($x.requested_free;$x.third_free.cost),focus_eligible:($x.joint.requested!=null),
 retain_admit:($rt and admits($y.words[$w[0].index];1)),
 other_losses:([range(2) as $j|select($j==1 or ($rt|not))|($w[$j].index) as $widx|(1-2*$j) as $sign|
   select(admits($b.words[$widx];$sign) and (admits($y.words[$widx];$sign)|not))]|length),
 other_word_gains:([range($y.words|length) as $j|select($j!=$w[0].index and $j!=$w[1].index)|
   select((admits($b.words[$j];1)|not) and admits($y.words[$j];1))]|length)}
] as $r | ($r|pairset) as $p |
{pairs:($p|length),rows:($r|length),source:(ids($p;"source")|length),full:(ids($p;"full")|length),
 fixed_pairs:(ids($p;"fixed")|length),fixed_rows:([$r[]|select(.fixed.relation=="win")]|length),
 local_pairs:(ids($p;"local")|length),free_pairs:(ids($p;"free")|length),both_free_pairs:(ids($p;"both_free")|length),
 eligible_pairs:(ids($p;"eligible")|length),eligible_rows:([$r[]|select(.focus_eligible)]|length),source_rows:([$r[]|select(.source)]|length),
 blocked:([$r[].fixed.blocked]|add),worst:([$p[].worst]|add),debt:([$r[].fixed.debt]|add),
 fixed_relations:([$r[].fixed.relation]|counts),free_relations:([$r[].free.relation]|counts),
 free_keys:([$r[].free.key|select(.!=null)]|counts),native_sites:([$r[].choice]|counts),
 retain_turns:([$r[]|select(.retain)]|length),retain_admit:([$r[]|select(.retain_admit)]|length),
 retain_focus_rows:([$r[]|select(.retain_focus)]|length),retain_focus_eligible:([$r[]|select(.retain_focus and .focus_eligible)]|length),
 other_losses:([$r[].other_losses]|add),other_word_gains:([$r[].other_word_gains]|add),
 source_ids:ids($p;"source"),fixed_ids:ids($p;"fixed"),full_ids:ids($p;"full"),
 by_family:($p|group_by(.family)|map({family:.[0].family} + grain)),
 by_verb:($p|group_by(.retain)|map({retain:.[0].retain}+grain)),
 by_focus:($p|group_by([.retain_focus,.sign])|map({retain_focus:.[0].retain_focus,sign:.[0].sign}+grain))}
