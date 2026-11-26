def need($x;$msg): if $x then . else error($msg) end;
def relation($a;$b):
  if $a==null and $b==null then "neither" elif $a==null then "ineligible"
  elif $b==null then "unopposed" elif $a<$b then "win" elif $a>$b then "loss" else "tie" end;
def word($row;$i):
  if $i==null then null else $row.coordinate_map.words[$i] as $a |
    {i:$i,bytes:$a.bytes,bpe:$a.tokens,text:($row.text[$a.bytes.start:$a.bytes.end])} end;
def site($row;$i):
  if $i==null then null else $row.coordinate_map.sites[$i] as $a |
    {i:$i,bytes:$a.bytes,bpe:$a.tokens,text:($row.text[$a.bytes.start:$a.bytes.end])} end;
def thirds($sites;$m;$field;$row):
  [$sites[]|select(.i!=$m.requested_site_index and .i!=$m.rival_site_index and .[$field].best!=null)] as $third |
  ($third|map(.[$field].best)|min) as $cost |
  {cost:$cost, sites:[$third[]|select(.[$field].best==$cost)|{site:site($row;.i),
       words:(if $field=="fixed" then [$m.governor_word_index] else .governors end),
       ties:.[$field].ties,first_tuple:.[$field].first_tuple}],
   eligible_sites:($third|length)};
need(($meta|length)==($raw|length) and ($native|length)==($rows|length);"missing observations") |
range($meta|length) as $k | $meta[$k] as $m | $raw[$k] as $c |
$native[$m.row_index] as $n | $rows[$m.row_index] as $row |
need(($row.text|length)==($row.text|utf8bytelength);"non-ASCII source requires byte-aware decoding") |
need([$m.row_index,$m.sign,$m.governor_word_index,$m.requested_site_index,$m.rival_site_index]==
    [$c.row,$c.sign,$c.word,$c.requested,$c.rival] and $n.row==$c.row and
    $m.requested.text==$m.rival.text and $c.requested!=$c.rival and
    ($c.sites|map(.i))==[range($row.coordinate_map.sites|length)] and
    (site($row;$c.requested).text==$m.requested.text) and
    (site($row;$c.rival).bytes==$m.rival.bytes) and (site($row;$c.requested).bpe==$m.requested.tokens) and
    (word($row;$c.word).bytes==$m.governor.bytes); "mismatched focus rows or addresses") |
(if $c.sign==1 then 0 else 1 end) as $side |
$c.sites[$c.requested] as $g | $c.sites[$c.rival] as $r |
thirds($c.sites;$m;"fixed";$row) as $fixed | thirds($c.sites;$m;"free";$row) as $third |
([$g.free.best,$r.free.best,$third.cost]|map(select(.!=null))|min) as $minimum |
need(if $n.tuple.ties>0 then $n.tuple.best==$minimum else $minimum==null end;"unrestricted rank disagrees with all site groups") |
(if $side==0 then [$m.requested.index,$m.opposite.index,$m.heading.index]
 else [$m.opposite.index,$m.requested.index,$m.heading.index] end) as $gold_sites |
(if $side==0 then [$c.word,$m.opposite_governor.index] else [$m.opposite_governor.index,$c.word] end) as $gold_words |
($n.tuple.sites[$side]) as $selected | ($n.tuple.ties==1) as $unique |
(if $g.local==null or $r.local==null then null
 elif $g.local[0]!=$r.local[0] then error("same governor changed lexical distance")
 else $r.local[1]-$g.local[1] end) as $margin |
{row:$c.row,pair_id:$m.pair_id,selector:$m.selector,family:$m.family,fact:$m.fact,sign:$c.sign,order:$m.governor_order,
 governor:word($row;$c.word),requested:site($row;$c.requested),rival:site($row;$c.rival),
 local:{requested:$g.local,rival:$r.local,margin:$margin,relation:relation($g.local;$r.local)},
 joint:{requested:$g.fixed.best,rival:$r.fixed.best,relation:relation($g.fixed.best;$r.fixed.best)},
 third_fixed:$fixed,third_free:$third,requested_free:$g.free.best,rival_free:$r.free.best,
 gold_vs_third:relation($g.fixed.best;$third.cost),gold_vs_fixed_third:relation($g.fixed.best;$fixed.cost),
 source:($unique and $n.links[$side].word==$c.word and $selected==$c.requested),
 full:($unique and ($n.links|map(.word))==$gold_words and $n.tuple.sites==$gold_sites),
 native:{cost:$n.tuple.best,ties:$n.tuple.ties,source:site($row;$selected),governor:word($row;$n.links[$side].word),
  choice:(if $unique|not then "unresolved" elif $selected==$c.requested then "requested"
          elif $selected==$c.rival then "same-spelling rival" else "third site" end)}}
