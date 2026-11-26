def check($raw;$tag):
  [range($gold|length) as $i | $gold[$i] as $g | $raw[$i] as $r |
      (if $tag=="gold" then $g.sites else $g.swapped_sites end) as $target |
      if $r.row!=$g.row_index or ($target|length)!=3 or ($g.governors|length)!=2 then error("control identity")
      else {id:$g.pair_id, unique:($r.tuple.ties==1),
         focus:($r.tuple.ties==1 and ($r.links|map(.word))==$g.governors and
              $r.tuple.sites[(if $g.sign==1 then 0 else 1 end)]==$target[(if $g.sign==1 then 0 else 1 end)]),
         full:($r.tuple.ties==1 and ($r.links|map(.word))==$g.governors and $r.tuple.sites==$target),
         original:($r.tuple.ties==1 and ($r.links|map(.word))==$g.governors and
            $r.tuple.sites[(if $g.sign==1 then 0 else 1 end)]==$g.sites[(if $g.sign==1 then 0 else 1 end)])} end] |
  {rows:length,unique:(map(select(.unique))|length),focus:(map(select(.focus))|length),full:(map(select(.full))|length),
    paired_directed:(group_by(.id)|map(select(length==2 and all(.[];.focus)))|length),
    paired_original:(group_by(.id)|map(select(length==2 and all(.[];.original)))|length),
    paired_full:(group_by(.id)|map(select(length==2 and all(.[];.full)))|length)};
if ($gold|length)!=256 or ($pos|length)!=256 or ($neg|length)!=256 then error("control census")
else {gold:check($pos;"gold"),swapped:check($neg;"swapped")} end
