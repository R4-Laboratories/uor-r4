[
  $terms[] | select(.candidate == $id)
] as $fit
| ($fit | length) == ($oracle | length) and ($fit | length) > 0
  and all(range(0; $fit | length);
    . as $i
    | $fit[$i] as $f | $oracle[$i] as $o
    | $f.row == $o.row and $f.sign == $o.sign and $f.word == $o.word
      and $f.requested == $o.requested_site and $f.rival == $o.rival_site
      and $f.local_cost == [$o.local_sites[$f.requested].cost[1], $o.local_sites[$f.rival].cost[1]]
      and $f.local_center == [
        $o.local_sites[$f.requested].transport_center, $o.local_sites[$f.rival].transport_center
      ]
      and [$f.joint[].cost] == [$o.requested.tuple.best, $o.rival.tuple.best]
      and [$f.joint[].ties] == [$o.requested.tuple.ties, $o.rival.tuple.ties]
      and [$f.joint[].eligible] == [$o.requested.tuple.eligible, $o.rival.tuple.eligible]
      and $f.joint_win == ($o.requested.tuple.best < $o.rival.tuple.best)
  )
