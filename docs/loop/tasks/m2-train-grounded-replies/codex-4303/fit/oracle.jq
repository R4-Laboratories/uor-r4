def fold_thirds($o; $f):
  [
    $o.local_sites[]
    | select(.site != $f.requested and .site != $f.rival and .joint.eligible > 0)
  ] as $available
  | ($available | map(.joint.best) | min // [null, null, null]) as $cost
  | ($available | map(select(.joint.best == $cost))) as $minima
  | {
      cost: $cost,
      eligible: ($available | map(.joint.eligible) | add // 0),
      ties: ($minima | map(.joint.ties) | add // 0),
      min_sites: ($minima | map(.site))
    };

($terms | map(select(.candidate == $id))) as $fit
| ($fit | length) == ($oracle | length) and ($fit | length) > 0
  and all(range($fit | length);
    . as $i
    | $fit[$i] as $f | $oracle[$i] as $o
    | $o.requested.tuple as $a
    | $o.local_sites[$f.requested] as $local
    | fold_thirds($o; $f) as $b
    | (if ($local.strict_sign | not) then "target_non_strict_sign"
       elif ($local.nonoverlap | not) then "target_overlap"
       elif ($local.role | not) then "target_no_role"
       elif $a.eligible == 0 then "target_no_completion"
       elif $b.eligible == 0 then "no_third"
       else "eligible" end) as $opportunity
    | $f.row == $o.row and $f.sign == $o.sign and $f.word == $o.word
      and $f.requested == $o.requested_site and $f.rival == $o.rival_site
      and $f.opportunity == $opportunity
      and ($f.joint[0] | [.cost, .eligible, .ties]) == [$a.best, $a.eligible, $a.ties]
      and ($f.joint[1] | {cost, eligible, ties, min_sites}) == $b
      and (if $a.eligible == 0 then $f.joint[0].site == null
           else $f.joint[0].site == $f.requested
             and $f.joint[0].local == $local.cost[1]
             and $f.joint[0].center == $local.transport_center
           end)
      and (if $b.eligible == 0 then $f.joint[1].site == null
           else ($b.min_sites | index($f.joint[1].site)) != null
             and $f.joint[1].local == $o.local_sites[$f.joint[1].site].cost[1]
             and $f.joint[1].center == $o.local_sites[$f.joint[1].site].transport_center
           end)
      and (if $opportunity != "eligible" then
             $f.margin == null and ($f.strict_win | not) and $f.debt == 0 and ($f.movable | not)
           else
             $f.strict_win == ($a.best < $b.cost)
             and $f.margin == ($b.cost[1] - $a.best[1])
             and $f.movable == ($a.best[0] == $b.cost[0])
             and $f.debt == (if $f.movable then ([0, (8 - $f.margin)] | max) else 0 end)
           end)
  )
