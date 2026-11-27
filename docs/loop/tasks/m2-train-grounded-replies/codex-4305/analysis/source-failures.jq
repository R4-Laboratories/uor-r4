def count_by($key): group_by(.[$key])|map({key:.[0][$key],value:length})|from_entries;
[$rows[] as $r |
 $meta[$r.row] as $m | $evidence[$r.row] as $e |
 if $m.case_id!=$r.case or $e.case_id!=$r.case then error("case changed") else . end |
 ([$e.good[]|select(.index==$r.native.n.i)|.role] | .[0]) as $role |
 ($r.other_link.g.i==$m.opposite_governor.index) as $og |
 ($r.other_link.n.i==$m.opposite.index) as $on |
 {pair:$r.pair,selector:$r.selector,order:$r.order,signed:$r.signed,full:$r.full,heading:$r.heading,
  g:$r.native.g.i,n:$r.native.n.i,unique:$r.unique,kind:$r.kind,
  winner:(if $r.kind=="tie" or $r.kind=="absent" then $r.kind else $role//"outside" end),
  opposite:(if $og and $on then "both" elif $og then "governor only" elif $on then "noun only" else "neither" end)}
] as $obs |
{when_signed:($obs|map(select(.signed))|{rows:length,heading:map(select(.heading))|length,opposite:count_by("opposite")}),
 by_governor_order:($obs|group_by(.order)|map({order:.[0].order,rows:length,winners:count_by("winner")})),
 pairs_same_original_noun:($obs|group_by(.pair)|map(select(length==2 and all(.[];.unique) and .[0].n==.[1].n))|length),
 pairs_same_original_governor:($obs|group_by(.pair)|map(select(length==2 and all(.[];.unique) and .[0].g==.[1].g))|length)}
