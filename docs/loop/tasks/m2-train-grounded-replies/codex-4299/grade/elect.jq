def need($test;$text): if $test then . else error($text) end;
. as $records | .[0] as $base | .[1] as $full |
need(length==18 and map(.candidate)==$schedule and $base.candidate=="base" and $base.changed==0 and
    $full.candidate=="full" and all(.[]; .assigned_tail_l1>=$full.assigned_tail_l1 and
    .assigned_l1-.assigned_tail_l1==$base.assigned_l1-$base.assigned_tail_l1);
    "median optimum, schedule or fixed prefix changed") |
[to_entries[]|select(.value.lost_protected==[])] as $eligible |
($eligible|min_by([.value.assigned_tail_l1,-(.value.far_full|length),.value.changed,.key])) as $choice |
($choice.value//$base) as $out |
{arm:$arm,qualified:($choice!=null),chosen:$out.candidate,base:$base,full:$full,final:$out,
 objective_before:$base.assigned_tail_l1,objective_after:$out.assigned_tail_l1,
 feasible_proposals:([$eligible[]|select(.key>0)]|length),proposals:17}

