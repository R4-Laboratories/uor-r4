def objective: [-.fixed_pairs,-.fixed_rows,.blocked,.worst,.debt];
def losses: ([.reg,.near]|add) as $kept | 28-$kept;
. as $records | (INDEX(.[];.id)) as $record | $record.b as $base |
[$records[]|select(.m>0 and .e>0)|. as $cross |
    $record[$cross.mid] as $m|$record[$cross.eid] as $e |
    ([$m,$e,$cross]|map(losses)|add) as $loss |
    (($m.t|objective) < ($base.t|objective)) as $mg |
    ($e.t.retain_admit>$base.t.retain_admit) as $eg |
    {m:$m.id,e:$e.id,combined:$cross.id,gain_m:$mg,gain_e:$eg,lost_guard_reads:$loss,
     rank:[$loss,-([$mg,$eg]|map(select(.))|length),-$cross.t.source,-$m.t.source,-$e.t.source,
           -$cross.t.fixed_pairs,-$m.t.fixed_pairs,-$e.t.retain_admit,-$cross.t.fixed_rows,-$m.t.fixed_rows,
           $e.t.other_losses,$cross.t.worst,$m.t.worst,$cross.t.debt,$m.t.debt,$e.e,$m.m]}] | sort_by(.rank) as $order |
if $order==$election[0].ranking and $order[0]==$election[0].choice then
    {choice:$order[0],grids:($order|length),verified:true}
else error("independent factorial election order differs") end
