def ensure($q;$why): if $q then . else error($why) end;
def margin: [(-.fixed_pairs),(-.fixed_rows),.blocked,.worst,.debt];
def missed($x): (4-$x.reg)+(24-$x.near);
def compare($b;$m;$e;$c):
 ($e.t.retain_admit>$b.t.retain_admit) as $eg |
 (($m.t|margin)<($b.t|margin)) as $mg |
 (missed($m)+missed($e)+missed($c)) as $miss |
 {m:$m.id,e:$e.id,combined:$c.id,gain_m:$mg,gain_e:$eg,lost_guard_reads:$miss,
  rank:[$miss,(-([$mg,$eg]|map(select(.))|length)),(-$c.t.source),(-$m.t.source),(-$e.t.source),
        (-$c.t.fixed_pairs),(-$m.t.fixed_pairs),(-$e.t.retain_admit),(-$c.t.fixed_rows),(-$m.t.fixed_rows),
        $e.t.other_losses,$c.t.worst,$m.t.worst,$c.t.debt,$m.t.debt,$e.e,$m.m]};
. as $rows | (INDEX(.[];.id)) as $map | $map.b as $b |
ensure(($rows|map(.id)|unique|length)==($rows|length) and $b.reg==4 and $b.near==24;"baseline or candidate identity") |
[$rows[]|select(.m>0 and .e>0)|. as $c|
  $map[$c.mid] as $m|$map[$c.eid] as $e|
  ensure($m.e==0 and $e.m==0 and ($m.sha!=$b.sha) and ($e.sha!=$b.sha) and $c.sha!=$e.sha and $c.sha!=$m.sha;
    "missing or unedited factor") | compare($b;$m;$e;$c)] |
ensure(length>0;"no genuinely edited grid") | sort_by(.rank) as $g |
{choice:$g[0],grids:length,protected:([$g[]|select(.lost_guard_reads==0)]|length),
 protected_both_gains:([$g[]|select(.lost_guard_reads==0 and .gain_m and .gain_e)]|length),
 ranking:$g}
