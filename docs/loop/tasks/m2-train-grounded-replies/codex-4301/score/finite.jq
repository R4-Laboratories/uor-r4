def best($x): if $x==null or ($x|type)=="array" and all($x[];.==null) then null else $x end;
if ($read|length)!=9 or ($old|length)!=9 or ($native|length)!=3 then error("fixture incomplete") else
 [range($read|length) as $i | $read[$i] as $r | $old[$i] as $p |
    if [$r.row,$r.sign,$r.word,$r.requested,$r.rival]!=[$p.row,$p.sign,$p.word,$p.requested_site,$p.rival_site]
    then error("wrong requested coordinates") else . end |
    [range($r.sites|length) as $s | $r.sites[$s] as $x | $p.local_sites[$s] as $o |
      if $x.local!=(if $o.eligible then $o.cost else null end) or
         [$x.fixed.best,$x.fixed.eligible,$x.fixed.ties]!=[best($o.joint.best),$o.joint.eligible,$o.joint.ties]
      then error("inherited local or complete ranks changed") else . end] |
    ($r.sites|map(.free.best)|map(select(.!=null))|min) as $min |
    if $min!=best($native[$r.row].tuple.best) or
       ([$r.sites[]|select(.free.best==$min and $min!=null)|.free.ties]|add//0)!=$native[$r.row].tuple.ties
    then error("free site group union differs from unrestricted native") else . end |
    [$r.sites[]|select(.i!=$r.requested and .i!=$r.rival)] as $third |
    {row:$r.row,grouped:($r.sites|length),thirds:($third|length),native_min:$min,
     alternate:([$third[]|select(.free.best!=null and ((.fixed.best==null) or (.free.best<.fixed.best)))]|length),
     absent:([$r.sites[]|select(.free.best==null)]|length),tied:([$r.sites[]|select(.free.ties>1)]|length)}] |
 if any(.[];.alternate>0) and any(.[];.absent>0) and any(.[];.tied>0) then
    {pass:true,requests:length,groups:(map(.grouped)|add),alternate_governor_thirds:(map(.alternate)|add),
     absent_groups:(map(.absent)|add),tied_groups:(map(.tied)|add)}
 else error("finite fixture did not discriminate alternate governors, absent groups, and ties") end
end
