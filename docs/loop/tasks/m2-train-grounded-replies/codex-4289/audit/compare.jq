def common: {row,sign,governor_word,gold_site,gold_in_slot,governor_admitted,eligible_for_fit,chosen_step,before,after,final,history};
if length!=($physical|length) or length==0 then error("incomplete independent replay") else . end |
[range(length) as $i | .[$i] as $r | $physical[$i] as $p |
  ($p.candidates|map({site,address,before_cost:.before.cost,after_cost:.after.cost,before_center:.before.center,after_center:.after.center})) as $pool |
  ($p.initial_rivals_after|map({site}+.after)) as $retained |
  select(($r|common)!=($p|common) or $r.governor_address!=$p.governor or $r.gold_address!=$p.gold or
     $r.pool!=$pool or ($r.pool|length)!=$p.eligible_rivals or $r.initial_rivals_after!=$retained or
     ($r.transition|join("_to_"))!=$p.flip or any($p.candidates[];. as $c |
        $c.before.margin!=(if $r.gold_in_slot then $c.before.cost-$r.before.gold_cost else null end) or
        $c.after.margin!=(if $r.gold_in_slot then $c.after.cost-$r.after.gold_cost else null end)) or
     any($p.initial_rivals_after[]; . as $v | all($r.pool[];.site!=$v.site or .address!=$v.address))) |
  {index:$i,expected:($r|common),observed:($p|common),pool_error:($r.pool!=$pool)}] as $errors |
if ($errors|length)>0 then error("physical margin differential failed: "+($errors[:1]|tostring))
else {events:length,edges_compared:(map(.pool|length)|add),epoch_reads:(map(.history|length)|add),disagreements:0} end
