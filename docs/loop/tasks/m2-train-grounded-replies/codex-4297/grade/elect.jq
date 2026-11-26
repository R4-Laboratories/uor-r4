def need($ok; $why): if $ok then . else error($why) end;
def loss: if $arm=="strongest" then .strongest_loss else .all_loss end;
.[0] as $base | .[1:] as $proposals |
need(($arm|IN("strongest","all","ordered-all")) and $base.id==null and $base.changed==0 and
     ($proposals|map(.id))==[range(68)] and ($menu|map(.id))==[range(68)] and
     all($menu[]; .group==((.id/4)|floor) and .step==([1,2,4,8][.id%4])) and
     all(.[]; (loss|type)=="number" and loss>=0 and .changed>=0);
     "proposal membership, schedule, or scored objective differs") |
[.[]|select(.lost_protected|length==0)] as $safe |
($safe|min_by([loss,-(.far_full|length),.changed,(.id//-1)])) as $chosen |
($chosen//$base) as $out |
{arm:$arm,proposals:68,base:$base,chosen:$out,qualified:($chosen!=null),
 eligible_proposals:([$proposals[]|select(.lost_protected|length==0)]|length),
 improving_safe:([$safe[]|select(loss<($base|loss))]|length),
 first_near_rejection:([$proposals[]|select(.lost_protected|length>0)|{id,lost_protected}]|first),
 objective_before:($base|loss),objective_after:($out|loss),
 gained_this_round:($out.far_full-$base.far_full),lost_this_round:($base.far_full-$out.far_full)}
