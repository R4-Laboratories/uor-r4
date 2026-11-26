def require($ok; $why): if $ok then . else error($why) end;
def choose($base; $moves): if ($moves|length)==0 then $base else
  $moves | min_by([.hinge,-(.far_full|length),-.strict_wins,.changed,.id]) end;
.[0] as $base | .[1:] as $moves |
require($base.id==null and $base.changed==0 and ($moves|length)>0 and
        ($moves|map(.id))==[range($moves|length)] and ($moves|map(.id))==($menu|map(.id)) and
        all($moves[];.changed>=0); "candidate budget, base or order differs") |
[$moves[] | select(.hinge<$base.hinge)] as $lower |
[$moves[] | select((.lost_protected|length)==0)] as $safe |
[$lower[] | select((.lost_protected|length)==0)] as $both |
def step($w): $w+{gained_from_round_base:($w.far_full-$base.far_full),
                   lost_from_round_base:($base.far_full-$w.far_full)};
{base:$base,free:step(choose($base;$lower)),protected:step(choose($base;$both)),
 counts:{proposals:($moves|length),changed:([$moves[]|select(.changed>0)]|length),
  protected:($safe|length),far_improving:($lower|length),protected_and_far_improving:($both|length),
  protected_and_far_improving_with_new_far:([$both[]|select((.far_gained|length)>0)]|length),
  protected_and_far_improving_with_net_far_gain:([$both[]|select((.far_gained|length)>(.far_lost|length))]|length)},
 by_kind:([$menu[] as $m | $moves[$m.id] as $v |
   {kind:$m.kind, lower:($v.hinge<$base.hinge),safe:(($v.lost_protected|length)==0)}] |
   group_by(.kind)|map({kind:.[0].kind,proposals:length,
     far_improving:map(select(.lower))|length,protected:map(select(.safe))|length,
     both:map(select(.lower and .safe))|length})),
 first_near_rejection:([$moves[]|select((.lost_protected|length)>0)]|first),
 first_far_improving_near_rejection:([$lower[]|select((.lost_protected|length)>0)]|first)}
