def need($ok; $why): if $ok then . else error($why) end;
. as $pairs | $initial[0] as $origin | $pin[0] as $keep | $score[0] as $s |
need(($score|length)==1 and $s.eligible==256 and $s.strict_wins+$s.strict_losses+$s.tied_edges==256 and
     $s.all_loss>=$s.strongest_loss and $s.strongest_loss>=0 and $s.changed>=0 and
     ($pairs|map({pair,ids,block,distance}))==($origin|map({pair,ids,block,distance}));
     "source identities or physical training measurements changed") |
[$pairs[]|select(.distance=="near" and .full)|.pair] as $n |
[$pairs[]|select(.distance=="far" and .full)|.pair] as $f |
need(($keep.protected|length)==24 and ($keep.far|length)==40; "protected precedent changed") |
$s+{near_full:$n,far_full:$f,lost_protected:($keep.protected-$n),
    far_gained:($f-$keep.far),far_lost:($keep.far-$f)}
