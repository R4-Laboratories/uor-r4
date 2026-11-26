def need($test;$text): if $test then . else error($text) end;
. as $pairs | $prior[0] as $origin | $protected[0] as $keep |
need(($score|length)==1 and $score[0].candidate==$candidate and
    all([$score[0].assigned_l1,$score[0].assigned_tail_l1,$score[0].changed][]; type=="number" and floor==. and .>=0) and
    $score[0].assigned_l1>=$score[0].assigned_tail_l1 and
    ($pairs|map({pair,ids,block,distance}))==($origin|map({pair,ids,block,distance})) and
    ($keep.protected|length)==24 and ($keep.far|length)==40; "TRAIN identity, budget or integer score differs") |
[$pairs[]|select(.distance=="near" and .full)|.pair] as $n |
[$pairs[]|select(.distance=="far" and .full)|.pair] as $f |
($score[0]|{candidate,changed,assigned_l1,assigned_tail_l1})+
{near_full:$n,far_full:$f,lost_protected:($keep.protected-$n),far_gained:($f-$keep.far),far_lost:($keep.far-$f)}

