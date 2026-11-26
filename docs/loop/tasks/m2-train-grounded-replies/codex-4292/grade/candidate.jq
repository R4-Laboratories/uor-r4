def require($ok; $why): if $ok then . else error($why) end;
def wins($p; $distance): [$p[] | select(.distance==$distance and .full) | .pair] | sort;
. as $pairs | $initial[0] as $parent | $score[0] as $s |
require(($score|length)==1 and $s.eligible==256 and ($s.hinge|type)=="number" and $s.hinge>=0 and
        $s.strict_wins>=0 and $s.strict_losses>=0 and $s.strict_wins+$s.strict_losses<=256 and
        ($s.changed|type)=="number" and $s.changed>=0; "invalid measured score") |
require((map({pair,ids,distance,block})==($parent|map({pair,ids,distance,block}))) and
        all(.[]; (.distance=="far" and .block=="common") or (.distance=="near" and .block=="replacement"));
        "TRAIN source or split changed") |
wins($pairs;"near") as $n | wins($pairs;"far") as $f |
wins($parent;"near") as $pn | wins($parent;"far") as $pf |
$s + {near_full:$n,far_full:$f,lost_protected:($pn-$n),far_gained:($f-$pf),far_lost:($pf-$f)}
