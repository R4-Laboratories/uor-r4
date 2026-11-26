def require($test;$message): if $test then . else error($message) end;
def status($candidate;$original):
    $candidate.score + {
        near_full:$candidate.near_full, far_full:$candidate.far_full,
        lost_protected:($original.near_full - $candidate.near_full),
        far_gained:($candidate.far_full - $original.far_full),
        far_lost:($original.far_full - $candidate.far_full)
    };
def winner($moves;$base;$protect):
    ($moves | map(select(.hinge < $base.hinge and
        (if $protect then (.lost_protected|length)==0 else true end))) |
     min_by([.hinge,-(.far_full|length),-.strict_wins,.changed,.id])) // $base;
def movements($winner;$base):
    $winner + {gained_from_round_base:($winner.far_full-$base.far_full),
               lost_from_round_base:($base.far_full-$winner.far_full)};
($origin[0].metrics[0]) as $original |
require(($original.near_full|length)>0 and (.metrics|length)>1 and .metrics[0].score.id==null;
    "identity and proposal premise missing") |
(.metrics[0] | status(.;$original)) as $base |
(.metrics[1:]|map(status(.;$original))) as $moves |
require(($moves|map(.id)) == [range($moves|length)]; "menu IDs changed") |
($grade | map({id,hinge,strict_wins,strict_losses,eligible,changed,near_full,far_full,lost_protected,far_gained,far_lost})) as $graded |
require(($graded == ([$base] + $moves)); "graded raw candidate records differ") |
movements(winner($moves;$base;true);$base) as $protected |
movements(winner($moves;$base;false);$base) as $free |
require($protected == $device[0].protected and $free == $device[0].free;
    "independent elections differ") |
([$moves[] | select((.lost_protected|length)>0)] | first) as $violation |
([$moves[] | select((.lost_protected|length)>0 and .hinge<$base.hinge)] | first) as $improving_violation |
def witness($bad):
    if $bad==null then null else
        .metrics[$bad.id+1].near_failures as $failures |
        ($bad.lost_protected | map($failures[.]) | min_by(.row)) as $row |
        {proposal:$bad.id, pair:(($failures|to_entries|map(select(.value.row==$row.row))|first).key), row:$row}
    end;
{
    pass:true, candidates:($moves|length), protected:$protected, free:$free,
    first_protected_reject:witness($violation),
    first_far_improving_protected_reject:witness($improving_violation)
}
