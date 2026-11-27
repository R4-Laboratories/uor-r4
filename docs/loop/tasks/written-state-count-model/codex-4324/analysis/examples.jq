def class:
 .arms as $a |
 if $a.learned.exact then "learned_exact"
 elif $a.first.exact and $a.post_b.exact then "either_recovers"
 elif $a.first.exact then "first_recovers"
 elif $a.post_b.exact then "post_b_recovers"
 elif $a.both.exact then "needs_both"
 else "still_wrong" end;
def trace: {ids,wrong,reached_b,choices:[.choices[]|{stage,score:.action_score,address:.requested_pos,id:.committed_id,via:.selected_source,intervention:.release.intervention}]};
group_by([class, (.arms.learned.wrong),(.arms.first.wrong),(.arms.post_b.wrong),(.arms.both.wrong)])|
map({class:(.[0]|class),cases:length,example:(.[0]|{pair_id,side,family,context,question,directive,reply,expected,positions,arms:(.arms|with_entries(.value|=trace))})})
