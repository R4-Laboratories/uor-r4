def ids($arm): group_by(.pair_id)|map(select(all(.[];.arms[$arm].exact))|.[0].pair_id)|sort;
def area($split;$which):
 if $which=="direct" then map(select(.axis=="direct-count-control"))
 elif $which=="first-period" then map(select(.contrast==0))
 elif $which=="scoped" then map(select(.one!=null)) else . end;
. as $unused |
[["first-period","direct","scoped","protected"][] as $which |
 select((($split=="prior-dev") and $which=="first-period") or (($split=="dev" or $split=="held") and ($which=="direct" or $which=="scoped")) or ($split=="protect" and $which=="protected")) |
 ($balanced|area($split;$which)) as $b | ($candidate|area($split;$which)) as $c |
 ["balanced","candidate"][] as $from |
 (if $from=="balanced" then $b else $c end|ids("learned")) as $prev |
 ["balanced","candidate"][] as $to | (if $to=="balanced" then $b else $c end) as $r |
 ["learned","first","post_b","both"][] as $a |
 ($r|ids($a)) as $curr | ($prev-$curr) as $lost |
 {split:$split,cohort:$which,from:$from,to:$to,arm:$a,previous:($prev|length),retained:(($prev|length)-($lost|length)),lost_ids:$lost,gained_ids:($curr-$prev)}]
