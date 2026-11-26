# The established three-noun exact gate also requires full witnessed pairs at each distance.
. as $v | .all as $a | .distance.near as $n | .distance.far as $f | $training[0] as $t |
($f.far_only.exact-$f.baseline.exact) as $fg |
{arm:$arm,pairs:128,reference_sha256:$ref_sha,endpoints_sha256:$ep_sha,
 protected_near_lost:$t.final.lost_protected,held_near_lost:($n.baseline.exact-$n[$arm].exact),
 matched_far_gains:$fg,matched_far_gains_lost:($fg-$f[$arm].exact),matched_far_informative:(($fg|length)>0),
 gate:{
   retained_train_sources:($t.qualified and ($t.final.lost_protected|length)==0),
   train_reduced_loss:($t.objective_after<$t.objective_before),
   train_new_far_pair:(($t.final.far_gained|length)>0),
   retains_held_near:(($n.baseline.exact|length)>0 and ($n.baseline.exact-$n[$arm].exact|length)==0),
   retains_matched_far_gains:(($fg-$f[$arm].exact|length)==0),
   improves_far:(($f[$arm].exact|length)>($f.baseline.exact|length)),
   total_gain:all(["baseline","native","unfitted"][];. as $k|($a[$arm].exact|length)-($a[$k].exact|length)>=8),
   gains_both_distances:all($v.distance[];. as $d|all(["native","unfitted"][];. as $k|($d[$arm].exact|length)>($d[$k].exact|length))),
   full_links_both_distances:all($v.distance[];(.[$arm].full|length)>0),
   no_family_below_native:all($v.family[];(.[$arm].exact|length)>=(.native.exact|length)),
   supplied_controls:(($a.correct_link.earlier|length)==128 and $a.exchanged_link.prompts==0 and
       ($a.correct_link.exact|length)-($a.exchanged_link.exact|length)>=8 and
       all($v.distance[];(.correct_link.exact|length)>(.exchanged_link.exact|length)))
 }} | .pass=all(.gate[];.)
