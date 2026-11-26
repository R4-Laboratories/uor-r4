def count($v): $v|length;
. as $v | .all as $a | .distance.near as $n | .distance.far as $f | $election[0] as $e |
($e.trajectory[$e.selections.retained]) as $t |
($f.far_only.exact-$f.baseline.exact) as $fresh_far_gains |
{pairs:128,reference_sha256:$ref_sha,endpoints_sha256:$writer_sha,
 original_near:$n.baseline.exact, lost_near:($n.baseline.exact-$n.retained.exact),
 far_only_gains:$fresh_far_gains, far_only_gains_lost:($fresh_far_gains-$f.retained.exact),
 far_gain_retention_informative:(count($fresh_far_gains)>0),
 gate:{
  positive_training_retention_and_far_gain:(count($e.original_near)>0 and $t.feasible and count($t.gained_common_far)>0),
  retains_real_held_near_wins:(count($n.baseline.exact)>0 and count($n.baseline.exact-$n.retained.exact)==0),
  retains_matched_far_gains:(count($fresh_far_gains-$f.retained.exact)==0),
  improves_far_over_parent:(count($f.retained.exact)>count($f.baseline.exact)),
  total_gain:all(["baseline","native","unfitted"][];. as $arm|count($a.retained.exact)-count($a[$arm].exact)>=8),
  gains_both_distances:all($v.distance[];. as $d|all(["native","unfitted"][];. as $arm|count($d.retained.exact)>count($d[$arm].exact))),
  full_links_both_distances:all($v.distance[];count(.retained.full)>0),
  no_family_below_native:all($v.family[];count(.retained.exact)>=count(.native.exact)),
  supplied_controls:(count($a.correct_link.earlier)==128 and $a.exchanged_link.prompts==0 and
    count($a.correct_link.exact)-count($a.exchanged_link.exact)>=8 and
    all($v.distance[];count(.correct_link.exact)>count(.exchanged_link.exact)))}
 } | .pass=all(.gate[];.)
