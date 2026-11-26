def count($s): $s|length;
.distance.near as $n | .distance.far as $f | .all as $a | . as $v |
($f.far_only.exact-$f.baseline.exact) as $gain |
{pairs:128,reference_sha256:$ref_sha,endpoints_sha256:$endpoint_sha,
 original_near:$n.baseline.exact,lost_near:($n.baseline.exact-$n.protected.exact),
 matched_far_gains:$gain,matched_far_gains_lost:($gain-$f.protected.exact),
 matched_far_gain_informative:(count($gain)>0),
 gate:{
  train:(($train[0].protected.lost_protected|length)==0 and ($train[0].protected.far_gained|length)>0 and
         $train[0].protected.hinge<$train[0].parent.hinge and count($train[0].parent.near_full)==24),
  retains_held_near:(count($n.baseline.exact)>0 and count($n.baseline.exact-$n.protected.exact)==0),
  retains_matched_far_gains:(count($gain-$f.protected.exact)==0),
  improves_far:(count($f.protected.exact)>count($f.baseline.exact)),
  total_gain:all(["baseline","native","unfitted"][];. as $k|count($a.protected.exact)-count($a[$k].exact)>=8),
  gains_both_distances:all($v.distance[];. as $d|all(["native","unfitted"][];. as $k|count($d.protected.exact)>count($d[$k].exact))),
  full_links_both_distances:all($v.distance[];count(.protected.full)>0),
  no_family_below_native:all($v.family[];count(.protected.exact)>=count(.native.exact)),
  supplied_controls:(count($a.correct_link.earlier)==128 and $a.exchanged_link.prompts==0 and
    count($a.correct_link.exact)-count($a.exchanged_link.exact)>=8 and
    all($v.distance[];count(.correct_link.exact)>count(.exchanged_link.exact)))
 }} | .pass=all(.gate[];.)
