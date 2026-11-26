. as $s | .arms as $a | .by_distance as $d |
{pairs:.pairs,endpoints:.endpoints,reference_sha256:$reference_sha,endpoints_sha256:$writer_sha,
  all_source_endpoints_sha256:$source_sha,arms:$a,by_distance:$d,by_family:.by_family,
  gate:{
    genuine_training_loss_flip:($training[0].eligible_losses>0 and $training[0].loss_to_win>0),
    gain_over_inherited:($a.learned.exact_pairs-$a.baseline.exact_pairs>=8),
    far_gain_over_inherited:($d.far.learned.exact_pairs>$d.far.baseline.exact_pairs),
    near_retention:($d.near.learned.exact_pairs>=$d.near.baseline.exact_pairs),
    net_over_native_and_unfitted:all(["native","unfitted"][]; . as $b|$a.learned.exact_pairs-$a[$b].exact_pairs>=8),
    both_distances_over_native_and_unfitted:all($d[];.learned.exact_pairs>.native.exact_pairs and .learned.exact_pairs>.unfitted.exact_pairs),
    full_links_at_both_distances:all($d[];.learned.full_pairs>0),
    no_family_below_native:all(.by_family[];.learned.exact_pairs>=.native.exact_pairs),
    supplied_controls:($a.correct_link.earlier_pairs==128 and $a.exchanged_link.exact_rows==0 and
      $a.correct_link.exact_pairs-$a.exchanged_link.exact_pairs>=8 and all($d[];.correct_link.exact_pairs>.exchanged_link.exact_pairs))
  }} | .pass=all(.gate[];.)
