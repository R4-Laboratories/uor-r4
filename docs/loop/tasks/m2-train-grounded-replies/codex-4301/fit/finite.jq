{
  original_blocks_both_sides: (
    [$terms[] | select(.candidate == "original")]
    | length == 4 and all(.[]; .margin == -1 and .joint_margin == -1 and .pair_both_win == false)
  ),
  learned_both_selectors_both_signed_banks: (
    [$terms[] | select(.candidate == $id)]
    | length == 4 and all(.[]; .margin > 0 and .joint_margin > 0 and .pair_both_win)
  ),
  exchanged_roles_fail: (
    [$wrong[] | select(.candidate == "original")] as $bad
    | [$terms[] | select(.candidate == $id)] as $good
    | ($bad | length) == 4 and all(range(0; 4);
      . as $i
      | $bad[$i].margin == (-$good[$i].margin)
        and $bad[$i].joint_margin == (-$good[$i].joint_margin)
        and $bad[$i].pair_both_win == false
    )
  ),
  real_learned_update: any($schedule[];
    .phase == "fit" and .candidate == $id and .accepted
    and .before.both_pair_wins == [0, 0] and .after.both_pair_wins == [1, 1]
    and .after.changed >= 2
  ),
  bounded: (
    $complete[0]
    | .rows == 4 and .pairs == 2 and .state_bytes == 13336 and .initial == 57
      and .proposal_evaluations <= 2649 and .exported <= 105
  )
}
| select(all(.[]; . == true))
