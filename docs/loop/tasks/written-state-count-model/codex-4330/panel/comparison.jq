def held($arm; $split): $arm.cohort_summary[] | select(.cohort == "squad" and .split == $split);
def protected($arm):
  $arm.protected_c021_c024 |
  map({case_id,source_correct,source_step_position,source_step_id,copied_first_supported,exact_eos,first_wrong});
def phase($held; $delay): $held.matched_source_phases[] | select(.source_delay == $delay);
def cell($arm; $split):
  held($arm; $split) as $held |
  {
    arm:$arm.arm,
    immediate:phase($held; 0),
    after_space:phase($held; 1),
    positive_source_immediate:$held.correct_source_immediate,
    positive_source_after_space:$held.correct_source_after_space,
    positive_reached_after_space:$held.positives_after_space_reached,
    all_unsupported_copies:$held.unsupported_copies,
    all_unsupported_claims:$held.unsupported_claims,
    wrong_same_id:$held.wrong_same_id,
    exact_eos:$held.exact_eos,
    reference_within_cap:$held.reference_within_cap,
    reference_over_cap:$held.reference_over_cap
  };

$old[0] as $base | $frozen[0] as $freeze | $fitted[0] as $fit |
cell($base; "reserve") as $base_reserve |
cell($freeze; "reserve") as $freeze_reserve |
cell($fit; "reserve") as $fit_reserve |
protected($fit) as $protected |
{
  schema:"uor-r4.whole-turn-fit-ordinary-comparison/1",
  reserve:[$base_reserve,$freeze_reserve,$fit_reserve],
  published_dev:[cell($base; "dev"),cell($freeze; "dev"),cell($fit; "dev")],
  protected_c021_c024:[{arm:$base.arm,rows:protected($base)},
    {arm:$freeze.arm,rows:protected($freeze)},{arm:$fit.arm,rows:$protected}],
  source_gate_parts:{
    reserve_pair_win_increase:($fit_reserve.immediate.support_flip_wins - $base_reserve.immediate.support_flip_wins),
    reserve_false_copy_reduction:($freeze_reserve.immediate.unsupported_copy_pairs - $fit_reserve.immediate.unsupported_copy_pairs),
    prior_duplicate_ids_preserved:(
      ($protected|map(.case_id)) == ["protected-c021","protected-c022","protected-c023","protected-c024"]
      and all($protected[]; .source_correct and .copied_first_supported)
    )
  }
}
