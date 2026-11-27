def stage:
  {eligible,opposed,strict,actual,authenticated,actual_with_zero,inverted,both_stop,both_copy,
   wrong_or_equal_gap,gap_one,gap_sufficient_missed_threshold};

def row:
  {rows,exact,terminal,exact_short,exact_long,first_error,source_auth_b_required,
   source_auth_b_learned,right_after_learned_b,right_after_supplied_first,
   exact_after_supplied_first,ceiling};

def axis:
  {view,axis,contrast,models:(.models|map_values(
     {pairs,exact,ceiling,period:(.period|stage),after_supplied_first:(.after_supplied_first|stage)})),
   equal_length:(.equal_length|map_values(
     {pairs,exact,period:(.period|stage),after_supplied_first:(.after_supplied_first|stage)})),
   pointwise_to_paired:(.pointwise_to_paired|map(
     {metric,pointwise,paired,preserved,lost:(.lost_ids|length),gained:(.gained_ids|length)}))};

def historic:
  {split,rows:(.rows|map_values({rows,exact,ceiling})),pairs:(.original_pairs|map_values({pairs,exact})),
   full_quartets,retention:[.retention[]|select(.view=="original" and (.from=="balanced" or .from=="old_joint"))|
     {from,to,previous,current,retained,lost:(.previous-.retained),gained:(.current-.retained)}]};

$held[0] as $fresh |
{schema:"codex-4327.paired-release-grade-evidence/1",
 held:{
   rows:($fresh.rows|map_values(row)),
   rows_by_side:($fresh.rows_by_side|map(.models|=map_values(row))),
   full_quartets:$fresh.full_quartets,
   equal_length_quartets:$fresh.equal_length_quartets,
   pair_axes:($fresh.pair_views|map(axis)),
   family:($fresh.family | map(
     {family,author:.wording_author,rows:(.rows|map_values({rows,exact})),
      quartets:(.quartets|map_values({contexts,exact,both_selector_strict,both_selector_actual})),
      contrasts:[.pairs[]|{axis,contrast,models:(.models|map_values(
        {pairs,exact,period_strict:.period.strict,period_authenticated:.period.authenticated,
         post_b_strict:.after_supplied_first.strict,post_b_authenticated:.after_supplied_first.authenticated}))}]}))
 },
 historical:($all[0]|map(select(.split!="held")|historic))}
