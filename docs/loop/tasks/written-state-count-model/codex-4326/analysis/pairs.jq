def require(ok; why): if ok then . else error(why) end;
def outcome($earlier; $later):
  if $earlier == null or $later == null then null
  elif $earlier then (if $later then "preserved" else "erased" end)
  elif $later then "gained" else "never" end;

def matched_stage($members; $model; $stage):
  [$members[].models[$model][$stage]] as $at |
  if any($at[]; . == null) then null else
    ($at | map(.expected_action)) as $target |
    ($at | map(.choice)) as $choices |
    ($at | map(.actual_action)) as $actual |
    ($target == ["stop", "copy"]) as $opposed |
    ($at | map(.score)) as $s |
    ($at | map(select(.choice == "native")) | length) as $ties |
    ($actual == $target and all($at[]; .right_id)) as $right |
    {target:$target, native_policy:$choices, actual_actions:$actual, scores:$s,
     actual_matches:$right, actual_with_native_zero:($right and $ties > 0),
     actual_authenticated:($right and all($at[]; .expected_action == "stop" or .copy_authenticated)),
     observed_ids:($at | map(.committed_id)), selected_sources:($at | map(.selected_source)), ties:$ties,
     strict_directed:(if $opposed then $s[0] < 0 and $s[1] > 0 else null end),
     copy_minus_stop:(if $opposed then $s[1] - $s[0] else null end),
     sum_of_margins:(if $opposed then $s[1] + $s[0] else null end)}
  end;

def transitions($models; $from):
  $models[$from] as $a | $models.full as $b |
  {exact:outcome($a.exact; $b.exact),
   period_strict:outcome($a.period.strict_directed; $b.period.strict_directed),
   period_actual:outcome($a.period.actual_matches; $b.period.actual_matches),
   period_authenticated:outcome($a.period.actual_authenticated; $b.period.actual_authenticated),
   post_b_strict:outcome($a.after_b.strict_directed; $b.after_b.strict_directed),
   post_b_actual:outcome($a.after_b.actual_matches; $b.after_b.actual_matches),
   post_b_authenticated:outcome($a.after_b.actual_authenticated; $b.after_b.actual_authenticated)};

def pair($split; $axis; $id):
  sort_by(.expected | length) as $rows |
  $rows as [$short, $long] |
  require(length == 2 and $short.family == $long.family and $short.context == $long.context
          and $short.question == $long.question and $short.source_ids == $long.source_ids
          and $short.positions == $long.positions and $short.cohort == $long.cohort
          and $short.contrast == $long.contrast and ($short.expected | length) == ($short.contrast + 3)
          and ($long.expected | length) == ($long.contrast + 4)
          and $short.expected[:-1] == $long.expected[:-2]; "mispaired identity: " + $id) |
  require((if $axis == "selector-swap" then
             $short.one_word_clause == $long.one_word_clause and $short.one_word_clause != null
             and $short.selector != $long.selector
           elif $axis == "clause-order-swap" then
             $short.selector == $long.selector and $short.selector != null
             and $short.one_word_clause != $long.one_word_clause
           else $short.one_word_clause == null and $long.one_word_clause == null end);
          "crossed key does not support " + $axis + ": " + $id) |
  (reduce ["zero", "period_only", "post_b_only", "full"][] as $model ({};
    .[$model] = {exact:all($rows[]; .models[$model].reply.exact),
                 period:matched_stage($rows; $model; "period"),
                 after_b:matched_stage($rows; $model; "after_b")})) as $models |
  {split:$split, axis:$axis, pair_id:$id, context_id:$short.context_id, family:$short.family,
   cohort:$short.cohort, contrast:$short.contrast, same_token_length:($short.prefix_length == $long.prefix_length),
   members:[$rows[] | {pair_id, side, selector, one_word_clause, directive, reply, prefix_length,
      expected, positions, source_ids,
      outcomes:(.models | map_values({reply, after_forced_first, learned_b_authenticated})), additivity}],
   models:$models, from_period_only:transitions($models; "period_only"), from_post_b_only:transitions($models; "post_b_only")};

. as $rows |
($rows | group_by(.pair_id)[] | pair($split; .[0].axis; .[0].pair_id)),
($rows | map(select(.selector_swap_id != null)) | group_by(.selector_swap_id)[] |
  pair($split; "selector-swap"; .[0].selector_swap_id))
