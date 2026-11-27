def require(ok; why): if ok then . else error(why) end;
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


def pair($axis; $id):
  sort_by(.expected | length) as $rows |
  $rows as [$short, $long] |
  require(length == 2 and $short.family == $long.family and $short.context == $long.context
          and $short.question == $long.question and $short.source_ids == $long.source_ids
          and $short.positions == $long.positions and $short.cohort == $long.cohort
          and $short.wording_author == $long.wording_author and $short.context_id == $long.context_id
          and $short.contrast == $long.contrast and ($short.expected | length) == ($short.contrast + 3)
          and ($long.expected | length) == ($short.contrast + 4)
          and $short.expected[:-1] == $long.expected[:-2]; "mispaired identity: " + $id) |
  require((if $axis == "selector-swap" then
             $short.one_word_clause == $long.one_word_clause and $short.one_word_clause != null
             and $short.selector != $long.selector
           elif $axis == "clause-order-swap" then
             $short.selector == $long.selector and $short.selector != null
             and $short.one_word_clause != $long.one_word_clause
           else $short.one_word_clause == null and $long.one_word_clause == null end);
          "crossed key does not support " + $axis + ": " + $id) |
  ($short.models | keys) as $names |
  (reduce $names[] as $model ({};
    .[$model] = {exact:all($rows[]; .models[$model].reply.exact),
                 ceiling:all($rows[]; .models[$model].ceiling.exact),
                 period:matched_stage($rows; $model; "period"),
                 after_b:matched_stage($rows; $model; "after_b")})) as $models |
  {axis:$axis, pair_id:$id, context_id:$short.context_id, family:$short.family,
   cohort:$short.cohort, contrast:$short.contrast, wording_author:$short.wording_author,
   same_token_length:($short.prefix_length == $long.prefix_length),
   members:[$rows[] | {pair_id, side, selector, one_word_clause, prefix_length}],
   models:$models};

. as $rows |
($rows | group_by(.pair_id)[] | pair(.[0].axis; .[0].pair_id) | .view = "original"),
($rows | map(select(.selector_swap_id != null)) | group_by(.selector_swap_id)[] |
  pair("selector-swap"; .[0].selector_swap_id) | .view = "selector-derived")
