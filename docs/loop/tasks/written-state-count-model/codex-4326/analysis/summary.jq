def n(p): map(select(p)) | length;
def count_by(f): group_by(f) | map({key:(.[0] | f | tostring), value:length}) | from_entries;
def model_names: ["zero", "period_only", "post_b_only", "full"];

def reduction($samples):
  {n:($samples | length), min:($samples | min), max:($samples | max),
   median:($samples | sort | .[(length / 2 | floor)]),
   negative:($samples | n(. < 0)), zero:($samples | n(. == 0)), positive:($samples | n(. > 0))};

def additivity($stage):
  map(.additivity[$stage] | select(. != null)) as $aligned |
  {rows:($aligned | length), nonadditive:($aligned | n(.residual != 0)),
   residual:reduction([$aligned[].residual]),
   score_ranges:(reduce model_names[] as $m ({}; .[$m] = reduction([$aligned[].scores[$m]])))};

def stage_rows:
  map(select(. != null)) as $at |
  {rows:($at | length), right_id:($at | n(.right_id)),
   native_zero:($at | n(.choice == "native")), right_native_zero:($at | n(.right_id and .choice == "native")),
   authenticated_copy:($at | n(.copy_authenticated)),
   outcomes:($at | count_by([.expected_action, .actual_action]))};

def model_rows($all; $model):
  [$all[].models[$model]] as $rows |
  {exact:($rows | n(.reply.exact)), capped:($rows | n(.reply.terminal | not)),
   first_error:($rows | count_by(.reply.first_error.token)),
   learned_authenticated_b_when_expected:($rows | n(.after_b != null and .learned_b_authenticated)),
   period:([$rows[].period] | stage_rows),
   after_authenticated_b:([$rows[].after_b] | stage_rows),
   forced_first_exact:($rows | n(.after_forced_first.exact)),
   ceiling:($rows | n(.ceiling.exact))};

def row_summary:
  . as $rows |
  {count:length, models:(reduce model_names[] as $m ({}; .[$m] = model_rows($rows; $m))),
   additivity:{period:additivity("period"), after_b:additivity("after_b")}};

def stage_pairs($stage):
  map(.[$stage] | select(. != null)) as $at |
  ($at | map(select(.strict_directed != null))) as $opposed |
  {eligible:($at | length), opposite_targets:($at | n(.strict_directed != null)),
   strict_directed:($at | n(.strict_directed)), native_tie_pairs:($at | n(.ties > 0)),
   native_tie_members:([$at[].ties] | add // 0), actual_matches:($at | n(.actual_matches)),
   actual_authenticated:($at | n(.actual_authenticated)), actual_with_native_zero:($at | n(.actual_with_native_zero)),
   actual_actions:($at | count_by(.actual_actions)), gap:reduction([$opposed[].copy_minus_stop]),
   wrongly_ordered_or_equal:($opposed | n(.copy_minus_stop <= 0)),
   gap_one:($opposed | n(.copy_minus_stop == 1)),
   separable_but_strict_miss:($opposed | n(.copy_minus_stop >= 2 and (.strict_directed | not)))};

def split_transitions($from):
  . as $pairs |
  reduce ["exact", "period_strict", "period_actual", "period_authenticated", "post_b_strict", "post_b_actual", "post_b_authenticated"][] as $field ({};
    .[$field] = ([$pairs[][$from][$field] | select(. != null)] as $items |
      reduce ["preserved", "erased", "gained", "never"][] as $k ({}; .[$k] = ($items | n(. == $k)))));

def pair_summary:
  . as $pairs |
  {count:length,
   models:(reduce model_names[] as $m ({};
     .[$m] = ([$pairs[].models[$m]] | {exact:n(.exact), period:stage_pairs("period"), after_authenticated_b:stage_pairs("after_b")}))),
   from_period_only:split_transitions("from_period_only"), from_post_b_only:split_transitions("from_post_b_only"),
   changes_to_period_gap:(map(.models | select(.full.period.copy_minus_stop != null)) as $m |
     {period_component:reduction([$m[] | .period_only.period.copy_minus_stop - .zero.period.copy_minus_stop]),
      post_b_component:reduction([$m[] | .post_b_only.period.copy_minus_stop - .zero.period.copy_minus_stop])}),
   changes_to_post_b_gap:(map(.models | select(.full.after_b.copy_minus_stop != null)) as $m |
     {period_component:reduction([$m[] | .period_only.after_b.copy_minus_stop - .zero.after_b.copy_minus_stop]),
      post_b_component:reduction([$m[] | .post_b_only.after_b.copy_minus_stop - .zero.after_b.copy_minus_stop])})};

def crossed_contexts:
  map(select(.selector_swap_id != null)) | group_by(.context_id) | map(
    if length == 4 and (map([.selector, .one_word_clause]) | sort) == [[1,1],[1,2],[2,1],[2,2]] then .
    else error("incomplete crossed context: " + .[0].context_id) end);

def context_stats:
  . as $groups |
  {count:length, exact:(reduce model_names[] as $m ({}; .[$m] = ($groups | n(all(.[]; .models[$m].reply.exact)))))};

def reference_pair($by_id): all(.members[]; $by_id[[.pair_id, .side] | @json].learned.exact);
def pair_ids: map({axis, pair_id, contrast});
def retention_counts($by_id; $include_ids):
  . as $pairs |
  map(select(reference_pair($by_id))) as $prior |
  {previous:($prior | length), models:(reduce model_names[] as $m ({};
     .[$m] = (($prior | map(select(.models[$m].exact))) as $kept |
              ($prior | map(select(.models[$m].exact | not))) as $lost |
              ($pairs | map(select((reference_pair($by_id) | not) and .models[$m].exact))) as $gained |
       {kept:($kept | length), lost:($lost | length), gained:($gained | length)} +
       (if $include_ids then {kept_ids:($kept | pair_ids), lost_ids:($lost | pair_ids), gained_ids:($gained | pair_ids)} else {} end))))} +
  (if $include_ids then {previous_ids:($prior | pair_ids)} else {} end);

def retention($baseline):
  ($baseline | INDEX(.[]; [.pair_id, .side] | @json)) as $by_id |
  . as $pairs |
  if all($pairs[].members[]; . as $row | $by_id[[.pair_id, .side] | @json] | . != null and .expected == $row.expected and .positions == $row.positions and .ids == $row.source_ids)
  then . else error("historical comparison membership or reference mismatch") end |
  retention_counts($by_id; true) +
  {groups:(group_by([.axis, .contrast, .cohort]) | map(.[0] as $k |
      {axis:$k.axis, contrast:$k.contrast, cohort:$k.cohort} + retention_counts($by_id; false)))};

($rows | crossed_contexts) as $crossed |
{split:$split, rows:($rows | row_summary),
 row_groups:($rows | group_by([(.one_word_clause != null), .contrast, .cohort]) | map(.[0] as $k |
   {scope:($k.one_word_clause != null), contrast:$k.contrast, cohort:$k.cohort} + row_summary)),
 contexts:($crossed | context_stats),
 equal_length_contexts:($crossed | map(select((map(.prefix_length) | unique | length) == 1)) | context_stats),
 groups:($pairs | group_by([.axis, .contrast, .cohort]) | map(.[0] as $k | {axis:$k.axis, contrast:$k.contrast, cohort:$k.cohort} + pair_summary)),
 by_family:($pairs | group_by([.axis, .contrast, .cohort, .family]) | map(.[0] as $k |
   {axis:$k.axis, contrast:$k.contrast, cohort:$k.cohort, family:$k.family} + pair_summary)),
 previous_balanced:($pairs | retention($balanced)), previous_linked:($pairs | retention($linked)),
 examples:($pairs | group_by([.axis, .contrast, .cohort, .from_period_only.period_strict, .from_post_b_only.post_b_strict]) |
   map(.[0] + {class_n:length}))}
