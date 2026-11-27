def require(ok; why): if ok then . else error(why) end;
def n(expr): map(select(expr)) | length;
def history: group_by(.) | map({value:.[0], rows:length});

def row_stats($model):
  . as $subset | map(.models[$model]) as $m |
  {rows:length, exact:($m | n(.reply.exact)), terminal:($m | n(.reply.terminal)),
   exact_short:($subset | n(.side == 0 and .models[$model].reply.exact)),
   exact_long:($subset | n(.side == 1 and .models[$model].reply.exact)),
   first_error:($m | map(.reply.first_error.token) | history),
   first_error_ids:($m | map(.reply.first_error | select(. != null)) | history),
   source_auth_b_required:($m | n(.after_b != null)),
   source_auth_b_learned:($m | n(.learned_b_authenticated)),
   right_after_learned_b:($m | n(.learned_after_b.right_id)),
   right_after_supplied_first:($m | n(.after_b.right_id)),
   exact_after_supplied_first:($m | n(.after_forced_first.exact)),
   ceiling:($m | n(.ceiling.exact))};

def stage($model; $gate):
  map(.models[$model][$gate] | select(. != null)) as $all |
  ($all | map(select(.strict_directed != null))) as $contrasts |
  {eligible:($all | length), opposed:($contrasts | length),
   strict:($contrasts | n(.strict_directed)),
   actual:($all | n(.actual_matches)), authenticated:($all | n(.actual_authenticated)),
   actual_with_zero:($all | n(.actual_with_native_zero)),
   inverted:($contrasts | n(.actual_actions == ["copy","stop"])),
   both_stop:($contrasts | n(.actual_actions == ["stop","stop"])),
   both_copy:($contrasts | n(.actual_actions == ["copy","copy"])),
   wrong_or_equal_gap:($contrasts | n(.copy_minus_stop <= 0)),
   gap_one:($contrasts | n(.copy_minus_stop == 1)),
   gap_sufficient_missed_threshold:($contrasts | n(.copy_minus_stop >= 2 and (.strict_directed|not)))};

def pair_stats($model):
  {pairs:length, exact:n(.models[$model].exact), ceiling:n(.models[$model].ceiling),
   period:stage($model; "period"), after_supplied_first:stage($model; "after_b")};

def quartet($model):
  {contexts:length,
   exact:n(all(.[];.models[$model].exact)),
   ceiling:n(all(.[];.models[$model].ceiling)),
   both_selector_strict:n(all(.[]|select(.axis=="selector-swap"); .models[$model].period.strict_directed)),
   both_selector_actual:n(all(.[]|select(.axis=="selector-swap"); .models[$model].period.actual_authenticated)),
   both_clause_strict:n(all(.[]|select(.axis=="clause-order-swap"); .models[$model].period.strict_directed)),
   both_clause_actual:n(all(.[]|select(.axis=="clause-order-swap"); .models[$model].period.actual_authenticated))};

def exact_ids($model):
  [ .[] | select(.models[$model].exact) | .pair_id ] | sort;

def retention($from; $to; $kind):
  (exact_ids($from)) as $had | (exact_ids($to)) as $has |
  {from:$from,to:$to,view:$kind,previous:($had|length),current:($has|length),
   retained:(($had-($had-$has))|length),lost_ids:($had-$has),gained_ids:($has-$had),
   retained_ids:($had-($had-$has))};

def movement($metric; quantity):
  (map(select(.models.pointwise | quantity) | .pair_id) | sort) as $had |
  (map(select(.models.paired | quantity) | .pair_id) | sort) as $has |
  {metric:$metric, pointwise:($had|length), paired:($has|length),
   preserved:(($had-($had-$has))|length), lost_ids:($had-$has), gained_ids:($has-$had)};

def into_models($names; $sub; $kind):
  $names | map(. as $model | {key:$model,value:($sub |
    if $kind == "rows" then row_stats($model)
    elif $kind == "pairs" then pair_stats($model)
    else quartet($model) end)}) | from_entries;

($rows[0].models | keys) as $names |
($pairs | map(select(.view == "original"))) as $original |
($pairs | map(select(.view == "selector-derived"))) as $selector |
($selector | map(.context_id) | unique) as $full_ids |
($full_ids | map({key:.,value:true}) | from_entries) as $full |
($pairs | map(select(.context_id as $id | $full[$id // ""] == true)) | group_by(.context_id)) as $quartets |
require(all($quartets[]; length == 4 and (map(.axis) | sort) ==
    ["clause-order-swap","clause-order-swap","selector-swap","selector-swap"]);
    "incomplete crossed context") |
require($split != "protect" or (($original | length) == 50 and ($original | exact_ids("balanced") | length) == 50);
    "original 50 exact balanced protection pairs changed") |
{split:$split,
 rows:into_models($names;$rows;"rows"),
 rows_by_side:($rows | group_by(.side) |
   map(. as $set | {side:.[0].side,models:into_models($names;$set;"rows")})),
 original_pairs:into_models($names;$original;"pairs"),
 derived_selector_pairs:into_models($names;$selector;"pairs"),
 pair_views:($pairs | group_by([.view,.axis,.contrast]) |
   map(. as $set | .[0] as $key |
     {view:$key.view,axis:$key.axis,contrast:$key.contrast,
      models:into_models($names;$set;"pairs"),
      equal_length:into_models($names;($set|map(select(.same_token_length)));"pairs"),
      pointwise_to_paired:[$set |
         movement("exact"; .exact),
         movement("period_strict"; .period.strict_directed),
         movement("period_authenticated"; .period.actual_authenticated),
         movement("post_b_strict"; .after_b.strict_directed),
         movement("post_b_authenticated"; .after_b.actual_authenticated)]})),
 full_quartets:into_models($names;$quartets;"quartets"),
 equal_length_quartets:into_models($names;($quartets|map(select(all(.[];.same_token_length))));"quartets"),
 cohorts:($rows | group_by(.cohort) | map(. as $set | .[0].cohort as $id |
   {cohort:$id,rows:into_models($names;$set;"rows"),
    original_pairs:into_models($names;($original|map(select(.cohort==$id)));"pairs")})),
 family:($rows | group_by(.family) | map(. as $family | .[0] as $key |
   {family:$key.family,wording_author:$key.wording_author,
    rows:into_models($names;$family;"rows"),
    pairs:($pairs | map(select(.family == $key.family)) | group_by([.view,.axis,.contrast]) |
      map(. as $set | .[0] as $view |
        {view:$view.view,axis:$view.axis,contrast:$view.contrast,
         models:into_models($names;$set;"pairs")})),
    quartets:into_models($names;($quartets|map(select(.[0].family==$key.family)));"quartets")})),
 wording_authors:($rows | map(select(.wording_author != null)) | group_by(.wording_author) |
   map(. as $author | .[0].wording_author as $id |
    {author:$id,rows:into_models($names;$author;"rows"),
     quartets:into_models($names;($quartets|map(select(.[0].wording_author==$id)));"quartets")})),
 retention:([
   $names[] | select(. == "balanced" or . == "linked" or . == "old_joint" or . == "pointwise")
   | . as $from | ["pointwise","paired"][] as $to | select($from != $to)
   | ($original | retention($from;$to;"original")), ($selector | retention($from;$to;"selector-derived"))
 ]),
 examples:($rows |
   map(select(.models.pointwise.reply.first_error != .models.paired.reply.first_error)) |
   group_by([.wording_author,.axis,.models.pointwise.reply.first_error.token,.models.paired.reply.first_error.token]) |
   map(.[0] | del(.models.balanced,.models.linked,.models.old_joint) |
    .models |= map_values({reply,period,after_b})))}
