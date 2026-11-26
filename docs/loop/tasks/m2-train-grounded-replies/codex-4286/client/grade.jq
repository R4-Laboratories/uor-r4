include "common";
include "raw";
include "score";

def with_position:
  .first_wrong |= if . == null then . else . + {position:(.index + 1)} end;

def native_winner($bound;$expected):
  $bound.selections[0] as $native |
  ($bound.selections | length) == 3 and
  all($bound.selections[1:][]; .status == "supplied") and
  (if $expected.selection == null then
     ($native.status | IN("tied","no_witness")) and $bound.scope == null
   else
     $native.status == "chosen" and
     [$native.witnesses[].bytes] == ($expected.selection | map(framed)) and
     [$native.witnesses[].tokens] == $expected.tokens and
     $native.distance == $expected.bank_cost
   end);

def disposition($arm;$expected;$r;$t;$stock):
  $t.role_action_trace as $trace |
  $trace.bound_instruction as $bound |
  assert_ok($bound.first_role == $r.gold_binding.first_role and
    $bound.pending == $r.gold_binding.pending and
    $bound.field_origins == [(if $arm == "native" then "learned" else "supplied" end),"supplied","supplied"] and
    $bound.origin == (if $arm == "native" then "hybrid" else "supplied" end);
    "ordinary scope replay changed role, pending or origins") |
  (if $arm == "native" then native_winner($bound;$expected)
   elif $expected.selection == null then
     $bound.status == "invalid_binding" and
     $bound.scope == {affirmed:(span_empty | framed),excluded:(span_empty | framed),referenced:(span_empty | framed)}
   else
     [$bound.scope.affirmed,$bound.scope.excluded,$bound.scope.referenced] == ($expected.selection | map(framed))
   end) as $selection_matches |
  (if $bound.binding == null then null
   else [$bound.binding.affirmed,$bound.binding.excluded,$bound.binding.referenced] |
        {bytes:map(.bytes),tokens:map(.tokens),polarity:$bound.binding.polarity}
   end) as $binding |
  (if $binding == null then null
   else $binding.bytes == ($expected.selection | map(framed)) and $binding.tokens == $expected.tokens end) as $binding_matches |
  ($trace.fallback and $trace.steps == [] and $bound.initial_route == null and
   $bound.release_reached == false and $bound.released_route == null and $bound.verified_first_copy == null) as $zero |
  ($zero and ($t | same_reply($stock)) and
   $t.question_reader_trace == $stock.question_reader_trace) as $stock_parity |
  {
    status:$bound.status,selection_matches:$selection_matches,
    accepted_binding_matches:$binding_matches,accepted_binding:$binding,
    source_rejected:($expected.selection != null and $binding == null and
      ($bound.status | IN("invalid_binding","ambiguous_binding"))),
    zero_bound_emission:$zero,stock_parity:$stock_parity,
    valid_abstention:($expected.selection == null and $selection_matches and $binding == null and $stock_parity),
    initial_route:$bound.initial_route,released_route:$bound.released_route,release_status:$bound.release_status,
    original_first_copy:$bound.original_first_copy,verified_first_copy:$bound.verified_first_copy,
    released_copy:$bound.released_copy
  };

INDEX($reference[0][]; .id) as $refs |
INDEX($device[]; .case_id) as $endpoints |
INDEX($stock[]; .id) as $signed |
INDEX($codec[0].model.vocab | to_entries[]; .value | tostring) | map_values(.key) as $pieces |
inputs as $observed |
$refs[$observed.id] as $r |
$endpoints[$observed.id] as $row |
$signed[$observed.id] as $stock_capture |
$row.arms[$arm] as $expected |
assert_ok($r != null and $expected != null and $stock_capture.exit == 0 and $observed.exit == 0 and
    ($stock_capture.record | valid_run($r;$stock_model;$codec_hash)) and
    ($observed.record | valid_run($r;$model;$codec_hash));
    "can't grade mismatched ordinary artifact, prompts, generated history or process for " + $observed.id + "/" + $arm) |
($stock_capture.record.turns[1]) as $stock_turn |
($observed.record.turns[1]) as $turn |
($r.first_prompt + $r.structural_history.ids | length) as $base |
assert_ok($r.user_content == null or
  ($r.user_content.start == $turn.role_action_trace.user_start and
   $r.user_content.end == $turn.role_action_trace.user_end);
  "ordinary current-user address differs from the frozen panel") |
lanes($observed.record.turns[0]) as $earlier |
lanes($stock_turn) as $stock_lanes |
lanes($turn) as $actual |
assert_ok($earlier.returned.ids == $r.structural_history.ids and
  $stock_lanes.returned.ids == $stock_turn.reply_ids and $stock_lanes.mode == "saved";
  "stock and study must have genuine matching history") |
binding_addresses($r;$turn.role_action_trace;$turn.prompt_ids;$pieces) as $decoded_scope |
disposition($arm;$expected;$r;$turn;$stock_turn) as $bound |
(if $r.admission == "take" then grounded_target($r;$arm)
 else {expected:$stock_turn.reply_ids,expected_stop:$stock_turn.stop,expected_text:$stock_turn.reply_text}
 end) as $target |
assert_ok($target != null and
   ($r.admission != "take" or (($target.units | length) >= 3 and $target.expected[-1] == 1));
   "missing original grounded target for " + $r.id) |
score($target;$actual.first;$turn.prompt_ids;$base;true) | with_position as $first |
score($target;$actual.returned;$turn.prompt_ids;$base;true) | with_position as $returned |
assert_ok(($returned.payload_equal | not) or $turn.reply_text == $target.expected_text;
   "returned IDs and stop match while visible target text differs for " + $r.id + "/" + $arm) |
assert_ok(($turn | same_payload($stock_turn) | not) or $turn.reply_text == $stock_turn.reply_text;
   "returned stock IDs and stop match while visible stock text differs for " + $r.id + "/" + $arm) |
{
  id:$r.id,pair_id:$r.admission_pair,family:$r.family,distance:$row.distance,arm:$arm,admission:$r.admission,
  target:{ids:$target.expected,stop:$target.expected_stop,text:$target.expected_text},
  stock:{ids:$stock_turn.reply_ids,stop:$stock_turn.stop,text:$stock_turn.reply_text},
  first:($first + {correct_prefix_at_cap:($first | correct_prefix_at_cap($target.expected))}),
  returned:($returned + {correct_prefix_at_cap:($returned | correct_prefix_at_cap($target.expected))}),
  returned_text:$turn.reply_text,
  first_turn_retry:($earlier.retry != null),retry:($actual.retry != null),retry_steps:($actual.retry.steps // 0),
  requested_role:$r.gold_binding.first_role,pending:$r.gold_binding.pending,
  actual_role:opening_role($turn.reply_ids),decoded_scope:$decoded_scope,disposition:$bound,
  source_identity_ok:($bound.selection_matches and $bound.accepted_binding_matches != false and
    ($expected.selection != null or $bound.valid_abstention)),
  gold_scope_exact:($expected.selection == $row.gold and $bound.accepted_binding_matches == true),
  correct_route:($bound.initial_route == (if $r.gold_binding.first_role == "home" then 0 else 2 end)),
  authenticated_take_eos:($r.admission == "take" and $bound.status == "take" and
    $returned.original_source_exact_through_eos and $bound.accepted_binding_matches == true),
  authenticated_defer_parity:($r.admission == "defer" and $bound.status == "defer" and $bound.stock_parity and
    $bound.accepted_binding_matches == true),
  authenticated_defer_eos:($r.admission == "defer" and $bound.status == "defer" and $bound.stock_parity and
    $bound.accepted_binding_matches == true and $returned.exact_through_eos),
  exact_terminal:($returned.exact_through_eos)
}
