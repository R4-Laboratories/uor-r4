def require($condition; $reason): if $condition then . else error($reason) end;

def token_difference($wanted; $read):
  ([$wanted,$read] | map(length) | max) as $length |
  first(range(0;$length) | select($wanted[.] != $read[.]) |
    {position:(.+1),expected_id:$wanted[.],actual_id:$read[.]}) // null;

def check_lane($target; $lane):
  token_difference($target.ids;$lane.ids) as $delta |
  $lane.first_wrong as $reported |
  ($lane.ids == $target.ids and $lane.stop == $target.stop) as $payload |
  ($lane.stop == "max_tokens" and ($lane.ids | length) == 48 and
   $lane.ids == $target.ids[:48]) as $cap |
  require(
    if $delta != null then
      $delta == ($reported | {position,expected_id,actual_id}) and
      $reported.index == $delta.position-1 and $reported.reason != "stop"
    elif $payload then $reported == null
    else $reported == {index:($lane.ids|length),position:(($lane.ids|length)+1),
                       expected_id:null,actual_id:null,reason:"stop"}
    end; "first differing position or ID disagrees with raw token streams") |
  require($lane.payload_equal == $payload and $lane.correct_prefix_at_cap == $cap and
    $lane.exact_through_eos == ($payload and $lane.stop == "eos" and $lane.ids[-1] == 1);
    "terminal or cap classification disagrees with raw token streams") |
  {
    first_differing_id:$delta,
    stop_only_difference:($delta == null and ($payload | not)),
    correct_prefix_at_cap:$cap,
    exact_through_eos:$lane.exact_through_eos
  };

require((.target.ids | length) <= 48; "target exceeds the enrolled 48-ID cap") |
. as $row |
check_lane(.target;.first) as $first |
check_lane(.target;.returned) as $returned |
require(($returned.correct_prefix_at_cap | not) or
  ($row.returned_text == $row.target.text and ($row.returned_text | length) > 0);
  "cap-matched IDs do not give the enrolled visible text") |
{id,arm,pair_id,admission,distance,first:$first,returned:$returned}
