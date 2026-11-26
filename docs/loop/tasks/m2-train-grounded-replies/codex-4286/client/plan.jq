include "common";

assert_ok(length > 0 and (map(.id) | length == (unique | length));
  "panel enrollment is empty or duplicated") |
assert_ok(($device | length) == length and ($device | map(.case_id) | length == (unique | length));
  "device enrollment is missing or duplicated") |
INDEX($device[]; .case_id) as $endpoints |
.[] as $r |
$endpoints[$r.id] as $rank |
assert_ok(($r.id | test("^[0-9a-f]{64}$")) and $r.fold == 8 and
  $rank.case_id == $r.id and $rank.pair_id == $r.admission_pair and
  $rank.family == $r.family and $rank.admission == $r.admission and
  $rank.distance == $r.distance and $rank.prompt_sha256 == $r.structural_prompt_sha256 and
  $rank.gold == ([$r.gold_binding.affirmed,$r.gold_binding.excluded,$r.gold_binding.referenced]) and
  ($rank.gold_tokens | length) == 3 and $rank.gold_tokens == ($r.native_scope | map(.tokens)) and
  ($rank.arms | keys) == (arms | sort);
  "source enrollment differs for " + $r.id) |
assert_ok(($r.turns | length) == 2 and
  ($r.turns[1] | utf8bytelength == length) and
  $r.prompt == ($r.first_prompt + $r.structural_history.ids + $r.second_suffix) and
  ($r.structural_prompt_sha256 | test("^[0-9a-f]{64}$")) and
  $r.structural_history.prompt == $r.first_prompt and $r.structural_history.stop == "eos" and
  ($r.gold_binding.first_role | IN("home","job")) and
  ($r.gold_binding.pending | IN("stop","continue"));
  "second turn is not bound to registered original bytes and generated history for " + $r.id) |
(arms[]) as $arm |
$rank.arms[$arm] as $candidate |
assert_ok(
  ($candidate.selection == null and $candidate.tokens == null) or
  (($candidate.selection | length) == 3 and
   all($candidate.selection[]; span_ok($r.turns[1] | utf8bytelength)) and
   ($candidate.tokens | length) == 3 and
   all($candidate.tokens[]; span_ok($r.prompt | length)));
  "invalid winner addresses for " + $r.id + "/" + $arm) |
{
  id: $r.id, arm: $arm, fold: $r.fold, selection: $candidate.selection,
  wire: ({prompt_sha256:$r.structural_prompt_sha256,
          first_role:$r.gold_binding.first_role,pending:$r.gold_binding.pending} +
    (if $arm == "native" then {learned_field:"scope"}
     else ($candidate.selection // [span_empty,span_empty,span_empty]) as $spans |
       {affirmed:$spans[0],excluded:$spans[1],referenced:$spans[2]} end))
}
