def require($condition; $reason): if $condition then . else error($reason) end;
def values($rows; f): [$rows[] | f] | unique;
def disjoint($left; $right): ($left - $right | length) == ($left | length);
def families($grammar): values($grammar.wordings; .family);
def facts($grammar): values($grammar.facts; .[]);
def nouns($grammar): values($grammar.wordings; .reference, .rival);
def templates($grammar):
  values($grammar.wordings; .neutral, .near_before, .near_after, .far_before,
    .far_after, .heading, .order, .ending);
def words: ascii_downcase | scan("[a-z]+");
def prompt: .prompt // (.structural_history.prompt + .structural_history.ids + .second_suffix);
def span($s): .[$s.start:$s.end];
def raw_address: .bytes |= (.start -= 1 | .end -= 1);
def present_once($list; $item): ([$list[] | select(. == $item)] | length) == 1;

. as $rows | ($past_rows | add) as $past | $design[0] as $grammar |
require(($past | length) == 2560 and all($past[];
  (.id|type) == "string" and (.structural_prompt_sha256|type) == "string" and
  (.turns[1]|type) == "string" and (prompt | length) > 0);
  "prior reference inventory changed") |
require(($teacher | length) == 512 and all($teacher[]; (.user|type) == "string");
  "prior teacher inventory changed") |
require(($past_designs | length) == 8 and all($past_designs[];
  (.wordings|length) == 4 and (.facts|length) == 2);
  "prior grammar inventory changed") |
require((values($grammar.wordings; [.authorize,.decline])) ==
  [["accept","reject"],["clear","block"],["follow","forget"],["keep","drop"]];
  "governor vocabulary changed") |
require((families($grammar) | length) == 4 and
  disjoint(families($grammar); ($past_designs | map(families(.)) | add) +
    values($past + $teacher; .family)); "wording family overlap") |
require((templates($grammar)|length) == 32 and
  disjoint(templates($grammar); ($past_designs | map(templates(.)) | add));
  "authored template overlap") |
require((facts($grammar)|length) == 6 and
  disjoint(facts($grammar); ($past_designs | map(facts(.)) | add) +
    values($past; .fact, .fact_spans[].value) + values($teacher; .fact_values[]));
  "fact value or ID overlap") |
require((nouns($grammar)|length) == 8 and disjoint(nouns($grammar);
  values($past; .turns[1] | words) + values($teacher; .user | words));
  "held target noun appears in historical user text") |
require(($rows|length) == 256 and (values($rows; .id)|length) == 256 and
  disjoint(values($rows; .id); values($past + $teacher; .id)); "case ID overlap") |
require((values($rows; .turns[1])|length) == 256 and
  disjoint(values($rows; .turns[1]); values($past; .turns[1]) + values($teacher; .user));
  "raw prompt overlap") |
require((values($rows; .structural_prompt_sha256)|length) == 256 and
  disjoint(values($rows; .structural_prompt_sha256); values($past; .structural_prompt_sha256));
  "encoded prompt digest overlap") |
require((values($rows; prompt)|length) == 256 and
  disjoint(values($rows; prompt); values($past; prompt)); "encoded prompt overlap") |
require(($rows | group_by(.distance) | map({key:.[0].distance,value:(
   group_by(.pair_id) | select(all(.[]; map(.admission) == ["defer","take"])) | length)})
   | from_entries) == {near:64,far:64}; "opposed pair counts changed") |
require(all($rows[]; . as $row | $row.structural_history ==
  {prompt:$history[0].prompt_ids,ids:$history[0].reply_ids,stop:"eos",text:$history[0].reply_text}
  and $row.prompt == ($history[0].prompt_ids + $history[0].reply_ids + $row.second_suffix)
  and $row.turns[0] == $history[0].user); "authenticated history changed") |
require(($coordinates|length) == 256 and all(range(0;256); . as $index |
  $rows[$index] as $row | $coordinates[$index] as $blind |
  $row.native_scope as [$a,$b,$h] | $row.neutral_rival as $n |
  (if $row.admission == "take" then $b else $a end) as $outside |
  (if $row.admission == "take" then $a else $b end) as $owner |
  (" " + $row.turns[1]) as $text |
  ($blind.tuples | map(map($blind.sites[.]))) as $tuples |
  $blind.ordinal == $index and $blind.id == $row.id and
  $blind.prompt_sha256 == $row.structural_prompt_sha256 and
  ($blind.tuples|length) == $row.scope_tuple_count and
  $owner != $n and $owner.tokens != $n.tokens and
  ($text|span($owner.bytes)) == ($text|span($n.bytes)) and
  ($text|span($owner.bytes)) == ($text|span($h.bytes)) and
  ($text|span($outside.bytes)) != ($text|span($h.bytes)) and
  all($row.governors.affirmative.native, $row.governors.excluded.native;
    present_once($blind.words; raw_address)) and
  all([$a,$b,$h],[$b,$a,$h],[$n,$outside,$h],[$outside,$n,$h];
    present_once($tuples; map(raw_address))));
  "blind byte/BPE address, governor word or tuple disagreement") |
([$rows[] | select(.family == "boatyard-rigging" and .distance == "far" and
                   .governor_order == "after")]) as $finite |
require(($finite|length) == 16 and all($finite[]; . as $row |
  all(.governors.referent, .governors.rival; . as $governor |
    $governor.gap >= 8 and $governor.visible_tokens == 0 and
    $row.turns[1][$governor.noun.end:$governor.bytes.start] == " after the tar and the twine run low our rigger will "));
  "finite post-nominal clause or its BPE gap changed") |
{
  held_endpoints:($rows|length), held_pairs:{near:64,far:64}, wording_families:4,
  prior_reference_records:($past|length), historical_teacher_records:($teacher|length),
  historical_grammars:($past_designs|length), novel_target_nouns:8,
  finite_post_nominal_endpoints:($finite|length), finite_post_nominal_pairs:($finite|map(.pair_id)|unique|length),
  blind_verified_endpoints:($coordinates|length), rank_used_for_enrollment:false,
  overlap:{families:0, templates:0, facts:0, case_ids:0, raw_prompts:0, encoded_prompts:0, encoded_digests:0}
}
