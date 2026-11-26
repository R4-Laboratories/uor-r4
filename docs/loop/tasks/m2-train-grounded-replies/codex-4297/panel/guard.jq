def require($ok; $why): if $ok then . else error($why) end;
def values($rows; f): [$rows[] | f] | unique;
def disjoint($x; $y): ($x - $y) == $x;
def words: ascii_downcase | scan("[a-z]+");
def family($g): $g.wordings[].family;
def noun($g): $g.wordings[] | .reference, .rival;
def facts($g): $g.facts[][];
def template($g): $g.wordings[] | .neutral, .near_before, .near_after, .far_before,
    .far_after, .heading, .order, .ending, (.contrast? | .before?, .after? | strings);
def framed_prompt: .prompt // .stock.prompt //
    ((.structural_history.prompt // .first_prompt) +
        (.structural_history.ids // $history[0].reply_ids) + .second_suffix);
def raw: .bytes |= (.start -= 1 | .end -= 1);
def span($s): .[$s.start:$s.end];
def unique_at($list; $item): ([$list[] | select(. == $item)] | length) == 1;
def range: {min:min,max:max};
def count: group_by(.) | map({key:.[0],value:length}) | from_entries;
def gap_relation:
  .governors.referent as $g | .neutral_rival.tokens as $n |
  ([$n.start-$g.tokens[1],$g.tokens[0]-$n.end,0] | max) as $gap |
  if $gap < $g.gap then "rival_nearer"
  elif $gap > $g.gap then "rival_farther" else "equal" end;

. as $rows | $design[0] as $grammar | ($earlier_rows | add) as $past |
($earlier_grammars | map(family(.)) | unique) as $past_families |
($earlier_grammars | map(template(.)) | unique) as $past_templates |
($earlier_grammars | map(facts(.)) | unique) as $past_facts |
require(length == $expected and (values($rows; .id)|length) == $expected
    and (values($rows; .structural_prompt_sha256)|length) == $expected
    and (values($rows; .turns[1])|length) == $expected
    and (values($rows; framed_prompt)|length) == $expected; "fresh row identities changed") |
require(($past|length) >= 3344 and all($past[];
    (.id|type) == "string" and (.turns[1]|type) == "string" and (framed_prompt|length) > 0)
    and ($teacher|length) == 512;
    "historical text/codec inventory is incomplete") |
require((values([$grammar]; family(.))|length) == 4 and
    disjoint(values([$grammar]; family(.)); $past_families + values($past + $teacher; .family));
    "wording family overlap") |
require((values([$grammar]; noun(.))|length) == 8 and
    disjoint(values([$grammar]; noun(.));
      values($past; .turns[] | words) + values($teacher; .user | words));
    "a proposed held surface may have appeared in historical text") |
require((values([$grammar]; facts(.))|length) == 6 and
    disjoint(values([$grammar]; facts(.));
        $past_facts + values($past; .fact, .fact_spans[]?.value) + values($teacher; .fact_values[]));
    "fact ID or value overlap") |
require(disjoint(values([$grammar]; template(.)); $past_templates);
    "authored template overlap") |
require(disjoint(values($rows; .id); values($past + $teacher; .id))
    and disjoint(values($rows; .turns[1]); values($past; .turns[]) + values($teacher; .user))
    and disjoint(values($rows; framed_prompt); values($past; framed_prompt))
    and disjoint(values($rows; .structural_prompt_sha256); values($past; .structural_prompt_sha256))
    and disjoint(values($rows; .second_suffix); values($past; .second_suffix));
    "held text, case ID, encoded turn or prompt digest overlaps an antecedent") |
require(($coordinates|length) == $expected and all(range(0;$expected); . as $index |
    $rows[$index] as $row | $coordinates[$index] as $blind |
    $row.native_scope as [$a,$b,$h] | $row.neutral_rival as $n |
    (if $row.admission == "take" then $b else $a end) as $outside |
    (if $row.admission == "take" then $a else $b end) as $owner |
    (" " + $row.turns[1]) as $text |
    ($blind.tuples | map(map($blind.sites[.]))) as $tuples |
    $row.structural_history ==
      {prompt:$history[0].prompt_ids,ids:$history[0].reply_ids,stop:"eos",text:$history[0].reply_text}
    and $row.prompt == ($history[0].prompt_ids + $history[0].reply_ids + $row.second_suffix)
    and $row.turns[0] == $history[0].user
    and $blind.ordinal == $index and $blind.id == $row.id
    and $blind.prompt_sha256 == $row.structural_prompt_sha256
    and ($blind.tuples|length) == $row.scope_tuple_count
    and $owner != $n and $owner.tokens != $n.tokens
    and ($text|span($owner.bytes)) == ($text|span($n.bytes))
    and ($text|span($owner.bytes)) == ($text|span($h.bytes))
    and ($text|span($outside.bytes)) != ($text|span($h.bytes))
    and all($row.governors.affirmative.native, $row.governors.excluded.native;
      unique_at($blind.words; raw))
    and all([$a,$b,$h],[$b,$a,$h],[$n,$outside,$h],[$outside,$n,$h];
      unique_at($tuples; map(raw))));
    "blind byte/BPE address, history, governor or registered tuple disagreement") |
{
  rows:$expected, pairs:(values($rows; .pair_id)|length), prior_reference_records:($past|length),
  prior_grammar_records:($earlier_grammars|length), prior_teacher_records:($teacher|length),
  overlaps:{families:0, target_words:0, templates:0, facts:0,
    case_ids:0, raw_prompts:0, encoded_turns:0, encoded_prompts:0, encoded_digests:0},
  signed_governor_addresses_verified:($expected*2), distinguished_original_tuples_verified:($expected*4),
  families:($rows|group_by(.family)|map({family:.[0].family,rows:length,
    strata:(group_by(.distance)|map({distance:.[0].distance,rows:length,
      pairs:(map(.pair_id)|unique|length),
      governor_order:(map(.governor_order)|count), admission:(map(.admission)|count),
      same_noun_rival_order:(map(.placement_name)|count),
      rival_gap_to_fixed_governor:(map(gap_relation)|count),
      requested_role:(map(.requested_role)|count), selector:(map(.contrast.selector // empty)|count),
      signed_gold_gaps:(map(.governors.affirmative.gap, .governors.excluded.gap)|range),
      signed_governor_visible_tokens:(map(.governors.affirmative.visible_tokens,
          .governors.excluded.visible_tokens)|range),
      eligible_tuples:(map(.scope_tuple_count)|range),
      reply_ids_through_eos:(map(.effective_grounded.nominal.expected|length)|range)}))}))
}
