def require($condition; $message): if $condition then . else error($message) end;
def distinct($train; $held):
  ($train | unique) as $known | all($held[]; . as $item | ($known | index($item)) == null);
def rows($records; field): [$records[] | field] | unique;
def pairs($design): [$design.wordings[] | [.authorize,.decline]] | unique;
def families($design): [$design.wordings[].family] | unique;
def nouns($design): [$design.wordings[] | .reference,.rival] | map(ascii_downcase) | unique;
def facts($design): [$design.facts[] | .[1],.[2]] | map(ascii_downcase) | unique;
def templates($design):
  [$design.wordings[] | .neutral,.near_before,.near_after,.far_before,.far_after,.heading,.order,.ending]
  | map(ascii_downcase) | unique;
def slice($text; $span): $text[$span.start:$span.end];
def words($records): [$records[].turns[1] | ascii_downcase | scan("[a-z]+")] | unique;

def enrollment($records; $design; $history):
  INDEX($design.wordings[]; .family) as $wordings |
  INDEX($design.facts[]; .[0]) as $facts |
  ($records | length) == 256 and
  (rows($records; .family) == families($design)) and
  (($records | group_by([.family,.fact,.distance,.governor_order,.placement_name,.requested_role,.admission]) | length) == 256) and
  all($records[];
    .turns[1] as $user | $wordings[.family] as $word | $facts[.fact] as $fact |
    [.structural_history.prompt,.structural_history.ids,.structural_history.stop,.structural_history.text]
      == [$history.prompt_ids,$history.reply_ids,$history.stop,$history.reply_text] and
    .turns[0] == $history.user and
    .first_prompt == $history.prompt_ids and
    .prompt == (.first_prompt + $history.reply_ids + .second_suffix) and
    .fact_spans[0].value == $fact[1] and .fact_spans[1].value == $fact[2] and
    slice($user; .gold_binding.referenced) == $word.reference and
    slice($user; .neutral_rival_bytes) == $word.reference and
    ([slice($user; .gold_binding.affirmed),slice($user; .gold_binding.excluded)] | sort)
      == ([$word.reference,$word.rival] | sort) and
    slice($user; .governors.affirmative.bytes) == $word.authorize and
    slice($user; .governors.excluded.bytes) == $word.decline and
    .governors.affirmative.noun == .gold_binding.affirmed and
    .governors.excluded.noun == .gold_binding.excluded and
    .governors.affirmative.native_noun == .native_scope[0].tokens and
    .governors.excluded.native_noun == .native_scope[1].tokens and
    .neutral_rival.bytes != .native_scope[0].bytes and .neutral_rival.bytes != .native_scope[1].bytes and
    .neutral_rival.tokens != .native_scope[0].tokens and .neutral_rival.tokens != .native_scope[1].tokens and
    .requested_role == .gold_binding.first_role and .scope_tuple_count >= 4
  );

. as $held_rows | $train[0] as $train_rows |
$training_grammar[0] as $training | $held_grammar[0] as $held |
require(distinct(families($training); families($held)); "wording family overlap") |
require(distinct(templates($training); templates($held)); "authored template overlap") |
require(distinct(words($train_rows); nouns($held)); "held noun appears in training text") |
require(distinct(facts($training); facts($held)); "fact value overlap") |
require(distinct(rows($train_rows; .id); rows($held_rows; .id)); "case ID overlap") |
require(distinct(rows($train_rows; .structural_prompt_sha256);
                 rows($held_rows; .structural_prompt_sha256)); "encoded prompt digest overlap") |
require(distinct(rows($train_rows; .turns[1]); rows($held_rows; .turns[1])); "authored prompt overlap") |
require(distinct(rows($train_rows; .prompt); rows($held_rows; .prompt)); "encoded prompt overlap") |
require(pairs($training) == pairs($held) and (pairs($held) | length) == 4;
        "expected four previously trained signed governor pairs") |
require(enrollment($train_rows; $training; $history[0]); "training reference does not match authority") |
require(enrollment($held_rows; $held; $history[0]); "held reference does not match authority") |
{
  training_endpoints:($train_rows | length),
  held_endpoints:($held_rows | length),
  held_opposed_pairs:(rows($held_rows; .pair_id) | length),
  held_fully_crossed_cells:($held_rows | length),
  matched_signed_governor_pairs:(pairs($held) | length),
  held_target_nouns:(nouns($held) | length),
  held_authored_templates:(templates($held) | length),
  held_fact_values:(facts($held) | length),
  overlap:{families:0,templates:0,held_nouns_in_training_text:0,fact_values:0,case_ids:0,
           authored_prompts:0,encoded_prompts:0,encoded_prompt_digests:0},
  first_turn_history:"exact historical generation in both splits"
}
