def require($condition; $message): if $condition then . else error($message) end;
def fields($rows; f): [$rows[] | f] | unique;
def disjoint($a; $b): (($a + $b) | unique | length) == (($a | unique | length) + ($b | unique | length));
def families($grammar): fields($grammar.wordings; .family);
def facts($grammar): fields($grammar.facts; .[0], .[1], .[2]);
def pairs($grammar): fields($grammar.wordings; [.authorize, .decline]);
def nouns($grammar): fields($grammar.wordings; .reference, .rival);
def templates($grammar):
  fields($grammar.wordings; .neutral, .near_before, .near_after, .far_before,
         .far_after, .heading, .order, .ending);
def whole_words: ascii_downcase | scan("[a-z]+");
def extract($text; $span): $text[$span.start:$span.end];
def bounds($span): $span.start >= 0 and $span.end > $span.start;

def fixture($records; $designs; $sizes; $history):
  INDEX($designs[].wordings[]; .family) as $wordings |
  INDEX($designs[].facts[]; .[0]) as $facts |
  ($records | length) == 256 and (fields($records; .id) | length) == 256 and
  (fields($records; .structural_prompt_sha256) | length) == 256 and
  (fields($records; .family) == ($designs | map(families(.)) | add | sort)) and
  ($records | group_by(.family) | all(.[]; length == $sizes.family)) and
  ($records | group_by(.distance) | map({key:.[0].distance,value:length}) | from_entries) == $sizes.distance and
  ($records | group_by([.family,.fact,.distance,.governor_order,.placement_name,
                         .requested_role,.admission]) | length) == 256 and
  ($records | group_by(.pair_id) | length == 128 and all(.[];
    map(.admission) == ["defer","take"] and
    .[0].native_scope == [.[1].native_scope[1], .[1].native_scope[0], .[1].native_scope[2]] and
    .[0].neutral_rival == .[1].neutral_rival and
    .[0].effective_grounded == .[1].effective_grounded)) and
  all($records[];
    .turns[1] as $user | $wordings[.family] as $wording | $facts[.fact] as $fact |
    .case_id == .id and .cohort == "fresh" and .fold == 8 and
    .turns[0] == $history.user and
    [.structural_history.prompt,.structural_history.ids,.structural_history.stop,.structural_history.text] ==
      [$history.prompt_ids,$history.reply_ids,$history.stop,$history.reply_text] and
    .first_prompt == $history.prompt_ids and .prompt == (.first_prompt + $history.reply_ids + .second_suffix) and
    [.fact_spans[0].value,.fact_spans[1].value] == $fact[1:3] and
    extract($user; .gold_binding.referenced) == $wording.reference and
    extract($user; .neutral_rival_bytes) == $wording.reference and
    ([extract($user; .gold_binding.affirmed),extract($user; .gold_binding.excluded)] | sort) ==
      ([$wording.reference,$wording.rival] | sort) and
    extract($user; .governors.affirmative.bytes) == $wording.authorize and
    extract($user; .governors.excluded.bytes) == $wording.decline and
    .governors.affirmative.noun == .gold_binding.affirmed and .governors.excluded.noun == .gold_binding.excluded and
    .governors.affirmative.native_noun == .native_scope[0].tokens and
    .governors.excluded.native_noun == .native_scope[1].tokens and
    ((.native_scope + [.neutral_rival]) | map(.bytes) | unique | length) == 4 and
    ((.native_scope + [.neutral_rival]) | map(.tokens) | unique | length) == 4 and
    all((.native_scope + [.neutral_rival])[]; bounds(.bytes) and bounds(.tokens)) and
    (.distance as $distance | all(.governors.affirmative, .governors.excluded;
      if $distance == "far" then .gap >= 8 and .visible_tokens == 0
      else .gap < 6 and .visible_tokens > 0 end)) and
    .requested_role == .gold_binding.first_role and .scope_tuple_count >= 4);

def describe($part):
  {endpoints:($part | length), pairs:(fields($part; .pair_id) | length),
   families:(fields($part; .family) | length),
   far_endpoints:([$part[] | select(.distance == "far")] | length),
   near_endpoints:([$part[] | select(.distance == "near")] | length),
   gap_by_distance:($part | group_by(.distance) | map({
     key:.[0].distance,
     value:([.[] | .governors.affirmative.gap, .governors.excluded.gap] | {min:min,max:max})
   }) | from_entries),
   proposer_tuples:([$part[].scope_tuple_count] | {min:min,max:max}),
   nominal_reply_ids:([$part[].effective_grounded.nominal.expected | length] | {min:min,max:max})};

. as $held | $training[0] as $train |
[$old_a[0], $old_b[0], $train_a[0], $train_b[0], $held_design[0]] as $designs |
($old_rows_a[0] + $old_rows_b[0]) as $old |
$legacy_words[0] as $teacher |
require(all($designs[]; (pairs(.) ==
  [["accept","reject"],["clear","block"],["follow","forget"],["keep","drop"]]));
  "the four pretrained signed governor pairs changed") |
require(($designs | map(families(.)) | add) as $all | ($all | length) == ($all | unique | length);
  "wording family overlap") |
require(($designs | map(facts(.)) | add) as $all | ($all | length) == ($all | unique | length);
  "fact value or fact ID overlap") |
require(($designs | map(templates(.)) | add) as $all | ($all | length) == ($all | unique | length);
  "authored template overlap") |
require(disjoint(fields($old; .id); fields($train + $held; .id)) and
        disjoint(fields($train; .id); fields($held; .id)); "case ID overlap") |
require(disjoint(fields($old; .structural_prompt_sha256); fields($train + $held; .structural_prompt_sha256)) and
        disjoint(fields($train; .structural_prompt_sha256); fields($held; .structural_prompt_sha256));
        "encoded prompt digest overlap") |
require(disjoint(fields($old; .prompt); fields($train + $held; .prompt)) and
        disjoint(fields($train; .prompt); fields($held; .prompt)); "encoded prompt overlap") |
require(disjoint(fields($old; .turns[1]); fields($train + $held; .turns[1])) and
        disjoint(fields($train; .turns[1]); fields($held; .turns[1])) and
        disjoint($teacher.users; fields($train + $held; .turns[1])); "authored prompt overlap") |
require((nouns($held_design[0]) | length) == 8 and
        disjoint(fields($old + $train; .turns[1] | whole_words) + $teacher.words; nouns($held_design[0]));
        "held target noun appears in new or historical user text") |
require((nouns($train_a[0]) | length) == 8 and
        fields($train_a[0].wordings; [.authorize,.decline,.reference,.rival]) ==
        fields($train_b[0].wordings; [.authorize,.decline,.rival,.reference]);
        "training noun spellings did not cross reference roles by signed verb pair") |
require(fixture($train; [$train_a[0],$train_b[0]]; {family:32,distance:{far:256}}; $history[0]);
        "far training reference does not match its authority") |
require(fixture($held; [$held_design[0]]; {family:64,distance:{far:128,near:128}}; $history[0]);
        "held reference does not match its authority") |
{training:describe($train), held:describe($held), historical_teacher_endpoints:($teacher.users | length),
 held_novel_target_nouns:8, trained_signed_governor_pairs:4, crossed_training_noun_spellings:8,
 overlap:{families:0,templates:0,facts:0,case_ids:0,prompt_digests:0,prompts:0,held_target_words:0},
 authentic_history:true, rank_used_for_enrollment:false}
