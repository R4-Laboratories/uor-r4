def require($condition; $reason): if $condition then . else error($reason) end;
def values($records; f): [$records[] | f] | unique;
def disjoint($left; $right): ($left - $right | length) == ($left | length);
def families($grammar): values($grammar.wordings; .family);
def facts($grammar): values($grammar.facts; .[]);
def nouns($grammar): values($grammar.wordings; .reference, .rival);
def templates($grammar):
  values($grammar.wordings; .neutral, .near_before, .near_after,
    .far_before, .far_after, .heading, .order, .ending);
def words: ascii_downcase | scan("[a-z]+");

def match_key:
  [.distance_pair,.family,.fact,.admission,.governor_order,.placement_name,
   .requested_role,.requested_horizon,.role_mentioned_first,.gold_binding.pending];

def without_scopes:
  reduce .scope_clauses[] as $clause (.turns[1];
    split($clause) | select(length == 2) | join("[scope]"));

def partner($near; $far):
  ($near | without_scopes) as $outer |
  ($near | match_key) == ($far | match_key) and
  $near.distance == "near" and $far.distance == "far" and
  $far.cell == ($near.cell + 2) and $near.id != $far.id and
  $near.structural_prompt_sha256 != $far.structural_prompt_sha256 and
  ($near.fact_spans | map(.value)) == ($far.fact_spans | map(.value)) and
  $near.effective_grounded == $far.effective_grounded and
  ($far | without_scopes) == $outer;

def matches($mixed; $far):
  [range(128;256) as $index | $mixed[$index] as $near | $far[$index] as $distant |
   require(partner($near; $distant); "unmatched B row " + ($index | tostring)) |
   {ordinal:$index, distance_pair:$near.distance_pair, family:$near.family,
    fact:$near.fact, direction:$near.governor_order, neutral:$near.placement_name,
    admission:$near.admission, first_role:$near.requested_role,
    near:{pair_id:$near.pair_id,id:$near.id}, far:{pair_id:$distant.pair_id,id:$distant.id}}]
  | select(length == 128);

def paired($rows): $rows | group_by(.pair_id) |
  length == 128 and all(.[]; map(.admission) == ["defer","take"]);

def describe($rows):
  {endpoints:($rows | length), pairs:(values($rows; .pair_id) | length),
   families:(values($rows; .family) | length),
   by_distance:($rows | group_by(.distance) | map({
     key:.[0].distance,
     value:{endpoints:length, pairs:(map(.pair_id)|unique|length),
       gaps:([.[] | .governors.affirmative.gap, .governors.excluded.gap] | {min:min,max:max}),
       tuples:([.[].scope_tuple_count] | {min:min,max:max}),
       nominal_reply_ids:([.[].effective_grounded.nominal.expected | length] | {min:min,max:max})}
   }) | from_entries)};

def fresh_rows($left; $right):
  require(disjoint(values($left; .id); values($right; .id)); "case ID overlap") |
  require(disjoint(values($left; .structural_prompt_sha256);
                   values($right; .structural_prompt_sha256)); "encoded prompt digest overlap") |
  require(disjoint(values($left; .prompt); values($right; .prompt)); "encoded prompt overlap") |
  require(disjoint(values($left; .turns[1]); values($right; .turns[1])); "authored prompt overlap");

. as $held | $mixed_rows[0] as $mixed | $far_rows[0] as $far |
[$a[0],$b[0],$held_design[0]] as $designs |
$prior_designs[0] as $past_designs | ($prior_rows[0] | add) as $past_rows |
([$mixed[], $far[128:][]]) as $train |
require(all($designs[]; values(.wordings; [.authorize,.decline]) ==
  [["accept","reject"],["clear","block"],["follow","forget"],["keep","drop"]]);
  "pretrained governor vocabulary changed") |
require(($designs | map(families(.)) | add) as $all |
        ($all | length) == ($all | unique | length) and
        disjoint($all; ($past_designs | map(families(.)) | add));
        "wording family overlap") |
require(($designs | map(facts(.)) | add) as $all |
        ($all | length) == ($all | unique | length) and
        disjoint($all; ($past_designs | map(facts(.)) | add));
        "fact value or ID overlap") |
require(($designs | map(templates(.)) | add) as $all |
        ($all | length) == ($all | unique | length) and
        disjoint($all; ($past_designs | map(templates(.)) | add));
        "authored template overlap") |
fresh_rows($held; $train) |
fresh_rows($held + $train; $past_rows) |
require(disjoint(values($held + $train; .turns[1]); $teacher[0].users);
        "authored prompt overlaps historical teacher") |
require((nouns($held_design[0]) | length) == 8 and
        disjoint(nouns($held_design[0]);
          (values($past_rows + $train; .turns[1] | words) + $teacher[0].words));
        "held target noun appears in new training or historical user text") |
require(all($mixed,$far,$held; length == 256 and paired(.) and
  (values(.; .id) | length) == 256 and (values(.; .structural_prompt_sha256) | length) == 256);
  "a split has missing, duplicated, or inconsistent pairs") |
require((values($train; .id) | length) == 384 and
        (values($train; .structural_prompt_sha256) | length) == 384 and
        ($mixed[0:128] == $far[0:128]) and
        all($mixed[0:128][]; .distance == "far") and
        (values($mixed[0:128]; .family) == families($a[0])) and
        (values($mixed[128:]; .family) == families($b[0])) and
        ([ $held[] | select(.distance == "near")] | length) == 128;
        "shared first block or split membership changed") |
matches($mixed; $far) as $matches |
require((values($matches; .distance_pair) | length) == 128 and
        (values($matches; .near.pair_id) | length) == 64 and
        (values($matches; .far.pair_id) | length) == 64 and
        ($matches | group_by([.near.pair_id,.far.pair_id]) |
         length == 64 and all(.[]; map(.admission) == ["defer","take"]));
        "B pair mapping is not one-to-one") |
([$mixed[] | select(.family == "practice-class" and .distance == "near" and
                    .governor_order == "before")]) as $short |
require(($short | length) == 16 and all($short[];
  all(.governors.affirmative,.governors.excluded; .gap == 1 and .visible_tokens > 0));
  "short after-warmup TRAIN branch changed its transport visibility") |
([$held[] | select(.family == "pottery-sorting" and .distance == "far" and
                   .governor_order == "after")]) as $aside |
require(($aside | length) == 16 and all($aside[];
  all(.scope_clauses[]; test(" you might,.*glaze.*kiln")) and
  all(.governors.affirmative,.governors.excluded; .gap >= 8 and .visible_tokens == 0));
  "held finite-you branch lost the post-target noun-bearing aside") |
{mixed:describe($mixed), far:describe($far), held:describe($held),
 common_far_pairs:64, matched_b_pairs:64, short_analogue_endpoints:($short | length),
 held_post_target_aside_endpoints:($aside | length), held_novel_target_nouns:8,
 historical_teacher_endpoints:($teacher[0].users | length),
 overlap:{families:0, templates:0, facts:0, case_ids:0, prompt_digests:0, prompts:0, held_target_words:0},
 rank_used_for_enrollment:false, matches:$matches}
