def key: [.pair_id, .side] | @json;
def require(ok; why): if ok then . else error(why) end;
def indexed($items):
  ($items | INDEX(.[]; key)) as $by_id |
  require(($by_id | length) == ($items | length); "duplicate pair member") | $by_id;
def decision($id): if $id == 1 then "stop" else "copy" end;

def trajectory($expected):
  . as $t | [.steps[].committed_id] as $ids |
  ([range(0; ([$ids, $expected] | map(length) | max)) as $i |
    select($ids[$i] != $expected[$i]) | $i][0]) as $wrong |
  require($t.first_mismatch == (if $wrong == null then null else $wrong + 1 end)
          and $t.terminal == ($ids[-1] == 1) and ($ids[:-1] | index(1)) == null;
          "contradictory termination or first mismatch") |
  {ids:$ids, terminal:$t.terminal, exact:($t.terminal and $wrong == null),
   first_error:(if $wrong == null then null else
     {token:($wrong + 1), expected:$expected[$wrong], observed:$ids[$wrong]} end)};

def copied($row; $i):
  .selected_source == "verified_copy" and .refusal == 0 and .release.source_authenticated == true
  and .requested_pos == $row.positions[$i] and .requested_id == $row.ids[$i]
  and .committed_id == $row.ids[$i];

def boundary($row; $i):
  (if $i == 1 then "first" else "post_b" end) as $name |
  require(.stage == ($i - 1) and (.action_score | type) == "number"
          and .release.eligible == $name and .release.intervention == null
          and .release.original_pos == $row.positions[$i] and .release.original_id == $row.ids[$i];
          "unaligned learned " + $name + " score") |
  (if .action_score < 0 then "eos" elif .action_score > 0 then "copy" else "native" end) as $policy |
  require(.release.learned_action == $policy and .release.effective_action == $policy;
          "score and native decision disagree") |
  [$row.expected[$i + 1], $row.ids[$i]] as [$target, $original] |
  {score:.action_score, choice:$policy, expected_action:decision($target),
   actual_action:(if .committed_id == 1 then "stop"
                  elif .committed_id == $original then "copy" else "other" end),
   right_id:(.committed_id == $target), copy_authenticated:copied($row; $i),
   committed_id, selected_source, refusal, original_pos:.release.original_pos, original_id:$original};

def model:
  . as $row | .arms as $arms |
  require(($arms | keys) == ["both", "first", "learned", "post_b"]; "invalid release interventions") |
  ($arms.learned | trajectory($row.expected)) as $reply |
  ($arms.first | trajectory($row.expected)) as $forced |
  ($arms.both | trajectory($row.expected)) as $ceiling |
  ($arms.post_b | trajectory($row.expected)) as $forced_post |
  ($row.expected | length > 3) as $b_required |
  require($ceiling.exact and $arms.both.steps[2].release.intervention == "first"
          and (if $b_required then ($arms.both.steps[2] | copied($row; 1))
                and $arms.both.steps[3].release.intervention == "post_b" else true end)
          and (if ($row.expected | length) == 5 then ($arms.both.steps[3] | copied($row; 2)) else true end);
          "supplied decisions failed authenticated EOS ceiling") |
  require(all($arms[]; [.steps[0].committed_id,.steps[1].committed_id] == $row.expected[:2]
          and .steps[0].copied == true and .steps[0].claim_pos == $row.positions[0]
          and .steps[0].read_pos == $row.positions[0] and .steps[0].source_id == $row.ids[0]
          and .steps[1].selected_source == "supplied_period"); "opening differs") |
  (if $b_required then
     require($forced.ids[:3] == $row.expected[:3] and ($arms.first.steps[2] | copied($row; 1))
             and $arms.first.steps[2].release.intervention == "first"
             and $arms.first.steps[2].release.next_stage == 1; "forcing first missed authenticated B") |
     ($arms.first.steps[3] | boundary($row; 2))
   else require($forced.exact; "forcing first failed short reply") | null end) as $after_b |
  ($b_required and ($arms.learned.steps[2] | copied($row; 1))) as $b_learned |
  {reply:$reply, after_forced_first:$forced, after_forced_b:$forced_post, ceiling:$ceiling,
   period:($arms.learned.steps[2] | boundary($row; 1)), after_b:$after_b,
   learned_b_authenticated:$b_learned,
   learned_after_b:(if $b_learned then ($arms.learned.steps[3] | boundary($row; 2)) else null end)};

($panel | indexed(.)) as $p | ($keys | indexed(.)) as $k |
({balanced:$balanced, linked:$linked, pointwise:$pointwise, paired:$paired}
 + (if ($old_joint | length) > 0 then {old_joint:$old_joint} else {} end)
 | map_values(indexed(.))) as $model_rows |
($p | keys) as $ids |
require(($ids | length) > 0 and all($k, $model_rows[]; keys == $ids); "panel, keys or GPU membership mismatch") |
$ids[] as $id | $p[$id] as $row | $k[$id] as $key |
($row | [.pair_id, .side, .contrast, .family, .prefix, .positions, .ids, .expected]) as $identity |
require(all($model_rows[][$id]; [.pair_id, .side, .contrast, .family, .prefix, .positions, .ids, .expected] == $identity)
    and $key.contrast == $row.contrast and ($key.family == null or $key.family == $row.family)
    and ($key.row_id == null or $key.row_id == ($row.pair_id + ":" + ($row.side | tostring)));
    "GPU or key identity mismatch: " + $id) |
require($row.expected == ([$row.ids[0],16] + $row.ids[1:($row.side + $row.contrast + 1)] + [1])
        and ($key.expected_word_count == null or $key.expected_word_count == ($row.expected | length) - 2)
        and ($key.expected_first_period_action == null or $key.expected_first_period_action == decision($row.expected[2]))
        and ($key.expected_post_b_action == null or $key.expected_post_b_action == decision($row.expected[3]));
        "invalid reference reply or decision key: " + $id) |
{pair_id:$row.pair_id, side:$row.side, contrast:$row.contrast, family:$row.family,
 context_id:$key.context_id, selector_swap_id:$key.selector_swap_id,
 selector:$key.operative_clause_ordinal, one_word_clause:$key.one_word_clause_ordinal,
 axis:($key.pair_axis // "direct-count-control"), cohort:($key.cohort // "historical"),
 wording_author:$key.wording_author, context:$key.context, question:$key.question,
 directive:$key.directive, reply:$key.reply,
 prefix_length:($row.prefix | length), positions:$row.positions, source_ids:$row.ids, expected:$row.expected,
 models:($model_rows | map_values(.[$id] | model))}
