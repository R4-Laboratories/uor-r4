def key: [.pair_id, .side] | @json;
def require(ok; why): if ok then . else error(why) end;
def indexed($items):
  ($items | INDEX(.[]; key)) as $by_id |
  require(($by_id | length) == ($items | length); "duplicate pair member") | $by_id;
def decision($id): if $id == 1 then "stop" else "copy" end;

def trajectory($expected):
  . as $t | [ .steps[].committed_id ] as $ids |
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
   committed_id, selected_source, refusal, original_pos:.release.original_pos, original_id:$original,
   causal_signature:[.gate_id, .native_proposal, .read_code, .recurrence_code, .causal_successor, .read_vector]};

def model:
  . as $row | .arms as $arms |
  require(($arms | keys) == ["both", "first", "learned", "post_b"]; "invalid release interventions") |
  ($arms.learned | trajectory($row.expected)) as $reply |
  ($arms.first | trajectory($row.expected)) as $forced |
  ($arms.both | trajectory($row.expected)) as $ceiling |
  ($arms.post_b | trajectory($row.expected)) as $forced_post |
  require($ceiling.exact; "supplied decisions failed ceiling") |
  require($reply.ids[:2] == $row.expected[:2] and $forced.ids[:2] == $row.expected[:2]; "opening differs") |
  (if ($row.expected | length) > 3 then
     require($forced.ids[:3] == $row.expected[:3] and ($arms.first.steps[2] | copied($row; 1))
             and $arms.first.steps[2].release.intervention == "first"
             and $arms.first.steps[2].release.next_stage == 1; "forcing first missed authenticated B") |
     ($arms.first.steps[3] | boundary($row; 2))
   else require($forced.exact; "forcing first failed short reply") | null end) as $after_b |
  {reply:$reply, after_forced_first:$forced, after_forced_b:$forced_post, ceiling:$ceiling,
   period:($arms.learned.steps[2] | boundary($row; 1)), after_b:$after_b,
   learned_b_authenticated:($arms.learned.steps[2] | copied($row; 1)),
   learned_post_b_score:$arms.learned.steps[3].action_score};

# All statistics use the raw i32 margin before the native zero branch. The kernel
# adds signed i16 words into the frozen i32 score, with no score clamp.
def sum_identity($models; $stage):
  [$models[][$stage]] as $at |
  if all($at[]; . == null) then null else
    require(all($at[]; . != null) and ($at | map([.causal_signature, .original_pos, .original_id, .expected_action]) | unique | length) == 1;
            "unaligned causal record at " + $stage) |
    ($models | map_values(.[$stage].score)) as $s |
    ($s.period_only + $s.post_b_only - $s.zero) as $predicted |
    {scores:$s, predicted_full:$predicted, residual:($s.full - $predicted),
     period_delta:($s.period_only - $s.zero), post_b_delta:($s.post_b_only - $s.zero)}
  end;

($panel | indexed(.)) as $p | ($keys | indexed(.)) as $k |
($zero | indexed(.)) as $z | ($period | indexed(.)) as $a |
($post_b | indexed(.)) as $b | ($full | indexed(.)) as $f |
($p | keys) as $ids |
require(($ids | length) > 0 and all($k, $z, $a, $b, $f; keys == $ids); "panel, keys or GPU membership mismatch") |
$ids[] as $id | $p[$id] as $row | $k[$id] as $key |
($row | [.pair_id, .side, .contrast, .family, .prefix, .positions, .ids, .expected]) as $identity |
require(all($z[$id], $a[$id], $b[$id], $f[$id];
    [.pair_id, .side, .contrast, .family, .prefix, .positions, .ids, .expected] == $identity)
    and $key.contrast == $row.contrast and ($key.family == null or $key.family == $row.family)
    and ($key.row_id == null or $key.row_id == ($row.pair_id + ":" + ($row.side | tostring)));
    "GPU or key identity mismatch: " + $id) |
require(($row.expected[-1] == 1) and ($row.expected | length) >= 3 and ($row.expected | length) <= 5
        and ($key.expected_word_count == null or $key.expected_word_count == ($row.expected | length) - 2)
        and ($key.expected_first_period_action == null or $key.expected_first_period_action == decision($row.expected[2]))
        and ($key.expected_post_b_action == null or $key.expected_post_b_action == decision($row.expected[3]));
        "invalid reference reply or decision key: " + $id) |
({zero:($z[$id] | model), period_only:($a[$id] | model), post_b_only:($b[$id] | model), full:($f[$id] | model)}) as $models |
{pair_id:$row.pair_id, side:$row.side, contrast:$row.contrast, family:$row.family,
 context_id:$key.context_id, selector_swap_id:$key.selector_swap_id,
 selector:$key.operative_clause_ordinal, one_word_clause:$key.one_word_clause_ordinal,
 axis:($key.pair_axis // "direct-count-control"), cohort:($key.cohort // "historical"),
 context:$key.context, question:$key.question, directive:$key.directive, reply:$key.reply,
 prefix_length:($row.prefix | length), positions:$row.positions, source_ids:$row.ids, expected:$row.expected,
 models:$models, additivity:{period:sum_identity($models; "period"), after_b:sum_identity($models; "after_b")}}
