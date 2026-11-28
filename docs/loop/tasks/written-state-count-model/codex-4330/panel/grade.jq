def guard($pass; $case; $why):
  if $pass then . else error($case + ": " + $why) end;
def original_pos($s):
  if $s.source_pos == null then null else $s.prefix_start + $s.source_pos end;
def cap: 16;
def first_error($expected; $actual):
  if $expected == null or $expected == $actual then null
  else [range(0; ([([($expected|length), ($actual|length)]|max), cap]|min)) as $i
        | select($expected[$i] != $actual[$i])
        | {step:$i, expected:$expected[$i], actual:$actual[$i]}][0] end;
def check_client($obs; $prefix):
  ($obs.client) as $c | ($obs.case_id) as $id |
  $c.generated_ids as $actual | $c.native_occurrence_steps as $steps |
  ($prefix|length) as $n | ($prefix + $actual) as $causal |
  guard($c.schema == "uor-r4.uor-chat-discrete-text/1" and $c.context == 192
        and $c.prompt_ids == $prefix and ($steps|length) == ($actual|length)
        and ($actual|length) >= 1 and ($actual|length) <= cap;
        $id; "prefix or step shape changed") |
  guard((if $c.stop == "eos" then $actual[-1] == 1 and ($actual[:-1]|index(1)) == null
         else $c.stop == "max_tokens" and ($actual|length) == cap and ($actual|index(1)) == null end);
        $id; "EOS and stop event disagree") |
  guard(all(range(0; $actual|length);
    . as $i | $steps[$i] as $s | ([$n + $i - $c.context, 0]|max) as $start |
    $s.prefix_start == $start and $s.prefix_len == ($n + $i - $start)
    and $s.previous_id == $causal[$n + $i - 1] and $s.committed_id == $actual[$i]
    and (if $s.source_pos == null then $s.source_id == null
         else original_pos($s) >= 0 and original_pos($s) < ($n + $i)
           and $causal[original_pos($s)] == $s.source_id and $s.source_id == $s.gate_id end)
    and (if $s.claim_pos == null then $s.claimed_id == null and $s.copied == false
         else $i == 0 and $s.source_pos != null and $s.claim_pos == $s.source_pos
           and $s.claimed_id == $s.source_id end)
    and (if $s.copied then $s.claim_pos != null and $s.committed_id == $s.source_id
         else $s.committed_id == $s.native_proposal end));
    $id; "trace violates original address or causal self-feed");

def score($key; $obs):
  $obs.client.generated_ids as $actual |
  $obs.client.native_occurrence_steps as $steps |
  $steps[0] as $head |
  $key.source_delay as $step |
  ($step == 0 or ($actual[0] == 223 and ($actual|length) > 1)) as $phase |
  (if $phase then $steps[$step] else null end) as $read |
  ($read // {source_pos:null}) as $phase_read |
  {
    case_id:$key.case_id, cohort:$key.cohort, split:$key.split, family:$key.family,
    contrast_group:$key.contrast_group, contrast_type:$key.contrast_type,
    paired_ids:$key.paired_ids, supported:$key.supported, target:$key.source,
    source_delay:$step, lexical_phase_reached:$phase,
    first_read_position:original_pos($head),
    source_step_position:original_pos($phase_read),
    source_step_id:$phase_read.source_id,
    source_correct:(
      if $key.supported then $phase and original_pos($phase_read) == $key.source.pos
        and $phase_read.source_id == $key.source.id
      else $head.source_pos == null end),
    wrong_same_id_position:($key.supported and $phase and $phase_read.source_id == $key.source.id
        and original_pos($phase_read) != $key.source.pos),
    claimed_first:($head.claim_pos != null),
    copied_first:($head.copied == true),
    same_id_control:($key.same_id_rival != null),
    same_id_rival_selected:($key.same_id_rival != null and original_pos($phase_read) == $key.same_id_rival.pos),
    copied_first_supported:($key.supported and $step == 0 and $head.copied == true
        and original_pos($head) == $key.source.pos and $head.committed_id == $key.source.id),
    reference_ids:(if $key.expected_ids == null then null else ($key.expected_ids|length) end),
    reference_over_cap:($key.expected_ids != null and ($key.expected_ids|length) > cap),
    first_id_correct:(if $key.expected_ids == null then null else $actual[0] == $key.expected_ids[0] end),
    first_wrong:first_error($key.expected_ids; $actual),
    exact_eos:(if $key.expected_ids == null or ($key.expected_ids|length) > cap then null
               else $actual == $key.expected_ids and $obs.client.stop == "eos" end),
    termination:$obs.client.stop,
    termination_id:(if $obs.client.stop == "eos" then $actual[-1] else null end),
    decoded_eos_text:(($obs.client.text // "") | contains("<|eos|>"))
  };

def pairs($rows):
  ($rows | INDEX(.case_id)) as $by_id |
  [ $rows[] | . as $x | .paired_ids[] | select($x.case_id < .) |
    $by_id[.] as $y | select($y != null) |
    {contrast_type:$x.contrast_type,
     source_delay:([$x.source_delay, $y.source_delay]|max),
     unsupported_claim:((($x.supported|not) and $x.claimed_first) or (($y.supported|not) and $y.claimed_first)),
     unsupported_copy:((($x.supported|not) and $x.copied_first) or (($y.supported|not) and $y.copied_first)),
     flips_support:($x.supported != $y.supported),
     both_source_correct:($x.source_correct and $y.source_correct),
     both_exact_eos:($x.exact_eos == true and $y.exact_eos == true)} ];
def summary($rows):
  pairs($rows) as $pairs |
  {
    rows:($rows|length),
    positives:([$rows[]|select(.supported)]|length),
    positives_immediate:([$rows[]|select(.supported and .source_delay == 0)]|length),
    positives_after_space:([$rows[]|select(.supported and .source_delay != 0)]|length),
    positives_after_space_reached:([$rows[]|select(.supported and .source_delay != 0 and .lexical_phase_reached)]|length),
    correct_source_immediate:([$rows[]|select(.supported and .source_delay == 0 and .source_correct)]|length),
    correct_source_after_space:([$rows[]|select(.supported and .source_delay != 0 and .source_correct)]|length),
    correct_authenticated_first_copy:([$rows[]|select(.copied_first_supported)]|length),
    no_read_cases:([$rows[]|select(.supported|not)]|length),
    no_read_correct:([$rows[]|select((.supported|not) and .source_correct)]|length),
    unsupported_claims:([$rows[]|select((.supported|not) and .claimed_first)]|length),
    unsupported_copies:([$rows[]|select((.supported|not) and .copied_first)]|length),
    duplicate_controls:([$rows[]|select(.same_id_control)]|length),
    correct_duplicate_controls:([$rows[]|select(.same_id_control and .source_correct)]|length),
    wrong_same_id:([$rows[]|select(.wrong_same_id_position)]|length),
    reference_replies:([$rows[]|select(.reference_ids != null)]|length),
    reference_within_cap:([$rows[]|select(.exact_eos != null)]|length),
    reference_over_cap:([$rows[]|select(.reference_over_cap)]|length),
    observed_eos:([$rows[]|select(.termination == "eos")]|length),
    canonical_eos:([$rows[]|select(.termination_id == 1)]|length),
    eos_like_surface:([$rows[]|select(.decoded_eos_text)]|length),
    eos_text_without_termination:([$rows[]|select(.decoded_eos_text and .termination != "eos")]|length),
    stopped_at_cap:([$rows[]|select(.termination == "max_tokens")]|length),
    first_id_correct:([$rows[]|select(.first_id_correct == true)]|length),
    exact_eos:([$rows[]|select(.exact_eos == true)]|length),
    first_wrong_step:([$rows[]|.first_wrong|select(. != null)|.step] | group_by(.) | map({step:.[0],rows:length})),
    pairs:($pairs|length),
    source_correct_pairs:([$pairs[]|select(.both_source_correct)]|length),
    exact_reply_pairs:([$pairs[]|select(.both_exact_eos)]|length),
    support_flip_pairs:([$pairs[]|select(.flips_support)]|length),
    correct_support_flip_pairs:([$pairs[]|select(.flips_support and .both_source_correct)]|length),
    matched_source_phases:($pairs | group_by(.source_delay) | map({
      source_delay:.[0].source_delay, pairs:length,
      source_correct_pairs:([.[]|select(.both_source_correct)]|length),
      support_flip_pairs:([.[]|select(.flips_support)]|length),
      support_flip_wins:([.[]|select(.flips_support and .both_source_correct)]|length),
      unsupported_claim_pairs:([.[]|select(.unsupported_claim)]|length),
      unsupported_copy_pairs:([.[]|select(.unsupported_copy)]|length)
    }))
  };

($eval | INDEX(.case_id)) as $model |
($gold | INDEX(.case_id)) as $keys |
($ordinary | INDEX(.case_id)) as $observed |
guard(($eval|length) == ($model|length) and ($gold|length) == ($keys|length)
      and ($ordinary|length) == ($observed|length)
      and ($model|keys) == ($keys|keys) and ($keys|keys) == ($observed|keys)
      and ($ordinary | map(.arm) | unique) == [$arm];
      $arm; "non-exact case join or mixed arms") |
[
  ($keys|keys[]) as $id |
  $model[$id] as $prompt |
  $observed[$id] as $obs |
  check_client($obs; $prompt.prefix) |
  score($keys[$id]; $obs)
] as $rows |
{
  schema:"uor-r4.whole-turn-fit-grade/1", arm:$arm,
  cohort_summary:($rows | group_by([.cohort,.split]) |
      map({cohort:.[0].cohort,split:.[0].split} + summary(.))),
  protected_groups:($rows | map(select(.cohort == "protected")) | group_by(.contrast_type) |
      map({group:.[0].contrast_type} + summary(.))),
  authored_groups:($rows | map(select(.cohort == "authored")) | group_by([.split,.contrast_type]) |
      map({split:.[0].split,group:.[0].contrast_type} + summary(.))),
  protected_c021_c024:([$rows[] | select(.case_id | test("^protected-c02[1-4]$"))]),
  rows:$rows
}
