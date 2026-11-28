def need($condition; $message):
  if $condition then . else error($message) end;

def by_case($field):
  map({key: .[$field], value: .}) | from_entries;

def authenticate($prefix):
  .read_pos != null and .claim_pos == .read_pos and .source_id != null and
  .source_id == $prefix[.read_pos] and .gate_id == .source_id;

def exact_source($gold):
  $gold != null and .read_pos == $gold.pos and .source_id == $gold.id;

def grade_arm($prefix; $gold; $expected; $eos):
  .first as $first
  | ([.first, .successor] | map(select(. != null) | .committed_id)) as $commits
  | {
      read_pos: $first.read_pos, claim_pos: $first.claim_pos, source_id: $first.source_id,
      gate_id: $first.gate_id, source_distance: $first.source_distance,
      minimum_distance: $first.minimum_distance, source_admitted: $first.source_admitted,
      native_proposal: $first.native_proposal,
      address_authenticated: ($first | authenticate($prefix)),
      exact_source: ($first | exact_source($gold)),
      copied: $first.copied, committed_ids: $commits,
      first_id_matches_reference: ($expected[0] != null and $commits[0] == $expected[0]),
      successor_matches_reference: ($expected[1] != null and $commits[1] == $expected[1]),
      exact_reply_within_two_steps: ($expected[-1] == $eos and $commits == $expected),
      eos_observed: ($commits[-1] == $eos)
    };

def first_reader_gate($gold; $gold_candidate):
  if .parsed_start == null then "framing_rejected"
  elif $gold == null then if .winner_pos == null then "unsupported_abstained" else "unsupported_read" end
  elif $gold_candidate == null or $gold_candidate.eligible != true then "verified_source_ineligible"
  elif .winner_pos == $gold.pos then "exact_source_selected"
  elif .best_score == null then "no_candidates"
  elif .winner_pos != null then "another_occurrence_selected"
  elif .best_ties > 1 then "top_score_tie"
  elif .best_score <= .no_read_score then "no_read_threshold"
  else error("reader assessment contains an unexplained abstention")
  end;

($truth | by_case("case_id")) as $truth_by_case
| ($assessment | by_case("pair_id")) as $assess_by_case
| ($traces | map(select(.artifact_arm == $arm and .client.authenticated_copy_steps != null))
    | group_by(.case_id) | map({key: .[0].case_id, value: .}) | from_entries) as $observed
| need(($truth | length) == ($copy | length) and ($truth | length) == ($assessment | length) and
       ($truth_by_case | keys) == ($copy | map(.pair_id) | sort) and
       ($truth_by_case | keys) == ($assess_by_case | keys);
       "copy or reader assessment case census differs")
| $copy[]
| . as $copy_row
| .pair_id as $id
| .prefix as $prefix
| $truth_by_case[$id] as $case
| $assess_by_case[$id] as $reader
| $case.sources[0] as $gold
| $case.rival as $rival
| $case.same_id_rival as $duplicate
| ($reader.candidates | map(select(.pos == $gold.pos)) | first) as $gold_candidate
| ($reader.candidates | map(select(.pos == $rival.pos)) | first) as $rival_candidate
| ($reader.candidates | map(select(.pos == $duplicate.pos)) | first) as $duplicate_candidate
| ($reader.candidates | map(select(.pos == $reader.winner_pos)) | first) as $selected
| ([$copy_row.arms.learned.first, $copy_row.arms.learned.successor] | map(select(. != null))) as $learned
| need($case.prompt_ids == $prefix and $reader.prefix == $prefix; "raw prefix differs: \($id)")
| need($reader.winner_pos == $copy_row.arms.learned.first.read_pos; "assessment differs from native read: \($id)")
| need(all($observed[$id][]; .client.authenticated_copy_steps[:($learned | length)] == $learned and
          .client.generated_ids[:($learned | length)] == ($learned | map(.committed_id)));
       "copy verifier differs from ordinary authenticated client: \($id)")
| need(all($copy_row.arms[]; . as $trajectory |
          (.successor == null) == (.first.committed_id == $case.eos_id) and
          .terminal == ((.successor // .first).committed_id == $case.eos_id) and
          (if .first.copied then (.first | authenticate($prefix)) and
               .first.source_admitted and .first.source_distance == .first.minimum_distance and
               .first.committed_id == .first.source_id
           else .first.committed_id == .first.native_proposal end) and
          ($trajectory.successor == null or
           ($trajectory.successor.copied == false and $trajectory.successor.claim_pos == null)));
       "invalid copy or self-fed continuation: \($id)")
| need($copy_row.arms.marker.first.read_pos == null and $copy_row.arms.marker.first.claim_pos == null and
       $copy_row.arms.missing.first.claim_pos == null and $copy_row.arms.missing.first.copied == false and
       $copy_row.arms.invalid.first.copied == false and $copy_row.arms.same_id_wrong_occurrence.first.copied == false;
       "a negative control copied: \($id)")
| if $gold_candidate.eligible then
    need($copy_row.arms.correct.first.read_pos == $gold.pos and
         $copy_row.arms.missing.first.read_pos == $gold.pos and
         ($copy_row.arms.correct.first | authenticate($prefix));
         "eligible supplied control did not hold the original read: \($id)")
  else . end
| if $gold_candidate.eligible and $duplicate_candidate.eligible then
    need($copy_row.arms.same_id_wrong_occurrence.first.read_pos == $gold.pos and
         $copy_row.arms.same_id_wrong_occurrence.first.claim_pos == $duplicate.pos and
         $copy_row.arms.same_id_wrong_occurrence.first.source_id == $gold.id and
         $copy_row.arms.same_id_wrong_occurrence.first.read_vector == $copy_row.arms.correct.first.read_vector and
         $copy_row.arms.same_id_wrong_occurrence.first.native_proposal == $copy_row.arms.correct.first.native_proposal;
         "same-ID refusal did not hold token and writer state fixed: \($id)")
  else . end
| {case_id: $id, group: $case.group, family: $case.family, fenced: $case.fenced,
   supported: $case.supported, fact_order: $case.fact_order, question_target: $case.question_target,
   expected_ids: $case.expected_ids, gold: $gold, rival: $rival, same_id_rival: $duplicate,
   reader_gate: ($reader | first_reader_gate($gold; $gold_candidate)),
   reader: {
     parsed_start: $reader.parsed_start, parsed_end: $reader.parsed_end,
     eligible_candidates: ($reader.candidates | map(select(.eligible)) | length),
     winner_pos: $reader.winner_pos, selected_score: $selected.score,
     best_score: $reader.best_score, best_ties: $reader.best_ties, no_read_score: $reader.no_read_score,
     passes_no_read_threshold: ($reader.best_score != null and $reader.best_score > $reader.no_read_score),
     gold_rank: $reader.gold_rank, gold_score: $gold_candidate.score,
     rival_rank: $reader.rival_rank, rival_score: $rival_candidate.score,
     same_id_rival_rank: $duplicate_candidate.rank, same_id_rival_score: $duplicate_candidate.score,
     verified_source_eligible: ($gold_candidate.eligible // false),
     same_id_rival_eligible: ($duplicate_candidate.eligible // false)
   },
   arms: ($copy_row.arms | map_values(grade_arm($prefix; $gold; $case.expected_ids; $case.eos_id)))}
