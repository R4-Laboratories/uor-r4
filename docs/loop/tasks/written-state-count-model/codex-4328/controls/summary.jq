def count_if(predicate): map(select(predicate)) | length;

def counts:
  . as $rows
  | {
      rows: length,
      reader_gates: (group_by(.reader_gate) | map({key: .[0].reader_gate, value: length}) | from_entries),
      supplied_eligible: count_if(.reader.verified_source_eligible),
      supplied_exact_commit: count_if(.arms.correct.address_authenticated and .arms.correct.exact_source and .arms.correct.copied),
      supplied_first_reference: count_if(.arms.correct.first_id_matches_reference and .supported),
      supplied_first_and_self_fed_second_reference: count_if(
        .supported and .arms.correct.first_id_matches_reference and .arms.correct.successor_matches_reference),
      supplied_exact_reply_in_two_steps: count_if(.arms.correct.exact_reply_within_two_steps and .supported),
      same_id_distinct_pair_eligible: count_if(.reader.verified_source_eligible and .reader.same_id_rival_eligible),
      same_id_distinct_pair_minimum_distance: count_if(.reader.verified_source_eligible and .reader.same_id_rival_eligible and
        .arms.same_id_wrong_occurrence.source_admitted and
        .arms.same_id_wrong_occurrence.source_distance == .arms.same_id_wrong_occurrence.minimum_distance),
      same_id_wrong_claim_copied: count_if(.arms.same_id_wrong_occurrence.copied),
      missing_claim_copied: count_if(.arms.missing.copied),
      no_occurrence_read_copied: count_if(.arms.marker.copied),
      learned_verified_copy: count_if(.arms.learned.address_authenticated and .arms.learned.exact_source and .arms.learned.copied),
      learned_wrong_supported_copy: count_if(.supported and .arms.learned.copied and (.arms.learned.exact_source | not)),
      learned_unsupported_copy: count_if((.supported | not) and .arms.learned.copied),
      learned_verified_copy_and_self_fed_second_reference: count_if(.arms.learned.address_authenticated and
        .arms.learned.exact_source and .arms.learned.copied and .arms.learned.first_id_matches_reference and
        .arms.learned.successor_matches_reference),
      arm_first_commit_eos: ($rows[0].arms | keys | map(. as $arm |
        {key: $arm, value: ($rows | count_if(.arms[$arm].committed_ids[0] == 1))}) | from_entries)
    };

{
  total: counts,
  cohorts: (group_by([.fenced, .supported, .family]) | map({
    fenced: .[0].fenced, supported: .[0].supported, family: .[0].family, counts: counts
  }))
}
