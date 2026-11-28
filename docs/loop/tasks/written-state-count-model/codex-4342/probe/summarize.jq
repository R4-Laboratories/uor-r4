def sum(f): [ .[] | f ] | add // 0;
def count(f): map(select(f)) | length;

def aggregate($arm; $band):
  [ .[] | .arms[$arm][$band] | select(. != null) ] |
  {
    reachable_groups: length,
    all_wording_strict_compatible: count(.all_wording_strict_compatible == true),
    no_comparison_for_unpaired_groups: count(.all_wording_strict_compatible == null),
    minimal_obstruction: ([ .[] | .minimal_obstruction_case_ids | length | select(. > 0) ] |
      group_by(.) | map({wordings: .[0], groups: length})),
    equal_openings: sum(.equal_openings),
    reverse_comparisons: sum(.reverse_comparisons),
    reverse_pairs_canceling: sum(.reverse_pairs_canceling),
    reverse_pairs_compatible: sum(.reverse_pairs_compatible),
    repeated_target_pairs: sum(.repeated_target_pairs),
    paraphrase_target_key_equal: sum(.paraphrase_target_key_equal),
    paraphrase_both_action_keys_equal: sum(.paraphrase_both_action_keys_equal),
    paraphrase_contrasts_equal: sum(.paraphrase_contrasts_equal),
    source_swap_changed_groups: count(.source_swap_changed_contrasts > 0),
    source_swap_changed_comparisons: sum(.source_swap_changed_contrasts),
    target_equal_third_sources: sum(.target_equal_third_sources),
    target_equal_no_read_sides: sum(.target_equal_no_read_sides)
  };

def coverage($rows):
  ([ .[].opening_admissions[] ]) as $intakes |
  ([ $intakes[] | .tokens | select(. != null) ]) as $lengths |
  {
    source_qualified: length, articles: (map(.article) | unique | length),
    ordinary_admitted_groups: count(all(.opening_admissions[]; .refusal == null)),
    ordinary_admitted_openings: ($intakes | count(.refusal == null)),
    ordinary_refusals: ([ $intakes[] | .refusal | select(. != null) ] |
      group_by(.) | map({reason: .[0], openings: length})),
    shortest_opening: ($lengths | min), longest_opening: ($lengths | max),
    fully_native_reachable: count((.arms | length) > 0),
    with_question_dependent_candidates: count(.different_fact_token_prefix or .different_candidate_spans),
    original_target_unreachable: sum(.original_target_unreachable),
    original_mate_unreachable: sum(.original_mate_unreachable),
    unreachable_address_reasons: ([ .[].unreachable_address_reasons[] ] |
      group_by(.action, .reason) |
      map({action: .[0].action, reason: .[0].reason, comparisons: length})),
    original_text_changed_by_protocol: sum(.original_text_changed_by_protocol),
    original_target_queries_missing: sum(.missing.target_queries),
    original_target_sources_missing: sum(.missing.target_sources),
    original_target_either_missing: sum(.missing.target_either),
    original_mate_queries_missing: sum(.missing.mate_queries),
    original_mate_sources_missing: sum(.missing.mate_sources),
    original_mate_either_missing: sum(.missing.mate_either),
    fully_reachable_with_both_neighborhoods: count((.arms | length) > 0 and (.missing.target_either == 0 and .missing.mate_either == 0)),
    truth_cases: sum(.truth_cases), labeled_no_read: sum(.labeled_no_read),
    no_read_choices_seen: sum(.no_read_choices_seen),
    source_choices_seen: sum(.source_choices_seen),
    all_source_choices_missing_query: sum(.all_source_choices_missing_query),
    all_source_choices_missing_source: sum(.all_source_choices_missing_source),
    all_source_choices_missing_either: sum(.all_source_choices_missing_either),
    structural_screen_group_pass: count(.registered_group_pass),
    arms: (
      [ "geometry", "count", "directed_geometry", "directed_count", "swapped_geometry" ] |
      map(. as $arm | {key: $arm, value: {
        all_banks: ($rows | aggregate($arm; "all_banks")),
        bank_10: ($rows | aggregate($arm; "bank_10"))
      }}) | from_entries
    )
  };

def for_groups:
  . as $rows | coverage($rows);

{
  schema: "uor-r4/directed-relation-structural-aggregate/1",
  source_sha256, artifact_context, source_qualified_cases, verified_client_cases, unadmitted_client_cases,
  verified_legacy_openings, verified_relation_openings,
  registered_group_pass_count, registered_article_pass_count, screen_licenses_later_fit_design,
  overall: (.qualified_group_rows | for_groups),
  by_split_stratum_contrast: (.qualified_group_rows | group_by(.split,.stratum,.contrast) |
    map(.[0] as $sample | {split: $sample.split, stratum: $sample.stratum, contrast: $sample.contrast} + for_groups)),
  by_article: (.qualified_group_rows | group_by(.split,.article) |
    map(.[0] as $sample | {split: $sample.split, article: $sample.article} + for_groups))
}
