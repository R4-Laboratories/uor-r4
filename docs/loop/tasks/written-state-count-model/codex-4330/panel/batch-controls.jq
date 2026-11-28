# Preserve each article, each authored statement type and each protected family.
# Named regression controls and both delayed human splits are included explicitly.
($prompts | INDEX(.case_id)) as $by_id |
[
  ($gold | map(select(.cohort == "squad")) | group_by([.split,.family]) | .[][0].case_id),
  ($gold | map(select(.cohort == "squad" and .source_delay == 1)) | group_by(.split) | .[][0].case_id),
  ($gold | map(select(.cohort == "authored")) | group_by([.split,.family,.contrast_type]) | .[][0].case_id),
  ($gold | map(select(.cohort == "protected")) | group_by([.family,.contrast_type]) | .[][0].case_id),
  ($gold[] | select(.case_id | test("^protected-c02[1-4]$")) | .case_id)
] | unique | .[] | $by_id[.]
