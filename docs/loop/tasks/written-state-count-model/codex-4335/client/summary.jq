group_by(.arm) | map(
    . as $all
    | map(select(.grade != null)) as $known
    | map(select(.grade == null)) as $unspecified
    | {
          arm: $all[0].arm,
          exact_raw_prefix_rows: ($all | length),
          supported: ($known | length),
          first_id_correct: ($known | map(select(.grade.first_id_correct)) | length),
          first_address_correct: ($known | map(select(.grade.first_address_correct)) | length),
          supported_selected_address: ($known | map(select(.writer.selected_address != null)) | length),
          supported_committed_address: ($known | map(select(.writer.committed_address != null)) | length),
          supported_eos: ($known | map(select(.eos_emitted)) | length),
          exact_through_eos: ($known | map(select(.grade.exact_through_eos)) | length),
          exact_address_and_through_eos: ($known | map(select(.grade.exact_through_eos and .grade.first_address_correct)) | length),
          matched_prefix_histogram: ($known | group_by(.grade.matched_prefix_ids) |
              map({length: .[0].grade.matched_prefix_ids, cases: length})),
          first_wrong_index_histogram: ($known | map(select(.grade.first_wrong != null)) |
              group_by(.grade.first_wrong.index) | map({index: .[0].grade.first_wrong.index, cases: length})),
          unspecified: ($unspecified | length),
          unspecified_selected_address: ($unspecified | map(select(.writer.selected_address != null)) | length),
          unspecified_committed_address: ($unspecified | map(select(.writer.committed_address != null)) | length),
          unspecified_eos: ($unspecified | map(select(.eos_emitted)) | length),
          unspecified_decisions: ($unspecified | group_by(.writer.decision) | map({decision: .[0].writer.decision, cases: length})),
          period_release_events: ($all | map(select(.writer.period_release != null)) | length)
      }
)
