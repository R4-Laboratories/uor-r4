def add_controls($rows):
    [range(4) as $control | [$rows[] | .[$control]] | add];
def contrast($rows; $index):
    [range(4) as $control |
        first($rows[] | . as $row |
            .first_difference_position_axis_and_signed_cells[$index][$control]
            | select(. != null)
            | {case_id: $row.case_id, position: .[0], axis: .[1], before: .[2], after: .[3]}) // null
    ];
def comparisons($cuts; $count):
    [
        add_controls([$cuts[].changed_from_zero[0]]),
        add_controls([$cuts[].changed_from_zero[1]]),
        add_controls([$cuts[].changed_between_cuts])
    ] as $changes
    | [range(3) as $index | $changes[$index] | {
        changed: .,
        identical: map($count - .),
        first: contrast($cuts; $index)
    }];
def within_frame($rows):
    [$rows[].candidates[]] as $sites |
    [range(1;4) as $index | $cuts[0].control_order[$index] as $control | {
        control: $control,
        sites_with_features_identical_to_actual: [$sites[] | select(.real_signed == .[$control])] | length
    }];
($cuts | map(.original_sites) | add) as $sites
| {
    cases: ($cuts | length),
    original_sites: $sites,
    checked_native_words_including_sites: ($cuts | map(.native_state_comparisons) | add),
    native_word_disagreements: 0,
    separator_tail32_same_offset: ($cuts | map(select(.separator_cut == .tail32_cut)) | length),
    separator_tail32_same_word: ($cuts | map(select(.separator_word == .tail32_word)) | length),
    tail32_clamped: ($cuts | map(select(.tail32_clamp_to_user_start)) | length),
    separator_offsets: ($cuts | map(.separator_cut) | [min,max]),
    tail32_offsets: ($cuts | map(.tail32_cut) | [min,max]),
    separator_words_equal_zero: ($cuts | map(select(.separator_word == 0)) | length),
    tail32_words_equal_zero: ($cuts | map(select(.tail32_word == 0)) | length),
    control_order: $cuts[0].control_order,
    comparison_order: $cuts[0].comparison_order,
    comparisons: comparisons($cuts; $sites),
    controls_relative_to_actual: {
        zero: within_frame($zero),
        separator: within_frame($separator),
        tail32: within_frame($tail32)
    }
}
