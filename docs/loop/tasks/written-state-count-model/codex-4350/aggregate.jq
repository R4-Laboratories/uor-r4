def terminal: .[-3];
def counts($field):
    map(.[$field] | terminal | .event)
    | group_by(.)
    | map({key: .[0], value: length})
    | from_entries;
{
  schema,
  finite_controls,
  over_width: (
    .over_width | {
      openings: length,
      fixed_release: counts("fixed"),
      variable_native_release: counts("variable_native"),
      variable_stop_release: counts("variable_stop"),
      variable_emitted_source_tokens: (map(.variable_native | map(select(.event == "copy")) | length) | add),
      variable_supplied_stops: (map(.variable_stop | terminal | select(.commit == "supplied_stop")) | length),
      delimiter_refusals: (map(
        select((.variable_native | terminal | .event) == "delimiter")
        | {
            source_width: (.expected_ids | length),
            emitted_before_delimiter: (.variable_native | map(select(.event == "copy")) | length)
          }
      ) | sort_by(.source_width, .emitted_before_delimiter))
    }
  ),
  reversal: (.reversal | map({
    opening_tokens, role, original_bytes, absolute,
    expected_source_tokens: (.expected_ids | length),
    fixed_release: (.fixed | terminal | .event),
    variable_release: (.variable_native | terminal | .event),
    variable_cursor_at_release: (.variable_native | terminal | .cursor),
    native_handoff: (.variable_native | terminal | .commit),
    supplied_stop: (.variable_stop | terminal | .commit)
  }))
}
