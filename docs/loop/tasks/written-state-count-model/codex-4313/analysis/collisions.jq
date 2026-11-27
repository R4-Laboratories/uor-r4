# Read one checkpoint's sealed device observations using jq -s.
def changes($key): map(select(.[0][$key] != .[1][$key])) | length;

map(select(.cohort == "controlled"))
| group_by(.pair_id)
| if all(.[]; length == 2 and .[0].side != .[1].side) then .
  else error("expected two distinct paired observations") end
| map(select(.[0].read_code == .[1].read_code and .[0].recurrence_code == .[1].recurrence_code))
| {
    pairs: length,
    changed_routes: changes("routes"),
    changed_read_vectors: changes("read_vector"),
    changed_frames: changes("recurrent_frames"),
    changed_sums: changes("recurrent_sum"),
    identical_frames_and_sums: (map(select(.[0].recurrent_frames == .[1].recurrent_frames and .[0].recurrent_sum == .[1].recurrent_sum)) | length),
    changed_recurrent_vectors: changes("recurrent_vector"),
    changed_source_positions: changes("source_position"),
    changed_source_ids: changes("source_token"),
    changed_gate_ids: changes("gate_selected"),
    changed_emitted_ids: changes("emitted")
  }
