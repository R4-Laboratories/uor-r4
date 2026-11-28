def need($condition; $message):
  if $condition then . else error($message) end;

def valid_token:
  type == "number" and . >= 0 and . < 4096 and . == floor;

def own_position($prefix):
  (.pos | type == "number" and . == floor and . >= 0 and . < ($prefix | length))
  and (.id | valid_token)
  and $prefix[.pos] == .id;

($truth | sort_by(.case_id)) as $rows
| ($traces | map(select(.artifact_arm == $arm and .client.authenticated_copy_steps != null))
    | group_by(.case_id) | map({key: .[0].case_id, value: .}) | from_entries) as $observed
| need(($rows | length) > 0 and ($rows | map(.case_id) | unique | length) == ($rows | length);
       "truth must have nonempty unique case IDs")
| need(($observed | keys) == ($rows | map(.case_id) | sort); "client cases differ from truth")
| $rows[]
| . as $case
| $case.prompt_ids as $prefix
| $case.sources[0] as $gold
| $case.case_id as $id
| need(($prefix | type == "array" and length > 0 and all(.[]; valid_token)); "invalid prompt: \($id)")
| need(($case.sources | type == "array") and $case.supported == ($gold != null);
       "inconsistent support truth: \($id)")
| need(all($case.sources[]; own_position($prefix)); "source differs from original token: \($id)")
| need($case.rival == null or ($case.rival | own_position($prefix)); "rival differs from original token: \($id)")
| need($case.same_id_rival == null or
       ($gold != null and ($case.same_id_rival | own_position($prefix)) and
        $case.same_id_rival.pos != $gold.pos and $case.same_id_rival.id == $gold.id);
       "same-ID rival lacks a distinct matching occurrence: \($id)")
| need(all($observed[$id][]; .client.prompt_ids == $prefix and .client.artifact_sha256 == $artifact);
       "ordinary client prompt or artifact identity differs: \($id)")
| need(($case.expected_ids | type == "array" and all(.[]; valid_token)); "invalid reference IDs: \($id)")
| {pair_id: $id, prefix: $prefix, raw_target: $case.expected_ids[0],
   target: ($gold.id // $case.eos_id), gold_pos: $gold.pos,
   rival_pos: $case.rival.pos, wrong_same_id_pos: $case.same_id_rival.pos}
