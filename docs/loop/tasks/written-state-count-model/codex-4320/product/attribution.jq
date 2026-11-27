# Aggregate read-only gate and writer diagnostics from the cold JSONL.
# jq -s -f attribution.jq cold.jsonl
def counts(f):
  group_by(f) | map({value:(.[0]|f), rows:length}) | sort_by(-.rows, .value);
def named_terms:
  {query:.[0], query_read:(.[1]*64), read:(.[2]*2),
   query_recurrence:(.[3]*32), read_recurrence:(.[4]*8), pointer:.[5]};
def dominant:
  named_terms | to_entries | (map(.value)|max) as $maximum |
  map(select(.value==$maximum and $maximum>0)|.key);
def candidate($id): [.third_trace.candidates[]|select(.id==$id)][0];
def scores(select_id):
  map((.|select_id) as $id
      | (.attribution.gate_positive_ranks|map(select(.[0]==$id)|.[1])|first) as $gate_rank
      | candidate($id) + {positive_gate_rank:$gate_rank}) |
  {candidate_rows:(map(select(.!=null))|length), score:counts(.score),
   weighted_terms:counts(.terms|named_terms), dominant:counts(.terms|dominant), unigram:counts(.unigram),
   in_gate_pool:counts(.in_pool), positive_gate_rank:counts(.positive_gate_rank),
   admitted:counts(.admitted), writer_rank:counts(.writer_rank),
   distance:counts(.distance), distance_ties:counts(.distance_ties), admitted_ties:counts(.admitted_ties)};
{rows:length,
 active_gate:counts(.third_trace.gate_id), mixed_token:counts(.third_trace.mixed_token), emitted:counts(.third_trace.writer_id),
 gate_score_ties:counts(.attribution.gate_score_ties), gate_unigram_ties:counts(.attribution.gate_equal_score_and_unigram),
 gate_margin:counts(.attribution.gate_score_margin), query_only_winner:counts(.attribution.query_only_winner),
 sensitivity: {no_pointer:counts(.third_trace.sensitivity[0]), no_read:counts(.third_trace.sensitivity[1]), no_recurrence:counts(.third_trace.sensitivity[2])},
 sensitivity_changed: {no_pointer:(map(select(.third_trace.sensitivity[0]!=.third_trace.mixed_token))|length),
                       no_read:(map(select(.third_trace.sensitivity[1]!=.third_trace.mixed_token))|length),
                       no_recurrence:(map(select(.third_trace.sensitivity[2]!=.third_trace.mixed_token))|length)},
 pointer:counts({location:.attribution.pointer_location, predecessor:.third_trace.pointer_predecessor[1], successor:.third_trace.pointer_successor[1]}),
 packed_read_codes:counts(.third_trace.read_code), recurrence_codes:counts(.third_trace.recurrence_code),
 minimum_ties:counts(.attribution.writer_minimum_ties), pool_size:counts(.third_trace.pool_size),
 g:scores(73), gate:scores(.third_trace.gate_id), eos:scores(1), expected_next_word:scores(.next_id)}
