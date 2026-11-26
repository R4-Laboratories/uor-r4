include "common";

def tally(condition): map(select(condition)) | length;
def distribution: group_by(.) | map({value:.[0],count:length});
def common_counts:
  {
    endpoints:length,
    source_identity_disagreements:tally(.source_identity_ok | not),
    rejected_winner:tally(.disposition.source_rejected),
    observed_status:(map(.disposition.status) | distribution),
    valid_abstentions:tally(.disposition.valid_abstention),
    zero_bound_emission:tally(.disposition.zero_bound_emission),
    stock_trace_parity:tally(.disposition.stock_parity),
    gold_scope_exact:tally(.gold_scope_exact),
    correct_writer_route:tally(.correct_route),
    exact_terminal:tally(.exact_terminal),
    authenticated_take_eos:tally(.authenticated_take_eos),
    authenticated_defer_parity:tally(.authenticated_defer_parity),
    authenticated_defer_eos:tally(.authenticated_defer_eos),
    authenticated_defer_cap:tally(.authenticated_defer_parity and .returned.stop == "max_tokens"),
    first_correct_prefix_at_cap:tally(.first.correct_prefix_at_cap),
    returned_correct_prefix_at_cap:tally(.returned.correct_prefix_at_cap),
    observed_stops:(map(.returned.stop) | distribution),
    retries:tally(.retry),
    retry_steps:(map(.retry_steps) | add),
    first_wrong_position:(map(.first.first_wrong.position) | distribution),
    returned_wrong_position:(map(.returned.first_wrong.position) | distribution)
  };

assert_ok(length == $pair_count * 2 * (arms | length);
  "ordinary readout does not cover the reviewed paired panel") |
. as $all |
{
  pair_count:$pair_count,
  source_identity_disagreements:[
    $all[] | select(.source_identity_ok | not) | {id,arm,pair_id,disposition}],
  arms:[
    (arms[]) as $arm |
    [$all[] | select(.arm == $arm)] as $rows |
    assert_ok(($rows | map(.id) | unique | length) == $pair_count * 2;
      "ordinary arm duplicates cases: " + $arm) |
    ($rows | group_by(.pair_id) |
      assert_ok(length == $pair_count and
        all(.[]; length == 2 and (map(.admission) | sort) == ["defer","take"]);
        "ordinary arm misses one admission direction: " + $arm)) as $pairs |
    {
      arm:$arm,all:($rows | common_counts),
      take:($rows | map(select(.admission == "take")) | common_counts),
      defer:($rows | map(select(.admission == "defer")) | common_counts),
      by_distance:($rows | group_by(.distance) | map({distance:.[0].distance,counts:common_counts})),
      both_exact_eos:($pairs | tally(all(.[]; .exact_terminal))),
      both_original_source_reply_eos:($pairs | tally(all(.[]; .authenticated_take_eos or .authenticated_defer_eos))),
      both_gold_scope_and_reply_eos:($pairs | tally(all(.[];.gold_scope_exact and
        (.authenticated_take_eos or .authenticated_defer_eos)))),
      paired_take_eos_and_defer_stock:($pairs | tally(all(.[]; .authenticated_take_eos or .authenticated_defer_parity)))
    }]
}
