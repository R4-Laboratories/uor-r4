include "lanes";

checked_rows($plan; $selected; $pair_count) |
group_by(.arm) |
map({arm:.[0].arm,events:length,first_exact:map(select(.first.exact_through_eos))|length,
    returned_exact:map(select(.returned.exact_through_eos))|length,
    exact_pairs:group_by(.pair_id)|map(select(all(.[];.returned.exact_through_eos)))|length,
    prefix_at_cap:map(select(.returned.correct_prefix_at_cap))|length,
    first_errors:map(.first.first_differing_id)|group_by(.)|map({id:.[0],n:length}),
    returned_errors:map(.returned.first_differing_id)|group_by(.)|map({id:.[0],n:length})})
