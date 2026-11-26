def require($ok; $message): if $ok then . else error($message) end;

def full_near_pairs:
    map(select(.distance == "near")) | group_by(.pair_id) |
    map(select(length == 2 and all(.[]; .arm.full == true)) | .[0].pair_id);

def source($text; $address):
    $address.bytes as [$start, $end] |
    [range(($text | length) + 1) as $i | select(($text[:$i] | utf8bytelength) == $start) | $i] as $starts |
    [range(($text | length) + 1) as $i | select(($text[:$i] | utf8bytelength) == $end) | $i] as $ends |
    require(($starts | length) == 1 and ($ends | length) == 1 and $start < $end; "invalid source byte range") |
    {text: $text[$starts[0]:$ends[0]], bytes: $address.bytes, bpe: $address.tokens};

. as $election |
($before | full_near_pairs) as $protected |
($protected - ($after | full_near_pairs)) as $first_lost |
require(($election.trajectory | map(.step)) == [range(9)] and $protected == $election.original_near and
        $first_lost == $election.trajectory[1].lost_original_near and ($first_lost | length) > 0;
        "checkpoint loss identities disagree with the actual joint reads") |
($first_lost[0]) as $pair |
($after | map(select(.pair_id == $pair and .arm.full == false)) | min_by(.ordinal)) as $row |
($before[$row.ordinal]) as $parent |
($physical[$row.ordinal]) as $raw |
($reference[0] | map(select(.id == $row.case_id)) | first) as $ref |
require($parent.case_id == $row.case_id and $parent.arm.full and $raw.row == $row.ordinal and
        $ref.structural_prompt_sha256 == $row.prompt_sha256 and $row.admission == $ref.admission;
        "witness row identity mismatch") |
($oracle | map(select(.row == $row.ordinal and .eligible_for_fit)) |
    map(select(. as $edge | $raw.links[$edge.sign] as $link |
        $link.governor == $edge.governor_address and $link.noun != $edge.gold_address and
        $link.site == $edge.history[1].rival_site and $edge.history[0].margin > 0 and $edge.history[1].margin < 0)) |
    first) as $edge |
($raw.links[$edge.sign]) as $link |
($ref.turns[1]) as $text |
require($edge != null and $edge.history[0].loss == 0; "no eligible strict-loss witness with an unchanged governor") |
{
    scope: "mixed TRAIN only",
    original_protected_near_count: ($protected | length),
    first_sweep_lost: $first_lost,
    protected_near_losses_by_step: ($election.trajectory | map({step, lost: (.lost_original_near | length)})),
    lost_at_every_updated_step: [$protected[] as $id |
        select(all($election.trajectory[1:][]; .lost_original_near | index($id) != null)) | $id],
    witness: {
        pair: $pair, case_id: $row.case_id, row: $row.ordinal, admission: $row.admission,
        sign: (["affirmed", "excluded"][$edge.sign]), source_error: $row.arm.first_source_error,
        governor: source($text; $edge.governor_address),
        true_noun: source($text; $edge.gold_address),
        selected_rival: source($text; $link.noun),
        edge_before: ($edge.history[0] | {margin, loss, gold_cost, rival_cost, gold_center, rival_center, rival_site}),
        edge_after_first_sweep: ($edge.history[1] | {margin, loss, gold_cost, rival_cost, gold_center, rival_center, rival_site}),
        cast_vote_at_step_zero: ($edge.history[0].loss > 0)
    }
}
