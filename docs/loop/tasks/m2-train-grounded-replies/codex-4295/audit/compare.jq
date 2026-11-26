def require($condition; $why): if $condition then . else error($why) end;
def candidate: {site,word:.noun.surface,bytes:.noun.bytes,bpe:.noun.tokens,cost:.cost[1],center:.transport_center};
def site_at($joint; $index):
    $joint.tuple.addresses[$index] | {site:$joint.tuple.sites[$index],word:.surface,bytes,bpe:.tokens};
["neutral", "winner", "control"] as $arms |
$independent[0] as $audit | $production[0] as $grade |
require($audit.pass and ($witnesses|length) == 48 and ($grade.per_prompt|length) == 48; "incomplete independent/production grade") |
reduce range(48) as $ordinal (. ;
    $witnesses[$ordinal] as $w | $grade.per_prompt[$ordinal] as $g |
    require($w.row == $g.row and $w.case_id == $g.case_id and $w.pair_id == $g.pair_id and
        $w.admission == $g.admission and $w.failed_sign == $g.failed_sign and $w.family == $g.family;
        "prompt identity disagrees at ordinal \($ordinal)") |
    reduce $arms[] as $arm (. ;
        $w.arms[$arm] as $a | $g[$arm] as $p |
        require($a.complete == $p.full and $a.signed.gold_rank == $p.gold_rank and
            $a.signed.gold_ties == $p.gold_ties and ($a.signed.gold|candidate) == $g.gold and
            ($a.signed.winners|map(candidate)) == $p.signed_winners and
            ($a.signed.strongest_wrong|map(candidate)) == $p.strongest_remaining_wrong and
            $a.joint.tuple.ties == $p.joint_ties and
            (if $a.joint.tuple.sites == null then null else [range(3) as $i | site_at($a.joint; $i)] end) == $p.joint_selected and
            ($a.joint.links|map({word,governor:(.governor|if .==null then null else {bytes,tokens} end)})) == $p.joint_governors;
            "independent rank or address disagreement at row \($w.row) in \($arm)"))) |
reduce $arms[] as $arm (. ;
    $audit.scores[$arm] as $a | $grade.arm[$arm] as $p |
    require($a.all.exact_prompts == $p.exact_prompts and $a.all.complete_pairs == $p.complete_pairs and
        $a.all.one_prompt_pairs == $p.one_side and $a.all.pairs == $p.pairs and
        $a.signed_gold_unique == $p.gold_unique_first and $a.joint_ties == $p.joint_ties and
        ($a.gold_rank_histogram|to_entries|map({value:(.key|tonumber),n:.value})|sort_by(.value)) == $p.gold_ranks;
        "independent scope counts disagree in \($arm)")) |
{pass:true,prompts:($witnesses|length),comparisons:(($witnesses|length)*($arms|length)),pairs:$audit.enrolled_pairs,
  matched_pairs:$grade.matched_pairs, complete_pairs:($arms|map({key:.,value:$audit.scores[.].all.complete_pairs})|from_entries),
  production_and_independent_rank_tie_address_disagreements:0}
