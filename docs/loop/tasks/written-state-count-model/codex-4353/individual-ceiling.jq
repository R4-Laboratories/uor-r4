# jq -n --arg pool POOL --slurpfile ranks held-ranks.jsonl -f held-feasibility.jq
# Every different-address rival concedes at least 65,535 * 2^32 Q48
# when the row's own gold addresses are maximized and all others minimized.
def require($ok; $why): if $ok then . else error($why) end;
def count(f): map(select(f))|length;
(65535 * 4294967296) as $limit
| INDEX($ranks[] | select(.arm=="zero"); .id) as $z
| INDEX($ranks[] | select(.arm=="geometry-435300"); .id) as $g1
| INDEX($ranks[] | select(.arm=="geometry-435301"); .id) as $g2
| require(($z|length)>0 and ($z|keys)==($g1|keys) and ($z|keys)==($g2|keys)
    and all(["zero","geometry-435300","geometry-435301"][]; . as $arm |
        ($ranks|count(.arm==$arm))==($z|length)); "held matrices differ")
| [($z|keys[]) as $id | $z[$id] as $base | $g1[$id] as $a | $g2[$id] as $b
    | [$base,$a,$b] as $group
    | require(all($group[]; .gold_lexical_id==$base.gold_lexical_id
        and .reference_lexical_offset==$base.reference_lexical_offset
        and .score.gold_addresses==$base.score.gold_addresses
        and .score.carry_code==$base.score.carry_code
        and .score.gold_prior_q48==$base.score.gold_prior_q48
        and .score.frozen_token==$base.score.frozen_token
        and .score.gold_prior_rank==$base.score.gold_prior_rank
        and .score.strongest_alias==$base.score.strongest_alias
        and .score.alias_blocked==$base.score.alias_blocked);
        "geometry changed the prior, gold address, or invariant alias")
    | require($base.score.token==$base.score.frozen_token
        and $base.score.score_q48==$base.score.prior_q48
        and $base.score.gold_score_q48==$base.score.gold_prior_q48
        and $base.score.prior_q48>=$base.score.gold_prior_q48
        and ($base.score.alias_blocked==0 or ($base.score.strongest_alias>0
            and $base.score.strongest_alias!=$base.gold_lexical_id
            and $base.score.gold_prior_rank>1));
        "invalid zero-weight winner or alias")
    | {id:$id, gold:$base.gold_lexical_id, alias:$base.score.strongest_alias,
        blocked:($base.score.alias_blocked!=0),
        gap:($base.score.prior_q48-$base.score.gold_prior_q48)}
  ] as $rows
| {pool:$pool,rows:($rows|length),threshold_q48:$limit,
   min_gap_q48:($rows|map(.gap)|min),max_gap_q48:($rows|map(.gap)|max),
   below_threshold:($rows|count(.gap<$limit)), equal_threshold:($rows|count(.gap==$limit)),
   above_threshold:($rows|count(.gap>$limit)), aliases_blocked:($rows|count(.blocked)),
   aliases_present:($rows|count(.alias!=0)),
   independently_attainable:($rows|count(.gap<$limit and (.blocked|not))),
   unresolved:($rows|count(.gap>=$limit and (.blocked|not)))}
