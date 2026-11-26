# Reduce the independently verified rows and the checksum-bound compact cost plane.
def check($condition; $reason): if $condition then . else error($reason) end;
($reads[0].rows | sort_by(.row)) as $observed |
($current | sort_by(.row)) as $costs |
check(($observed | length) == 64 and ($costs | length) == 64; "wrong fit row enrollment") |
[
    range(64) as $i | $observed[$i] as $row | $costs[$i] as $record |
    check($row.row == $i and $record.row == $i; "wrong fit row order") |
    $record.sites[$record.requested].local as $gold |
    $record.sites[$record.rival].local as $rival |
    check($gold == null or $rival == null or $gold[0] == $rival[0]; "one fixed governor changed lexical tier") |
    {pair:$row.pair_id,win:$row.follows,full:$row.full,third:($row.free_third.correct_governor_relation=="requested_cheaper"),
     margin:(if $gold != null and $rival != null then $rival[1] - $gold[1] else null end)}
] |
group_by(.pair) |
check(length == 32 and all(.[]; length == 2); "wrong fit pair enrollment") |
map(
    ([.[] | .margin | select(. != null) | [0, 1 - .] | max]) as $deficits |
    {id:.[0].pair,win:all(.[];.win),full:all(.[];.full),local:all(.[];.margin != null and .margin > 0),
     third:all(.[];.third),missing:(2 - ($deficits | length)),worst:($deficits | max // 0),sum:($deficits | add // 0)}
) as $pairs |
($regression[0].pairs | map(select(.follows).id) | sort) as $reg |
($near[0].protected_pairs | to_entries | map(select(.value).key) | sort) as $protected |
($pairs | map(select(.win).id) | sort) as $wins |
($pairs | map(select(.full).id) | sort) as $full |
{id:$state,ordinal:$ordinal,kind:$kind,train:$wins,train_full:$full,near:$protected,regression:$reg,
 local:([$pairs[] | select(.local)] | length), third:([$pairs[] | select(.third)] | length),
 missing:([$pairs[].missing] | add), worst:([$pairs[].worst] | add), sum:([$pairs[].sum] | add)}
