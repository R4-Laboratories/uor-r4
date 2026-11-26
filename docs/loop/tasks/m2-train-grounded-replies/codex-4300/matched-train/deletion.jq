def require($condition; $reason):
    if $condition then . else error($reason) end;
def keep($edits):
    to_entries | map(. as $entry | select($edits | index($entry.key) | not) | .value);

map(. as $edited |
    ($primary[0] | map(select(.family==$edited.family and .fact==$edited.fact and
        .admission==$edited.admission and .governor_order==$edited.governor_order and
        .contrast.selector==$edited.contrast.selector))) as $sources |
    require(($sources|length)==1; "edited prompt missing exactly one primary TRAIN prior") |
    $sources[0] as $source |
    $source.contrast.tokens[0] as $selector |
    [$selector - 8, $selector - 5, $selector + 8] as $indices |
    [$source.prompt[$indices[]]] as $deleted |
    require($edited.id != $source.id and
        ($source.turns[1] |
            sub(" put in the red file,"; " in the file,") |
            sub(", when the small board meets,"; ", when the board meets,")) == $edited.turns[1] and
        ($deleted == [572,677,563]) and
        ($source.prompt | keep($indices)) == $edited.prompt;
        "the edit was not exactly the registered three TRAIN word deletions") |
    {case_id:$edited.id, pair_id:$edited.pair_id, selector:$edited.contrast.selector,
     primary_case_id:$source.id, removed:[range(3) as $index |
        {word:["put","red","small"][$index],primary_bpe_index:$indices[$index],bpe_id:$deleted[$index]}]}
) |
require(length==8; "expected eight derived endpoints")
