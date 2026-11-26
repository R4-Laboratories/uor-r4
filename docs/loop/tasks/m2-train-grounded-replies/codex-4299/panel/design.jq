def need($ok; $message): if $ok then . else error($message) end;
def count_by(f): group_by(f) | map({key:(.[0]|f),value:length}) | from_entries;
def token_gap($governor; $noun; $direction):
    if $direction == "before" then $noun.start - $governor[1]
    elif $direction == "after" then $governor[0] - $noun.end
    else error("unknown governor direction") end;
def rival_geometry:
    .governors.referent.tokens as [$start,$end] | .neutral_rival.tokens as $rival |
    .governors.referent.gap as $gold |
    ([$rival.start - $end, $start - $rival.end, 0] | max) as $gap |
    need($gap > 0; "the rival overlaps its signed governor") |
    if $gap < $gold then "nearer"
    elif $gap > $gold then "farther" else "equal" end;

need(all(.[];
    . as $row |
    all(range(2); . as $sign |
        $row.governors[["affirmative","excluded"][$sign]] as $g |
        $row.native_scope[$sign].tokens as $noun |
        token_gap($g.tokens; $noun; $row.governor_order) as $gap |
        $g.native_noun == $noun and $g.gap == $gap and
        (if $row.distance == "near" then $gap >= 0 and $gap < 6
         elif $row.distance == "far" then $gap >= 8 else false end)));
    "a signed gold span or token gap differs from the original address") |
group_by(.family) |
need(length == 4; "expected four independently authored wording families") |
if $cohort == "held" then
    map(need(length == 64 and
        (group_by([.fact,.distance,.governor_order,.placement_name,.requested_role,.admission]) | length) == 64 and
        (group_by(.fact) | length) == 2 and
        (group_by([.fact,.requested_horizon]) | length) == 2;
        "primary family is missing an independent paired factor") |
        {family:.[0].family, pairs:(map(.pair_id)|unique|length),
         strata:(group_by(.distance)|map(
             count_by(rival_geometry) as $rivals |
             need((if .[0].distance == "far" then $rivals == {nearer:16,farther:16}
                   else $rivals == {farther:32} end);
                 "unexpected nearest-rival balance at the original governor") |
             {distance:.[0].distance, rows:length, rival_by_bpe_gap:$rivals}))})
elif $cohort == "contrast" then
    map(need(length == 16 and
        (group_by([.fact,.governor_order,.admission,.contrast.selector]) | length) == 16 and
        all(.[]; .distance == "far") and
        (group_by(.pair_id) | length) == 8;
        "selector family is missing a sign, direction, fact or selector peer") |
        need(all(group_by([.governor_order,.admission])[];
            (count_by(rival_geometry) == {nearer:2,farther:2}) and
            (count_by(.contrast.selector) == {first:2,later:2}));
            "selector pairs failed to exchange the nearer and farther original occurrences") |
        {family:.[0].family,pairs:8, strata:(group_by([.governor_order,.admission])|
          map({direction:.[0].governor_order,admission:.[0].admission,
               pairs:(map(.pair_id)|unique|length),
               rival_by_bpe_gap:count_by(rival_geometry)}))})
else error("unknown cohort") end
