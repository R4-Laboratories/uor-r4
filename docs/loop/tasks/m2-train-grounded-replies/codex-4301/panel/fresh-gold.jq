def require($ok; $problem): if $ok then . else error($problem) end;

. as $row |
require((.requests | length) == 1; "fresh row has more than one signed selector") |
.requests[0] as $request |
(if $request.sign == 1 then 0 elif $request.sign == -1 then 1
 else error("invalid fresh selector sign") end) as $selected |
([$request.requested.index,$request.opposite.index,$request.heading.index] |
    if $selected == 0 then . else [.[1],.[0],.[2]] end) as $sites |
($sites | .[$selected] = $request.rival.index) as $swapped |
require($sites != $swapped and ($sites|unique|length) == 3 and ($swapped|unique|length) == 3 and
    any($row.coordinate_map.tuples[]; . == $sites) and
    any($row.coordinate_map.tuples[]; . == $swapped);
    "expected or swapped sensitivity tuple is not legal") |
{
    row_index, pair_id, case_id, sign:$request.sign, selector,
    governors:([$request.governor.index,$request.opposite_governor.index] |
        if $selected == 0 then . else reverse end), sites:$sites, swapped_sites:$swapped
}
