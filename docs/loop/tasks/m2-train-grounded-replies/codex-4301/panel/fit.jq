def require($ok; $problem): if $ok then . else error($problem) end;

require(length == $pairs * 2 and ([.[].row_index] == [range($pairs * 2)]);
    "wrong canonical request count or row order") |
group_by(.pair_id) |
require(length == $pairs; "wrong number of requested selector pairs") |
sort_by(map(.row_index) | min) |
to_entries[] |
.key as $pair | (.value | sort_by(.row_index)) as [$first,$later] |
require((.value | length) == 2 and $first.selector == "first" and
    $later.selector == "later" and $later.row_index == $first.row_index + 1 and
    ($first.sign == 1 or $first.sign == -1) and $first.sign == $later.sign and
    $first.governor == $later.governor and $first.opposite_governor == $later.opposite_governor and
    $first.opposite == $later.opposite and $first.heading == $later.heading and
    $first.requested == $later.rival and $first.rival == $later.requested and
    $first.requested.index != $first.rival.index and $first.case_id != $later.case_id;
    "pair lost the canonical first/later role or original addresses") |
[$first,$later][] |
[$pair,.row_index,.sign,.governor_word_index,.requested_site_index,.rival_site_index] | @tsv
