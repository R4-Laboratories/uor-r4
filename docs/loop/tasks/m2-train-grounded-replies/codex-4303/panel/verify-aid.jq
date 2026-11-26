def require($ok; $reason): if $ok then . else error($reason) end;

def unpack($bytes): [$bytes | scan("[0-9]+") | tonumber];

([length] + [.[] | .governors as [$a,$e] |
    (if $lane == "gold" then .sites else .swapped_sites end) as [$s,$t,$h] |
    $a,$s,$e,$t]) as $expected_links |
([length] + [.[] | .governors as [$a,$e] |
    (if $lane == "gold" then .sites else .swapped_sites end) as [$s,$t,$h] |
    $a,$e,(if .sign == 1 then $a,$s,$e,$t else $e,$t,$a,$s end)]) as $expected_aid |
require(unpack($links) == $expected_links and unpack($aid) == $expected_aid;
    "signed-link or aid original word/site sequence disagrees with the registered lane") |
true
