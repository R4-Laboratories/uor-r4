def require($ok; $reason): if $ok then . else error($reason) end;

def address($source; $axis; $index):
    $source.coordinate_map[$axis][$index] as $span |
    {bytes:[$span.bytes.start,$span.bytes.end],tokens:[$span.tokens.start,$span.tokens.end]};

def verify($labels; $source; $physical; $expected):
    require($physical.row == $labels.row_index and $physical.arm == "correct"
        and $physical.tuple.ties == 1 and $physical.tuple.eligible == 1 and
        $physical.tuple.sites == $expected and
        $source.coordinate_map.tuples[$physical.tuple.ordinal] == $expected and
        $physical.tuple.addresses == [$expected[] | address($source;"sites";.)] and
        ($physical.links|map(.word)) == $labels.governors and
        ($physical.links|map(.site)) == $expected[0:2] and
        ($physical.links|map(.governor)) == [$labels.governors[] | address($source;"words";.)] and
        ($physical.links|map(.noun)) == [$expected[0:2][] | address($source;"sites";.)];
        "supplied-control capture disagrees with its exact original addresses");

require(($geometry|length) == 256 and ($gold|length) == 256 and
    ($correct|length) == 256 and ($swap|length) == 256;
    "incomplete supplied controls") |
[range(256) as $i |
    $geometry[$i] as $source | $gold[$i] as $labels |
    require($source.row_index == $i and $labels.row_index == $i and
        $source.case_id == $labels.case_id and $source.pair_id == $labels.pair_id;
        "supplied-control source/gold identity mismatch") |
    verify($labels;$source;$correct[$i];$labels.sites) |
    verify($labels;$source;$swap[$i];$labels.swapped_sites) |
    {row:$i, pair_id:$labels.pair_id, case_id:$labels.case_id,
     target:$labels.sites, swapped:$labels.swapped_sites, governor:$labels.governors}
] as $rows |
{
    gold_exact_original_rows:($rows|length),
    gold_exact_original_pairs:($rows|map(.pair_id)|unique|length),
    swapped_exact_wrong_original_rows:($rows|length),
    swapped_exact_wrong_original_pairs:($rows|map(.pair_id)|unique|length),
    first:$rows[0]
}
