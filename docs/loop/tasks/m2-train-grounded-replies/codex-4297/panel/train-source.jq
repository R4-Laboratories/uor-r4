def need($ok; $message): if $ok then . else error($message) end;
def gold:
    .native_scope | map({bytes:[(.bytes.start-1),(.bytes.end-1)],
        tokens:[.tokens.start,.tokens.end]});
def attach($source):
    $source.tuple | {sites,addresses,best,rival,ties,eligible};

($summary[0].ordered_base_original_misses) as $lost |
need(length==256 and ($endpoint|length)==256 and ($ordered|length)==256; "incomplete TRAIN source trace") |
([ .[] | select(.pair_id as $id | $lost|index($id)) | .pair_id ]|first) as $pair |
[to_entries[] | select(.value.pair_id==$pair) | .key as $i | .value as $ref |
    $endpoint[$i] as $before | $ordered[$i] as $after | ($ref|gold) as $target |
    need($before.row==$i and $after.row==$i and $before.tuple.addresses==$target
        and $after.tuple.addresses!=$target and $before.tuple.ties==1 and $after.tuple.ties==1
        and ($before.links|map(.governor))==($after.links|map(.governor));
        "the expected exact-to-wrong ordered switch or governor identity disagrees") |
    {ordinal:$i,id:$ref.id,admission:$ref.admission,gold:$target,
     gold_gaps:[$ref.governors.affirmative.gap,$ref.governors.excluded.gap],
     governors:($before.links|map(.governor)),endpoint:attach($before),ordered:attach($after),
     changed_slots:([range(3) as $slot|select($before.tuple.addresses[$slot]!=$after.tuple.addresses[$slot])|$slot])}
] | need(length==2 and (map(.admission)|sort)==["defer","take"]; "original near pair changed") |
{pair:$pair,identical_base_state_sha256:$state,rows:.}
