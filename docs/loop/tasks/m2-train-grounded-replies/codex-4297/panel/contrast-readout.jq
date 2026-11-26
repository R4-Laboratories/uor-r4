def need($ok; $message): if $ok then . else error($message) end;
def raw($address):
    if $address==null then null else {bytes:[$address.bytes.start,$address.bytes.end],
        tokens:[$address.tokens.start,$address.tokens.end]} end;
def chosen($row; $role):
    if $row.arm.selection==null then null else
        raw({bytes:$row.arm.selection[$role],tokens:$row.arm.tokens[$role]}) end;
def expected($row; $role): raw({bytes:$row.gold[$role],tokens:$row.gold_tokens[$role]});
def word($text; $address):
    if $address==null then null
    else $text[$address.bytes[0]:$address.bytes[1]] end;
def flags($x):
    {pairs_with_no_selection:([$x[]|select(.abstention)]|length),
     unchanged_controlled_selection:([$x[]|select(.fixed)]|length),
     changed_controlled_selection:([$x[]|select(.changed)]|length),
     correct_controlled_flips:([$x[]|select(.correct_flip)]|length),
     changed_without_correct_flip:([$x[]|select(.changed and (.correct_flip|not))]|length),
     full_pairs:([$x[]|select(.full)]|length)};
def endpoint_counts($rows):
    {endpoints:($rows|length), both_governors_exact:([$rows[]|select(.both_governors)]|length),
     controlled_noun_exact:([$rows[]|select(.target_correct)]|length),
     captured_other_same_spelling_address:([$rows[]|select(.selected_foil)]|length),
     unchanged_noun_exact:([$rows[]|select(.other_correct)]|length),
     full_endpoints:([$rows[]|select(.full)]|length)};

need(($rows|length)==64 and ($device|length)==64 and ($ref[0]|length)==64
    and ($matched[0].records|length)==32; "incomplete sealed contrast inputs") |
[range(64) as $i |
    $rows[$i] as $row | $device[$i] as $trace | $ref[0][$i] as $reference |
    ($reference.turns[1]) as $text |
    ($row.admission=="take" | if . then 0 else 1 end) as $role |
    need($i==$row.ordinal and $i==$trace.row and $row.case_id==$reference.id
        and $row.prompt_sha256==$reference.structural_prompt_sha256 and $row.pair_id==$reference.pair_id
        and $row.selector==$reference.contrast.selector and $row.admission==$reference.admission
        and ($text|length)==($text|utf8bytelength)
        and all(range(3); . as $k | chosen($row;$k)==$trace.tuple.addresses[$k]);
        "selector identity or raw joint occurrence differs") |
    (expected($row;$role)) as $target |
    (chosen($row;$role)) as $selection |
    (raw($reference.neutral_rival)|.bytes|map(.-1)) as $foil |
    ($matched[0].records[($i/2|floor)].edges[$i%2]) as $witness |
    need($witness.noun_bytes==$target.bytes and $witness.noun_bpe==$target.tokens;
        "contrast target disagrees with the sealed matched bridge address") |
    {ordinal:$i,case_id:$row.case_id,pair:$row.pair_id,selector:$row.selector,
     role:(if $role==0 then "affirmed" else "excluded" end), direction:$reference.governor_order,
     selector_bpe:$reference.contrast.tokens,
     controlled_governor:$trace.links[$role].governor,
     target:$target,target_site:$witness.site_ordinal,
     selected:$selection,selected_site:$trace.tuple.sites[$role],
     selected_word:word($text;$selection),target_word:word($text;$target),
     selected_foil:($selection!=null and $selection.bytes==$foil),
     target_correct:($target==$selection),
     other_target:expected($row;1-$role),other_selected:chosen($row;1-$role),
     other_correct:(expected($row;1-$role)==chosen($row;1-$role)),
     heading_target:expected($row;2),heading_selected:chosen($row;2),
     both_governors:$row.arm.both_governors,full:$row.arm.full,
     joint_best:$trace.tuple.best,joint_rival:$trace.tuple.rival,
     joint_ties:$trace.tuple.ties,joint_eligible:$trace.tuple.eligible}] as $endpoints |
[$endpoints | group_by(.pair)[] | sort_by(.ordinal) |
    need(length==2 and (map(.selector)==["first","later"]) and
        .[0].role==.[1].role and .[0].direction==.[1].direction and
        .[0].target!=.[1].target; "contrast comparison moved the wrong stratum or target") |
    .[0] as $first | .[1] as $later |
    need(any($matched[0].records[];.pair_id==$first.pair and
        .first_id==$first.case_id and .later_id==$later.case_id);
        "pair does not match independently sealed geometry witness") |
    (any(.[];.selected==null)) as $abstained |
    {pair:$first.pair,role:$first.role,direction:$first.direction,first_ordinal:$first.ordinal,
        abstention:$abstained,fixed:(($abstained|not) and $first.selected==$later.selected),
        changed:(($abstained|not) and $first.selected!=$later.selected),
        correct_flip:all(.[];.target_correct),full:all(.[];.full),endpoints:.}
] | sort_by(.first_ordinal) as $pairs |
{arm:$arm,roles:[$pairs|group_by(.role)[] |
    . as $stratum | need(length==16; "selector role census changed") |
    {role:.[0].role,pairs:length,rows:endpoint_counts([$stratum[].endpoints[]]),
     results:flags($stratum),
     direction:[$stratum|group_by(.direction)[]|. as $d |
        {direction:.[0].direction,pairs:length,rows:endpoint_counts([$d[].endpoints[]]),results:flags($d)}]}],
 first_physical_pair:$pairs[0], first_correct_controlled_flip:([$pairs[]|select(.correct_flip)]|first // null),
 first_changed_without_correct_flip:([$pairs[]|select(.changed and (.correct_flip|not))]|first // null)}
