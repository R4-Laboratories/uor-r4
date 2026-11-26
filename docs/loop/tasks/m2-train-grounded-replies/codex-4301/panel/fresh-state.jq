def need($ok; $message): if $ok then . else error($message) end;

def rank($a; $b):
    if $a == null and $b == null then "neither"
    elif $a == null then "ineligible"
    elif $b == null then "unopposed"
    elif $a < $b then "win"
    elif $a > $b then "loss"
    else "tie" end;

def first_key($a; $b):
    rank($a;$b) as $relation |
    if $a==null or $b==null then {key:$relation}
    elif $relation=="tie" then {key:"tie"}
    else
        [range($a|length) as $i | select($a[$i]!=$b[$i]) | $i][0] as $i |
        {key:["lexical","transport","bank"][$i],
         requested:$a[$i],rival:$b[$i],requested_minus_rival:($a[$i]-$b[$i])}
    end;

def address($row; $kind; $i):
    $row.coordinate_map[$kind][$i] as $a |
    need($i >= 0 and $a != null; "observed ordinal is outside the frozen address map") |
    {i:$i, bytes:$a.bytes, bpe:$a.tokens,
     text:$row.text[$a.bytes.start:$a.bytes.end]};

def audit_address:
    {bytes:[.bytes.start,.bytes.end],tokens:[.bpe.start,.bpe.end]};

def strongest($row; $graded; $winner; $side):
    need($winner == null or ($graded.cost == $winner.best and
       ($graded.sites | map([.site.i,.words,.ties,.first_tuple])) ==
       ($winner.sites | map([.site,.governors,.ties,.tuple_ordinals[0]])));
       "physical grade and independently replayed strongest-third minima differ") |
    ($winner.sites // [$graded.sites[] | {site:.site.i,governors:.words,ties,
        tuple_ordinals:[.first_tuple]}]) as $candidates |
    $graded + {sites:[
        $candidates[] as $site |
        address($row;"sites";$site.site) as $address |
        [$site.governors[] | address($row;"words";.)] as $governors |
        need(($site.tuple_ordinals | length) > 0 and
           ($winner == null or (($address | audit_address) == $site.address and
            ($governors | map(audit_address)) == $site.governor_addresses));
           "independent strongest-third addresses do not map to the original prompt") |
        {site:$address, governors:$governors, ties:$site.ties,
         tuples:[ $site.tuple_ordinals[] as $ordinal |
             $row.coordinate_map.tuples[$ordinal] as $triple |
             need($triple[$side] == $site.site;
                "tied tuple does not contain its strongest third site in the signed slot") |
             {ordinal:$ordinal, sites:[$triple[] | address($row;"sites";.)]} ]}
    ],all_minimizing_tuple_ordinals:($winner != null)};

def local_third($row; $cost):
    [$cost.sites[] | select(.i != $cost.requested and .i != $cost.rival and
        .local != null and .fixed.best != null)] as $eligible |
    ($eligible | map(.local) | min) as $best |
    {cost:$best, eligible_sites:($eligible|length),
     sites:[$eligible[] | select(.local == $best) | address($row;"sites";.i)]};

def current_gap($gov; $noun; $order):
    if $order == "before" then $noun.bpe.start - $gov.bpe.end
    elif $order == "after" then $gov.bpe.start - $noun.bpe.end
    else error("unknown governor order") end;

need(($physical|length) == $expected and ($grade|length) == $expected and
    (($observed|length)==0 or ($observed[0].rows|length) == $expected) and ($raw|length) == $expected;
    "fresh physical, reference, grade and independent replay counts differ") |
range($expected) as $i |
$physical[$i] as $row | $grade[$i] as $g | $observed[0].rows[$i] as $a | $raw[$i] as $cost |
(if $g.sign == 1 then 0 elif $g.sign == -1 then 1 else error("invalid focus sign") end) as $side |
need([$g.row,$cost.row,$row.row_index] == [$i,$i,$i] and $g.sign == $cost.sign and
    $row.pair_id == $g.pair_id and $row.selector == $g.selector and $row.family == $g.family and
    $g.requested.i == $cost.requested and $g.rival.i == $cost.rival and
    $g.governor.i == $cost.word and ($a == null or
    ($a.row == $i and $g.sign == $a.sign and $row.case_id == $a.case_id and $g.pair_id == $a.pair_id and
     $g.family == $a.family and $g.selector == $a.selector and $g.source == $a.follows and
     $g.full == $a.full and $g.native.ties == $a.native_ties and
     ($g.requested | audit_address) == $a.requested_address and ($g.rival | audit_address) == $a.rival_address));
    "fresh result changed row identity, sign, original addresses or native verdict") |
current_gap($g.governor;$g.requested;$g.order) as $requested_gap |
current_gap($g.governor;$g.rival;$g.order) as $rival_gap |
(if $g.selector == "first" then [$requested_gap,$rival_gap] else [$rival_gap,$requested_gap] end) as $gap |
local_third($row;$cost) as $local |
$g + {
    case_id:$row.case_id, gap:$gap,
    opposite_governor:$row.requests[0].opposite_governor,
    opposite:$row.requests[0].opposite, heading:$row.requests[0].heading,
    local_third:$local,
    local_third_relation:rank($g.local.requested;$local.cost),
    third_fixed:strongest($row;$g.third_fixed;$a.fixed_third;$side),
    third_free:strongest($row;$g.third_free;$a.free_third;$side),
    free_target_vs_third:rank($g.requested_free;$g.third_free.cost),
    deciding_keys:{local_target_vs_same_spelling:first_key($g.local.requested;$g.local.rival),
         local_target_vs_third:first_key($g.local.requested;$local.cost),
         fixed_target_vs_same_spelling:first_key($g.joint.requested;$g.joint.rival),
         fixed_target_vs_fixed_third:first_key($g.joint.requested;$g.third_fixed.cost),
         fixed_target_vs_free_third:first_key($g.joint.requested;$g.third_free.cost),
         free_target_vs_free_third:first_key($g.requested_free;$g.third_free.cost)}
}
