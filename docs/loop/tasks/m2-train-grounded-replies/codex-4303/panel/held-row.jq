def require($ok; $reason): if $ok then . else error($reason) end;

def address($source; $axis; $i):
    if $i == null then null
    else $source.coordinate_map[$axis][$i] as $span |
        require($i >= 0 and $span != null; "ordinal outside current-user coordinate map") |
        {i:$i, bytes:[$span.bytes.start,$span.bytes.end],
         tokens:[$span.tokens.start,$span.tokens.end],
         word:$source.text[$span.bytes.start:$span.bytes.end]}
    end;

def checked_link($source; $link):
    address($source;"words";$link.word) as $w |
    address($source;"sites";$link.site) as $s |
    require($link.governor == ($w | if . == null then null else {bytes,tokens} end) and
        $link.noun == ($s | if . == null then null else {bytes,tokens} end);
        "native link original word or site address mismatch") |
    {governor:$w,noun:$s,eligible:$link.eligible,link_ties:$link.link_ties,best:$link.best};

def checked_tuple($source; $tuple):
    if $tuple.ordinal == null then
        require($tuple.ties != 1; "a unique native tuple lost its original ordinal") | null
    else
        [$tuple.sites[] | address($source;"sites";.)] as $sites |
        require($tuple.sites == $source.coordinate_map.tuples[$tuple.ordinal] and
            $tuple.addresses == ($sites|map({bytes,tokens})); "native complete tuple original addresses changed") |
        {ordinal:$tuple.ordinal,sites:$sites,best:$tuple.best,ties:$tuple.ties}
    end;

def compare($target; $third; $keys):
    if $target == null then {status:"missing-target"}
    elif $third == null then {status:"no-third"}
    else
        require(($target|length) == ($third|length) and ($third|length) == ($keys|length);
            "cost components differ in arity") |
        [range($keys|length) as $j | $target[$j] - $third[$j]] as $delta |
        ([range($keys|length) | select($delta[.] != 0)] | .[0]) as $first |
        {status:(if $target < $third then "win" elif $target > $third then "loss" else "tie" end),
         first_key:(if $first == null then "tie" else $keys[$first] end),
         target:$target, third:$third, target_minus_third:$delta}
    end;

def witness($source;$site;$what;$side;$pinned_word):
    $site[$what] as $group |
    $source.coordinate_map.tuples[$group.first_tuple] as $tuple |
    require($group.first_tuple != null and $tuple[$side] == $site.i;
        "strongest third complete tuple fails its signed slot") |
    {site:address($source;"sites";$site.i),ties:$group.ties,first_tuple:$group.first_tuple,
     tuple_sites:([$tuple[] | address($source;"sites";.)]),
     governors:([(if $what == "fixed" then $pinned_word else $site.governors[] end) |
        address($source;"words";.)])};

def strongest($source;$candidates;$what;$side;$pinned_word):
    ($candidates | map(.[$what].best) | min) as $best |
    {cost:$best,eligible_sites:($candidates|length),
     sites:[$candidates[] | select(.[$what].best == $best) |
        witness($source;.;$what;$side;$pinned_word)]};

def lexical($probe; $word; $sign):
    $probe.words[$word] as $read |
    require(($read|length) >= 2 and all($read[0:2][]; type == "number");
        "missing physical lexical costs") |
    {word:$word, a:$read[0],e:$read[1],
     status:(if $read[0] == $read[1] then "tie"
       elif ($sign == 1 and $read[0] < $read[1]) or ($sign == -1 and $read[1] < $read[0])
       then "own" else "opposite" end)};

def lexical_changes($source;$current;$old;$gold):
    if $old == null then []
    else
        require($old.row == $current.row and ($old.words|length) == ($current.words|length);
            "baseline lexical observation row differs") |
        [range($current.words|length) as $i |
            lexical($old;$i;1).status as $before |
            lexical($current;$i;1).status as $after |
            select($before != $after) |
            {word:address($source;"words";$i),
             role:(if $gold.governors[0] == $i then "affirmative"
                   elif $gold.governors[1] == $i then "excluded"
                   else "other" end),
             before:$before,after:$after}]
    end;

def failure($ties;$word_match;$site_match;$same_spelling;$full):
    if $ties == 0 then "no-tuple"
    elif $ties != 1 then "tuple-tie"
    elif $word_match and $site_match then
        if $full then "exact-full" else "signed-only" end
    elif $site_match then "wrong-governor"
    elif $word_match then
        if $same_spelling then "same-spelling-wrong-address" else "third-site" end
    elif $same_spelling then "wrong-governor-and-same-spelling-address"
    else "wrong-governor-and-third-site"
    end;

require(($geometry|length) == 256 and ($meta|length) == 256 and ($gold|length) == 256 and
    ($registered|length) == 256 and ($reads|length) == 256 and
    ($costs|length) == 256 and ($probe|length) == 256 and
    (($baseline_probe|length) == 0 or ($baseline_probe|length) == 256) and
    (($review|length) == 0 or ($review[0].rows|length) == 256);
    "fresh packet or physical captures are incomplete") |
range(256) as $i |
$geometry[$i] as $source | $meta[$i] as $m | $gold[$i] as $g | $registered[$i] as $r |
$reads[$i] as $n | $costs[$i] as $c | $probe[$i] as $p | $review[0].rows[$i] as $independent |
require([$source.row_index,$m.row_index,$g.row_index,$r.row_index,$n.row,$c.row,$p.row] ==
    [$i,$i,$i,$i,$i,$i,$i] and $source.case_id == $m.case_id and $m.case_id == $r.case_id and
    $m.case_id == $g.case_id and $m.pair_id == $r.pair_id and
    [$m.sign,$m.governor_word_index,$m.requested_site_index,$m.rival_site_index] ==
    [$c.sign,$c.word,$c.requested,$c.rival] and
    ($p.words|length) == ($source.coordinate_map.words|length) and
    ([$c.sites[].i] == [range($source.coordinate_map.sites|length)]);
    "physical row or original registered identity differs") |
(if $m.sign == 1 then 0 elif $m.sign == -1 then 1 else error("unknown signed slot") end) as $side |
[$n.links[] | checked_link($source;.)] as $links |
checked_tuple($source;$n.tuple) as $tuple |
($links[$side].governor.i == $c.word) as $word_match |
($links[$side].noun.i == $c.requested) as $site_match |
($links[$side].noun.i == $c.rival) as $wrong_same |
($n.tuple.ties == 1 and $word_match and $site_match
     and $tuple.sites[$side].i == $c.requested) as $source_match |
($source_match and ($links|map(.governor.i)) == $g.governors
     and ($links|map(.noun.i)) == $g.sites[0:2] and
     (($tuple.sites // [])|map(.i)) == $g.sites) as $full |
$c.sites[$c.requested] as $target |
$c.sites[$c.rival] as $rival |
[$c.sites[] | select(.i != $c.requested and .i != $c.rival)] as $others |
[$others[] | select(.fixed.best != null)] as $fixed_candidates |
[$others[] | select(.free.best != null)] as $free_candidates |
strongest($source;$fixed_candidates;"fixed";$side;$c.word) as $fixed |
strongest($source;$free_candidates;"free";$side;$c.word) as $free |
($fixed_candidates | map(.local) | min) as $local |
lexical($p;$g.governors[0];1) as $affirmative |
lexical($p;$g.governors[1];-1) as $excluded |
[$affirmative,$excluded][$side] as $focus |
require(($target.local != null) == ($focus.status == "own");
    "requested link and strict physical lexical sign disagree") |
require($independent == null or
    ($independent.case_id == $m.case_id and $independent.follows == $source_match and
     $independent.full == $full and $independent.native_ties == $n.tuple.ties and
     $independent.fixed_third.best == $fixed.cost and $independent.free_third.best == $free.cost);
    "fresh raw capture disagrees with the independent original-address replay") |
{
    row:$i, case_id:$m.case_id, pair_id:$m.pair_id, selector:$m.selector, family:$m.family,
    fact:$m.fact, order:$m.governor_order, sign:$m.sign, affirmative_verb:$r.affirmative.text,
    contested_verb:$m.governor.text, requested_role:$r.requested_role,
    requested_horizon:$r.requested_horizon, gap_bpe:$r.requested_gap_bpe,
    expected_governor:address($source;"words";$c.word), requested:address($source;"sites";$c.requested),
    same_spelling_rival:address($source;"sites";$c.rival),
    native:$links[$side], native_tuple:$tuple, tuple_ties:$n.tuple.ties,
    source:$source_match, full:$full,
    status:failure($n.tuple.ties;$word_match;$site_match;$wrong_same;$full),
    lexical:{affirmative:$affirmative,excluded:$excluded,contested:$focus},
    lexical_membership_changes:lexical_changes($source;$p;$baseline_probe[$i];$g),
    local_eligible:($target.local != null), completion_eligible:($target.fixed.best != null),
    same_spelling:compare($target.local;$rival.local;["lexical","transport"]),
    local_third:compare($target.local;$local;["lexical","transport"]),
    local_third_sites:[$fixed_candidates[] | select(.local == $local) | address($source;"sites";.i)],
    pinned_third:compare($target.fixed.best;$fixed.cost;["lexical","transport","bank"]),
    pinned_target_free_third:compare($target.fixed.best;$free.cost;["lexical","transport","bank"]),
    both_free_third:compare($target.free.best;$free.cost;["lexical","transport","bank"]),
    strongest_fixed:$fixed, strongest_free:$free,
    pinned_target_cost:$target.fixed.best, free_target_cost:$target.free.best
}
