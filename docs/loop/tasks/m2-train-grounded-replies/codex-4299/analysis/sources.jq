def need($ok; $why): if $ok then . else error($why) end;
def role($sign): ["affirmative","excluded"][$sign];
def span($pair): if $pair == null then null else {start:$pair[0],end:$pair[1]} end;
def addr($bytes; $bpe): if $bytes == null then null else {bytes:$bytes,bpe:$bpe} end;
def raw_addr: if . == null then null else addr(span(.bytes); span(.tokens)) end;
def joined_addr: if . == null then null else addr(.bytes; .tokens) end;
def gold_noun($ref; $i):
    $ref.native_scope[$i] | addr({start:(.bytes.start-1),end:(.bytes.end-1)}; .tokens);
def gold_governor($ref; $i):
    $ref.governors[role($i)] | addr(.bytes; span(.tokens));
def observed($row; $i):
    if $row.arm.selection == null then null else addr($row.arm.selection[$i]; $row.arm.tokens[$i]) end;
def spell($text; $address):
    if $address == null then null
    else need(($text|length)==($text|utf8bytelength) and $address.bytes.start>=0 and
              $address.bytes.start<$address.bytes.end and $address.bytes.end<=($text|length);
              "source witness needs a valid ASCII byte span") |
         $address+{text:$text[$address.bytes.start:$address.bytes.end]}
    end;

def status($row):
    if $row.arm.full then "full"
    elif $row.arm.eligible == 0 then "no_eligible"
    elif $row.arm.ties != 1 then "tie"
    elif any($row.arm.signs[]; .governor and .noun) or $row.arm.earlier then "partial"
    else "wrong" end;

def census($items):
    reduce $items[] as $item ({full:0,partial:0,no_eligible:0,tie:0,wrong:0}; .[$item] += 1);
def pair_status($pair):
    if all($pair[];.=="full") then "full"
    elif any($pair[];.=="full" or .=="partial") then "partial"
    elif any($pair[];.=="no_eligible") then "no_eligible"
    elif any($pair[];.=="tie") then "tie"
    else "wrong" end;
def transitions($rows):
    {rows:($rows|length),before:census([$rows[].before]),after:census([$rows[].after]),
     full_gained:([$rows[]|select(.before!="full" and .after=="full")]|length),
     full_lost:([$rows[]|select(.before=="full" and .after!="full")]|length),
     partial_gained:([$rows[]|select(.before!="partial" and .before!="full" and .after=="partial")]|length),
     partial_lost:([$rows[]|select(.before=="partial" and .after!="partial" and .after!="full")]|length),
     tie_entered:([$rows[]|select(.before!="tie" and .after=="tie")]|length),
     tie_exited:([$rows[]|select(.before=="tie" and .after!="tie")]|length),
     no_eligible_entered:([$rows[]|select(.before!="no_eligible" and .after=="no_eligible")]|length),
     no_eligible_exited:([$rows[]|select(.before=="no_eligible" and .after!="no_eligible")]|length)};
def by_family($rows):
    $rows|unique_by(.pair)|group_by(.family)|map({family:.[0].family,pairs:(map(.pair))});

def classify_sources($row):
    [$row.arm.governors[0]|joined_addr] + [observed($row;0)] +
    [$row.arm.governors[1]|joined_addr] + [observed($row;1), observed($row;2)];
def source_label($which):
    ["affirmative_governor","affirmative_noun","excluded_governor","excluded_noun","heading"][$which];
def case_info($row; $ref):
    {ordinal:$row.ordinal,pair:$row.pair_id,case:$row.case_id,prompt_sha256:$row.prompt_sha256,
     family:$row.family,fact:$ref.fact,admission:$row.admission,distance:$row.distance};

def attachment($raw; $sign; $text):
    $raw.links[$sign] |
    {governor:spell($text;(.governor|raw_addr)),noun:spell($text;(.noun|raw_addr)),
     word,site,best_lexical_transport:.best,rival_lexical_transport:.rival,word_ties,site_ties,link_ties};
def witness($i):
    $before[$i] as $a | $after[$i] as $b | $reference[0][$i] as $ref | $ref.turns[1] as $text |
    classify_sources($a) as $old | classify_sources($b) as $new |
    [gold_governor($ref;0),gold_noun($ref;0),gold_governor($ref;1),gold_noun($ref;1),gold_noun($ref;2)] as $gold |
    ([range(5) | select($old[.] != $new[.])] | .[0]) as $diff |
    (case_info($a;$ref) + {
        first_changed_source:(if $diff==null then null else
            {role:source_label($diff),expected:spell($text;$gold[$diff]),
             before:spell($text;$old[$diff]),after:spell($text;$new[$diff])} end),
        original_nouns:([range(3) | spell($text;gold_noun($ref;.))]),
        before:{status:status($a),first_source_error:$a.arm.first_source_error,ties:$a.arm.ties,eligible:$a.arm.eligible,
                tuple_best:$a.arm.best,tuple_rival:$a.arm.rival,nouns:([range(3) | spell($text;observed($a;.))]),
                attachments:[range(2) | attachment($before_raw[$i];.;$text)]},
        after:{status:status($b),first_source_error:$b.arm.first_source_error,ties:$b.arm.ties,eligible:$b.arm.eligible,
               tuple_best:$b.arm.best,tuple_rival:$b.arm.rival,nouns:([range(3) | spell($text;observed($b;.))]),
               attachments:[range(2) | attachment($after_raw[$i];.;$text)]}
    });

def capture_pair($group):
    {pair:$group[0].pair,family:$group[0].family,
     before:pair_status([$group[].before]),after:pair_status([$group[].after]),
     any_tie_before:any($group[];.before=="tie"),any_tie_after:any($group[];.after=="tie"),
     any_no_eligible_before:any($group[];.before=="no_eligible"),any_no_eligible_after:any($group[];.after=="no_eligible"),
     entries:$group};
def first_or_null: .[0] as $i | if $i==null then null else witness($i) end;
def compare_sources($records):
    ($records|group_by(.pair)|map(capture_pair(.))) as $pairs |
    {rows:transitions($records),pairs:(transitions($pairs)|.pairs=.rows|del(.rows)),
     pairs_with_any_tie:{before:([$pairs[]|select(.any_tie_before)]|length),after:([$pairs[]|select(.any_tie_after)]|length)},
     pairs_with_any_no_eligible:{before:([$pairs[]|select(.any_no_eligible_before)]|length),after:([$pairs[]|select(.any_no_eligible_after)]|length)},
     address_changes:([$records[]|select(.changed)]|length),
     first_differing_source:([$records[]|select(.changed)|.ordinal]|first_or_null),
     first_before_error:([$records[]|select(.before!="full")|.ordinal]|first_or_null),
     first_after_error:([$records[]|select(.after!="full")|.ordinal]|first_or_null)};

def marked($rows):
    [range($rows) as $i | $before[$i] as $a | $after[$i] as $b |
        {ordinal:$i,pair:$a.pair_id,family:$a.family,distance:$a.distance,
         before:status($a),after:status($b),changed:(classify_sources($a)!=classify_sources($b))}];

def same_source($row; $raw):
    $row.ordinal==$raw.row and $row.arm.ties==$raw.tuple.ties and $row.arm.eligible==$raw.tuple.eligible and
    $row.arm.best==$raw.tuple.best and $row.arm.rival==$raw.tuple.rival and
    ([$row.arm.governors[] | joined_addr] == [$raw.links[].governor | raw_addr]) and
    ((if $raw.tuple.addresses==null then [null,null,null] else $raw.tuple.addresses|map(raw_addr) end) ==
        [range(3) | observed($row;.)]);
def valid($rows):
    ($reference[0]|length)==$rows and ($before|length)==$rows and ($after|length)==$rows and
    ($before_raw|length)==$rows and ($after_raw|length)==$rows and
    all(range($rows); . as $i | $before[$i] as $a | $after[$i] as $b | $reference[0][$i] as $ref |
        $i==$a.ordinal and $i==$b.ordinal and [$a.case_id,$a.pair_id,$a.prompt_sha256,$a.family,$a.distance,$a.admission] ==
        [$b.case_id,$b.pair_id,$b.prompt_sha256,$b.family,$b.distance,$b.admission] and
        [$a.case_id,$a.pair_id,$a.prompt_sha256,$a.family,$a.distance,$a.admission] ==
        [$ref.id,$ref.pair_id,$ref.structural_prompt_sha256,$ref.family,$ref.distance,$ref.admission] and
        [$a.gold,$a.gold_tokens]==[$b.gold,$b.gold_tokens] and
        [range(3) | gold_noun($ref;.)] == [range(3) | addr($a.gold[.];$a.gold_tokens[.])] and
        same_source($a;$before_raw[$i]) and same_source($b;$after_raw[$i]));

def selector_observation($row; $i):
    $reference[0][$row.ordinal] as $ref | observed($row;$i) as $seen |
    {seen:$seen,expected:gold_noun($ref;$i),correct:($seen==gold_noun($ref;$i))};
def selector_read($pair; $sign):
    ($pair[0]|selector_observation(.;$sign)) as $first | ($pair[1]|selector_observation(.;$sign)) as $later |
    ($pair[0]|selector_observation(.;1-$sign)) as $opposite_first |
    ($pair[1]|selector_observation(.;1-$sign)) as $opposite_later |
    {first:$first,later:$later,opposite_first:$opposite_first,opposite_later:$opposite_later,
     distinct:($first.seen!=null and $later.seen!=null and $first.seen!=$later.seen),
     correct:($first.correct and $later.correct)};
def selector_example($compare; $key; $sign):
    $compare.seen[$key] as $read |
    {pair:$compare.pair,role:role($sign),opposite_role:role(1-$sign),
     selections:(["first","later"]|map(. as $selector |
        (if $selector=="first" then $compare.first else $compare.later end) as $i |
        $reference[0][$i] as $ref | $ref.turns[1] as $text |
        {selector:$selector,case:$ref.id,family:$ref.family,fact:$ref.fact,
         target:{gold:spell($text;$read[$selector].expected),chosen:spell($text;$read[$selector].seen)},
         opposite:{gold:spell($text;$read["opposite_"+$selector].expected),
                   chosen:spell($text;$read["opposite_"+$selector].seen),correct:$read["opposite_"+$selector].correct}}))};
def selector_counts($moves; $key; $sign):
    [$moves[]|select(.seen[$key].correct)] as $success |
    {first_correct:([$moves[]|select(.seen[$key].first.correct)]|length),
     later_correct:([$moves[]|select(.seen[$key].later.correct)]|length),
     distinct_original_occurrences:([$moves[]|select(.seen[$key].distinct)]|length),
     correct_flips:($success|length),
     opposite_on_correct_flips:{first_correct:([$success[]|select(.seen[$key].opposite_first.correct)]|length),
         later_correct:([$success[]|select(.seen[$key].opposite_later.correct)]|length),
         both_correct:([$success[]|select(.seen[$key].opposite_first.correct and .seen[$key].opposite_later.correct)]|length),
         both_wrong:([$success[]|select((.seen[$key].opposite_first.correct or .seen[$key].opposite_later.correct)|not)]|length),
         examples:([$success[] | selector_example(.;$key;$sign)])}};
def selectors($sign):
    ($before|group_by(.pair_id)|map(sort_by(.selector))) as $pairs |
    need(($pairs|length)==32 and all($pairs[];map(.selector)==["first","later"]);
            "selector panels did not preserve paired first and later wording") |
    [$pairs[]|. as $a | ($a|map($after[.ordinal])) as $b |
        select(($a[0]|selector_observation(.;$sign)|.expected)!=($a[1]|selector_observation(.;$sign)|.expected)) |
        {pair:$a[0].pair_id,first:$a[0].ordinal,later:$a[1].ordinal,
         seen:{before:selector_read($a;$sign),after:selector_read($b;$sign)}}] as $moves |
    {role:role($sign),gold_changes:($moves|length),before:selector_counts($moves;"before";$sign),
     after:selector_counts($moves;"after";$sign),gained:([$moves[]|select(.seen.after.correct and (.seen.before.correct|not))|.pair]),
     lost:([$moves[]|select(.seen.before.correct and (.seen.after.correct|not))|.pair])};

def training_detail($pairs):
    $protected[0].protected as $reference_near | $protected[0].far as $reference_far |
    [$pairs[]|select(.entries[0].distance=="near" and (.pair as $id|$reference_near|index($id))!=null)] as $near |
    [$near[]|select(.after!="full")|.entries[]|select(.after!="full")|.ordinal] as $post_failures |
    [$near[]|select(.before=="full" and .after!="full")|.entries[]|select(.after!="full")|.ordinal] as $new_failures |
    [$near[]|select(.before!="full" and .after=="full")|.entries[]|select(.before!="full")|.ordinal] as $recoveries |
    [$pairs[]|select(.entries[0].distance=="far" and .after=="full")] as $far |
    [$far[]|select(.before!="full")|.entries[]|select(.before!="full")|.ordinal] as $new_far |
    [$pairs[]|select(.entries[0].distance=="far" and .after!="full")|.entries[]|select(.after!="full")] as $far_misses |
    need(($near|length)==24 and ($reference_far|length)==40; "protected precedence changed") |
    {original_protected:($near|length),
     missed_after:by_family([$near[]|select(.after!="full")]),
     newly_lost:by_family([$near[]|select(.before=="full" and .after!="full")]),
     already_missed:by_family([$near[]|select(.before!="full" and .after!="full")]),
     recovered:by_family([$near[]|select(.before!="full" and .after=="full")]),
     first_protected_failure:($post_failures|sort|first_or_null),
     first_new_protected_failure:($new_failures|sort|first_or_null),
     first_protected_recovery:($recoveries|sort|first_or_null),
     far_wins:by_family($far),far_wins_gained:by_family([$far[]|select(.before!="full")]),
     far_wins_beyond_protected_reference:by_family([$far[]|select((.pair as $id|$reference_far|index($id))==null)]),
     first_far_gain:($new_far|sort|first_or_null),
     residual_far_example:($far_misses|sort_by([if .after=="tie" then 0 elif .after=="partial" then 1 else 2 end,.ordinal])|
                           map(.ordinal)|first_or_null)};

(if $domain=="contrast" then 64 else 256 end) as $count |
need(valid($count);"released original addresses, prompt identity or tuple trace disagree") |
marked($count) as $records |
{geometry:$geometry,from:$from,to:$to,
 strata:($records|group_by(.distance)|map(. as $r|{distance:$r[0].distance}+compare_sources($r)))} +
(if $domain=="contrast" then
    need(all(range($count);. as $i|$before[$i].selector==$after[$i].selector and
         $before[$i].selector==$reference[0][$i].contrast.selector); "selector identity changed") |
    {signed_selectors:[range(2)|selectors(.)]}
 elif $domain=="mixed" then
    {training:($records|group_by(.pair)|map(capture_pair(.))|training_detail(.))}
 else {} end)
