def require($ok; $problem): if $ok then . else error($problem) end;

def original_words($row; $source):
    [$row.coordinate_map.words | to_entries[] |
        .value as $span |
        {i:.key, bytes:$span.bytes, bpe:$span.tokens,
         text:$row.text[$span.bytes.start:$span.bytes.end],
         ids:$source.prompt[$span.tokens.start:$span.tokens.end]}];

def disjoint($a; $b):
    $a.end <= $b.start or $b.end <= $a.start;

def gap($order; $word; $noun):
    if $order == "before" then $noun.tokens.start - $word.bpe.end
    elif $order == "after" then $word.bpe.start - $noun.tokens.end
    else error("unknown governor order")
    end;

def ordinal_word($words; $ref; $role):
    $ref.governors[$role] as $from |
    [$words[] |
        select(.bytes == $from.bytes and .bpe == {start:$from.tokens[0],end:$from.tokens[1]})] |
    require(length == 1; "governor did not resolve to one original word") | .[0];

def expected_noun($row; $ref; $word; $site; $role):
    $row.coordinate_map.sites[$site] as $from |
    $ref.governors[$role] as $governor |
    require($from.bytes == $governor.noun
        and $from.tokens == $governor.native_noun
        and ($from.bytes | disjoint($word.bytes; .))
        and ($from.tokens | disjoint($word.bpe; .));
        "word/source overlap or wrong original source") |
    $from + {i:$site, text:$row.text[$from.bytes.start:$from.bytes.end]};

def third_sites($row; $cut; $word; $requested; $rival):
    [$row.coordinate_map.tuples[] | .[$cut]] | unique |
    map(. as $i |
        select($i != $requested and $i != $rival) |
        $row.coordinate_map.sites[$i] |
        select(disjoint(.bytes; $word.bytes) and disjoint(.tokens; $word.bpe)) |
        {i:$i,bytes,tokens,text:$row.text[.bytes.start:.bytes.end]});

$refs[0] as $ref |
require(($ref|length) == ($rows|length) and ($rows|length) == length;
    "reference/row/gold count mismatch") |
to_entries[] |
.key as $i | .value as $gold | $rows[$i] as $row | $ref[$i] as $source |
require($gold.row_index == $i and $row.row_index == $i and
    $gold.case_id == $row.case_id and $gold.case_id == $source.id;
    "row identity mismatch") |
original_words($row;$source) as $words |
ordinal_word($words;$source;"affirmative") as $affirmative |
ordinal_word($words;$source;"excluded") as $excluded |
require([$affirmative.i,$excluded.i] == $gold.governors
    and ($affirmative.text == "retain" or $affirmative.text == "accept")
    and $excluded.text != "retain" and $excluded.text != "accept"
    and [$words[] | select(.text == "retain" or .text == "accept")] == [$affirmative];
    "the only literal retain/accept word must be the registered affirmative governor") |
expected_noun($row;$source;$affirmative;$gold.sites[0];"affirmative") as $affirmed |
expected_noun($row;$source;$excluded;$gold.sites[1];"excluded") as $excluded_noun |
(if $gold.sign == 1 then 0 elif $gold.sign == -1 then 1 else error("bad sign") end) as $cut |
[$affirmative,$excluded][$cut] as $word |
third_sites($row;$cut;$word;$gold.sites[$cut];$gold.swapped_sites[$cut]) as $third |
require(($third|length) > 0; "signed slot has no structurally legal third site") |
[$affirmed,$excluded_noun][$cut] as $noun |
gap($row.governor_order;$word;$noun) as $gap |
require($gap == $source.governors.referent.gap; "signed source gap differs from BPE tape") |
{
    row_index:$i, case_id:$row.case_id, pair_id:$row.pair_id, family:$row.family,
    fact:$row.fact, order:$row.governor_order, selector:$row.selector, sign:$gold.sign,
    requested_role:$source.requested_role, requested_horizon:$source.requested_horizon,
    affirmative:$affirmative, excluded:$excluded, requested_word:$word, requested_noun:$noun,
    requested_gap_bpe:$gap, signed_third_geometry:$third
}
