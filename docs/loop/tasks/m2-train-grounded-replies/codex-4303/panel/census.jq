def require($ok; $problem): if $ok then . else error($problem) end;

def shape:
    [.order,.sign,.selector,.requested_role,.requested_horizon];

def span_variants:
    map([.bytes.start,.bytes.end,.bpe.start,.bpe.end]) | unique | length;

def side:
    {rows:length,
     selector_word_ordinals:(map(.requested_word.i)|unique),
     affirmative_word_ordinals:(map(.affirmative.i)|unique),
     affirmative_address_count:(map(.affirmative)|span_variants),
     site_ordinals:(map(.requested_noun.i)|unique),
     signed_gap_pairs:(group_by(.pair_id) | map(sort_by(.row_index)|map(.requested_gap_bpe)) | unique),
     min_structural_third_sites:(map(.signed_third_geometry|length) | min),
     max_structural_third_sites:(map(.signed_third_geometry|length) | max)};

require(length == $facts * 32 and (map(.case_id)|unique|length) == length and
    (map(.pair_id)|unique|length) == $facts * 16;
    "incorrect enrollment or duplicate identity") |
. as $rows |
group_by(.family) |
require(length == 4; "expected four whole wording families") |
map(
    . as $family |
    require(length == $facts * 8 and (group_by(.pair_id)|length) == $facts * 4
        and (group_by(.fact)|length) == $facts
        and all(group_by(.fact)[]; length == 8
            and (map([.order,.sign,.selector])|unique|length) == 8
            and (map(.affirmative.text)|unique|length) == 1)
        and all(group_by(shape)[]; length == $facts / 2
            and (group_by(.affirmative.text)|map(length)) == [$facts / 4, $facts / 4]);
        "family lost a crossed fact, order, sign, selector, role, horizon or affirmative verb") |
    {family:.[0].family, pairs:(length/2), facts:(map(.fact)|unique),
     by_verb:(group_by(.affirmative.text) | map({verb:.[0].affirmative.text} + side))}
) as $families |
($rows | group_by(.fact) |
    all(.[]; (group_by(.affirmative.text)|map(length)) == [16,16])) as $counterbalanced |
require($counterbalanced; "affirmative verbs did not exchange across families for each fact") |
{
    rows:($rows|length), pairs:($rows|length/2), families:$families,
    by_affirmative_verb:($rows | group_by(.affirmative.text)
        | map({verb:.[0].affirmative.text} + side)),
    distinct_affirmative_verb_ordinals:($rows | map(.affirmative.i)|unique|length)
}
