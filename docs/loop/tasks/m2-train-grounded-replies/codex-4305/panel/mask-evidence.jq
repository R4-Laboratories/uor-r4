def require($ok; $reason): if $ok then . else error($reason) end;

def literal_start($text; $literal):
    ($text | indices($literal)) |
    require(length == 1; "literal did not resolve uniquely: \($literal)") | .[0];

def head_span($text; $clause; $head):
    literal_start($text; $clause) as $start |
    [($clause | indices($head))[] as $at |
        select(($at == 0 or ($clause[$at-1:$at] | test("[A-Za-z]") | not)) and
            ($clause[$at+($head|length):$at+($head|length)+1] | test("[A-Za-z]") | not)) |
        $start + $at] |
    require(length == 1; "noun head did not resolve uniquely: \($head) in \($clause)") |
    {start:.[0],end:(.[0]+($head|length))};

def member($bytes; $role; $reason): {bytes:$bytes, role:$role, reason:$reason};

def fresh_heads($source; $text; $rule):
    [$design[0].wordings[] | select(.family == $source.family)] |
    require(length == 1; "fresh wording lacks unique grammar") | .[0] as $wording |
    $source.scope_clauses as [$first,$opposite] |
    ($wording.heading | sub("\\{noun\\}"; $wording.reference)) as $heading |
    [
        ($wording.contrast.marks[] as $mark |
            ($mark + " " + $wording.reference) as $np |
            require(($first | contains($np)); "enumerated argument is out of scope") |
            member(head_span($text; $np; $wording.reference); "enumerated_argument";
                "head of a marked member of the acted-on pair")),
        member(head_span($text; $first; $rule.alternative); "independent_alternative";
            "head of the separately named allowed fallback object in the same contested construction"),
        member(head_span($text; $opposite; $wording.rival); "opposite_argument";
            "head of the object in the other signed construction"),
        member(head_span($text; $heading; $wording.reference); "reported_object";
            "head of the object selected by the reporting imperative"),
        member(head_span($text; $heading; $rule.alternative); "alternate_reported_object";
            "head of the alternate reporting object if the fallback argument is used"),
        ($rule.opposite_alias[$source.governor_order] as $alias |
            select($alias != null) |
            member(head_span($text; $opposite; $alias); "opposite_object_alias";
                "nominal object head coreferential with the apposed other signed object"))
    ];

def archived_heads($source; $text; $rule):
    $source.scope_clauses as [$first,$opposite] |
    [
        (if $cohort == "regression" then
            (["red proposal", "small proposal"][] as $np |
                require(($first | contains($np)); "historical pair member is out of scope") |
                member(head_span($text; $np; "proposal"); "enumerated_argument";
                    "head of an explicitly enumerated object in the frozen planning review"))
         else
            member($source.governors.referent.noun; "archived_action_object";
                "head governed in the recorded task directive")
         end),
        member($source.governors.rival.noun; "opposite_argument";
            "head of the object in the other signed construction"),
        member($source.gold_binding.referenced; "reported_object";
            "original referenced object from the independent reporting directive"),
        ($rule.heading_alias as $heading |
            select($heading != null) |
            member(head_span($text; $heading[0]; $heading[1]); "report_object_head";
                "syntactic object of the reporting imperative, modified by the reported object")),
        ($rule.object_alias[$source.governor_order] as $alias |
            select($alias != null) |
            [$first,$opposite][] as $clause |
            member(head_span($text; $clause; $alias); "archived_object_alias";
                "nominal head coreferential with the separately named signed object"))
    ];

def disjoint($a; $b): $a.end <= $b.start or $b.end <= $a.start;

def distinct_address($a; $b):
    disjoint($a.bytes; $b.bytes) and disjoint($a.tokens; $b.tokens);

def census($site; $governor; $slot; $tuples):
    if distinct_address($site; $governor) then
        [$tuples[] | select(.[$slot] == $site.index)] | length
    else 0 end;

def resolve_head($sites; $head):
    [$sites[] | select(.bytes == $head.bytes)] |
    require(length == 1; "annotated object head missing from native sites: \($head)") |
    .[0] + ($head | {role,reason});

. as $row |
$refs[0][$row.row_index] as $source |
$grammar[0] as $rules |
require($row.case_id == $source.id and $row.text == $source.turns[1] and
    $row.prompt_sha256 == $source.structural_prompt_sha256;
    "original row index or source tape changed") |
$row.requests[0] as $req |
require($source.governors.referent.bytes == $req.governor.bytes and
    $source.governors.rival.bytes == $req.opposite_governor.bytes and
    $req.governor.index != $req.opposite_governor.index and
    distinct_address($req.governor; $req.opposite_governor);
    "signed treatment does not address the two distinct original words") |
(if $req.sign == 1 then 0 elif $req.sign == -1 then 1 else error("invalid sign") end) as $slot |
$row.coordinate_map.tuples as $tuples |
([$req.requested.index,$req.opposite.index,$req.heading.index] |
    if $slot == 0 then . else [.[1],.[0],.[2]] end) as $gold |
[$row.coordinate_map.sites | to_entries[] |
    .value + {index:.key,
        text:$row.text[.value.bytes.start:.value.bytes.end],
        bpe_ids:$source.prompt[.value.tokens.start:.value.tokens.end]}] as $sites |
(if $cohort == "fresh" then
    $rules.fresh[$row.family] as $rule |
    require($rule != null; "missing fresh grammar for \($row.family)") |
    fresh_heads($source; $row.text; $rule)
 else
    $rules.archived[$row.family] as $rule |
    require($rule != null; "missing archived grammar for \($row.family)") |
    archived_heads($source; $row.text; $rule)
 end) as $heads |
require(($heads|length) == ($heads|unique_by(.bytes)|length); "duplicate annotated source span") |
($heads | map(resolve_head($sites; .)) | sort_by(.index)) as $good |
($good | map(.index)) as $good_ids |
([$good[] | select(.role == "alternate_reported_object") | .index] | .[0]) as $alt_heading |
([$sites[] |
    . as $site | $rules.displaced_function_words[$site.text|ascii_downcase] as $function |
    select($function != null and ($good_ids | index($site.index)) == null
        and census($site; $req.governor; $slot; $tuples) > 0
        and census($site; $req.opposite_governor; $slot; $tuples) > 0) |
    . + {role:"displaced_function_word", reason:$function}] | .[:($good|length)]) as $wrong |
require(($wrong|length) == ($good|length) and
    all($gold[]; . as $i | $good_ids | index($i) != null) and
    any($tuples[]; . == $gold);
    "good or displaced set failed the original geometry/control check on \($row.case_id)") |
($good | map(. + {
    legal_with_governor:census(.; $req.governor; $slot; $tuples),
    legal_with_wrong_governor:census(.; $req.opposite_governor; $slot; $tuples),
    completes_with_original_opposite_and_heading:(.index as $i |
        distinct_address(.; $req.governor) and any($tuples[]; . == ($gold | .[$slot] = $i))),
    completes_with_original_opposite_and_alternate_heading:(.index as $i |
        $alt_heading != null and distinct_address(.; $req.governor) and
        any($tuples[]; . == ($gold | .[$slot] = $i | .[2] = $alt_heading)))
})) as $eligible |
require(if $cohort == "near" then true else
    ([ $eligible[] | select(.role == "enumerated_argument") ] as $tested |
     ($tested|length) == 2 and
     all($tested[]; .legal_with_governor > 0 and .completes_with_original_opposite_and_heading)
     and any($tested[]; .index == $req.rival.index)
     and ($cohort != "fresh" or any($eligible[]; .role == "independent_alternative"
         and .legal_with_governor > 0 and .completes_with_original_opposite_and_alternate_heading))) end;
    "an explicit rival cannot complete with the original opposite object and its valid heading: \($row.case_id)") |
{
    cohort:$cohort, row_index, case_id, pair_id, family, fact, selector, governor_order, sign:$req.sign,
    guard:($cohort == "regression" or ($cohort == "near" and
        ($guard[0] | any(.[]; .row_index == $row.row_index)))),
    governor:$req.governor, wrong_governor:$req.opposite_governor,
    good:$eligible, wrong:($wrong|map(. + {
        legal_with_governor:census(.; $req.governor; $slot; $tuples),
        legal_with_wrong_governor:census(.; $req.opposite_governor; $slot; $tuples)
    }))
}
