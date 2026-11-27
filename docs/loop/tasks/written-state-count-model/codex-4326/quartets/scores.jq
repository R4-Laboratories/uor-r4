# Derive the copy-minus-stop mixed difference from GPU scores on the same four
# instructions. A positive difference is necessary, not sufficient, for all four
# signed decisions. Integer scores strictly on the correct sides imply >= 4.
def row_id: [.pair_id, .side] | @json;
def require(condition; reason): if condition then . else error(reason) end;
($keys | INDEX(.[]; row_id)) as $by_row |
[$rows[] | . as $r | $by_row[row_id] as $k
 | require($k != null; "unknown input row")
 | select($k.one_word_clause_ordinal != null and $k.cohort == "fresh")
 | .arms.learned.steps[2] as $step
 | require($step.stage == 0 and ($step.action_score | type) == "number"; "missing period decision")
 | {ctx:$k.context_id, family:$k.family, selector:$k.operative_clause_ordinal,
    one:$k.one_word_clause_ordinal, sign:(if $k.expected_word_count == 1 then -1 else 1 end),
    score:$step.action_score}]
| group_by(.ctx) | map(
    require(length == 4 and (map([.selector,.one])|unique|length)==4 and (map(.sign)|sort)==[-1,-1,1,1]; "incomplete factorial")
    | require(all(.[]; (.selector == .one) == (.sign == -1)); "wrong factorial polarity")
    | {context_id:.[0].ctx, family:.[0].family, scores:sort_by(.selector,.one)|map(.score),
       copy_minus_stop:(map(.sign*.score)|add), weakest_signed_margin:(map(.sign*.score)|min)})
