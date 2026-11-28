# Exact equality on all 32 signed coordinates, using the JSON array as key.
def tuples($rows; $keys; $arm):
    ($keys | INDEX(.case_id)) as $index
    | [$rows[] as $row | $index[$row.case_id] as $key
       | select($key.target != null)
       | ($row.candidates | map(select(.pos==$key.target.pos and .id==$key.target.id))) as $target
       | if ($target|length)!=1 then error("gold address missing or repeated") else . end
       | {case_id:$key.case_id,target:($target[0][$arm]|tojson),
          rivals:[$row.candidates[]|select(.pos!=$key.target.pos)|.[$arm]|tojson]}];
def membership($items): reduce $items[] as $item ({}; .[$item]=true);
def measure($a; $b):
    membership([$a[].rivals[]]) as $rivals
    | membership([$b[].target]) as $other_gold
    | {supported:($a|length),distinct_gold:([$a[].target]|unique|length),
       gold_ties_rival_in_own_row:([$a[]|select(.target as $t|any(.rivals[];.==$t))]|length),
       gold_occurs_as_rival_in_panel:([$a[]|select($rivals[.target])]|length),
       gold_occurs_as_gold_in_other_panel:([$a[]|select($other_gold[.target])]|length)};
["real_signed","fixed_signed","swapped_signed","no_question_signed"][]
| . as $arm | tuples($earlier;$k0;$arm) as $a | tuples($recent;$k1;$arm) as $b
| {arm:$arm,earlier:measure($a;$b),recent:measure($b;$a)}
