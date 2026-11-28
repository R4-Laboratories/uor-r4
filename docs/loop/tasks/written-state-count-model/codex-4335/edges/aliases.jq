# Run only on sealed feature output. The truth keys never enter the extractor.
def tuples($rows; $keys; $arm):
    if ([$rows[].case_id] | sort) != ([$keys[].case_id] | sort)
       or ([$keys[].case_id] | unique | length) != ($keys | length)
    then error("feature and key panel identities differ")
    else
        ($keys | INDEX(.case_id)) as $index
        | [$rows[] as $row | $index[$row.case_id] as $key
           | select($key.target != null)
           | ($row.candidates | map(select(.pos == $key.target.pos and .id == $key.target.id))) as $gold
           | if ($gold | length) != 1 then error("gold address missing or repeated") else . end
           | {case_id:$key.case_id, gold:($gold[0][$arm] | tojson),
              rivals:[$row.candidates[] | select(.pos != $key.target.pos) | .[$arm] | tojson]}]
    end;

def counts($panel; $other):
    ($panel + $other) as $all
    | {
          supported: ($panel | length),
          distinct_gold: ([$panel[].gold] | unique | length),
          gold_ties_rival_in_own_row: ([$panel[] | select(.gold as $g | any(.rivals[]; . == $g))] | length),
          own_row_equal_rival_sites: ([$panel[] | .gold as $g | .rivals[] | select(. == $g)] | length),
          gold_ties_rival_in_other_supported_row: ([$panel[] as $row |
              select(any($all[]; .case_id != $row.case_id and any(.rivals[]; . == $row.gold)))]
              | length),
          gold_ties_gold_in_other_supported_row: ([$panel[] as $row |
              select(any($all[]; .case_id != $row.case_id and .gold == $row.gold))]
              | length),
          gold_ties_rival_in_other_panel: ([$panel[] as $row |
              select(any($other[]; any(.rivals[]; . == $row.gold)))]
              | length),
          gold_ties_gold_in_other_panel: ([$panel[] as $row |
              select(any($other[]; .gold == $row.gold))]
              | length)
      };

if ([$earlier_key[].case_id, $recent_key[].case_id] | unique | length)
    != (($earlier_key | length) + ($recent_key | length))
then error("case IDs overlap across the two panels") else . end
| ["real_signed","fixed_signed","swapped_signed","no_question_signed"]
| map(. as $arm
      | tuples($earlier; $earlier_key; $arm) as $a
      | tuples($recent; $recent_key; $arm) as $b
      | {arm:$arm, earlier:counts($a; $b), recent:counts($b; $a)})
