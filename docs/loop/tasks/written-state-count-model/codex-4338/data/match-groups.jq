# Rank an entire published pair as one unit. An unreachable target removes the
# whole unit. Singletons retain their actual unpaired labels.
($coverage[0].cases | INDEX(.case_id)) as $seen |
sort_by([.case_id,.swapped_case_id] | map(select(. != null)) | min) |
group_by([.case_id,.swapped_case_id] | map(select(. != null)) | min)[] |
select(all(.[]; $seen[.case_id].gold_present and $seen[.case_id].swapped_gold_present != false)) |
(if length == 1 and .[0].swapped_case_id == null and .[0].start != null then "single"
elif length == 2 and .[0].swapped_case_id == .[1].case_id and .[1].swapped_case_id == .[0].case_id then
  if all(.[]; .start != null) then "both"
  elif any(.[]; .start != null) then "mixed" else error("no positive") end
else error("invalid reciprocal group") end) as $kind |
[$kind,(sort_by([(.start==null),.case_id])|map(.case_id)|join(" "))] | @tsv
