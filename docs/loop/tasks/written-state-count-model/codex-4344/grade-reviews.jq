def decisions:
  split("\n")
  | if .[-1] == "" then .[:-1] else error("judgments must end in a newline") end
  | map(split("\t")
    | if length == 4 and (.[2] | IN("s", "r", "a")) and (.[3] | length > 0)
      then . else error("invalid source judgment") end);

map(select(.role == "stop_candidate") | [.first.record, (.first.turn | tostring)]) as $keys
| ($a | decisions) as $review_a
| ($b | decisions) as $review_b
| if $keys != ($review_a | map(.[:2])) or $keys != ($review_b | map(.[:2]))
  then error("judgments must cover the sealed source candidates in order")
  else
    [range($keys | length) | [$review_a[.][2], $review_b[.][2]]] as $labels
    | { candidates: ($labels | length),
        reviewer_a: ($review_a | map(.[2]) | sort | group_by(.) | map({key: .[0], value: length}) | from_entries),
        reviewer_b: ($review_b | map(.[2]) | sort | group_by(.) | map({key: .[0], value: length}) | from_entries),
        agreed_supported: ([$labels[] | select(. == ["s", "s"])] | length),
        agreed_rejected: ([$labels[] | select(. == ["r", "r"])] | length),
        agreed_ambiguous: ([$labels[] | select(. == ["a", "a"])] | length),
        disagreements: ([$labels[] | select(.[0] != .[1])] | length)
      }
  end
