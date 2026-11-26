# Descriptive breakdown of the registered far-wording analogue.
($reference[0] | INDEX(.case_id)) as $refs
| [inputs
   | select(.family == "pottery-sorting" and .distance == "far")
   | . as $row
   | ($refs[.case_id] // error("missing held source")) as $ref
   | if ($row.admission != $ref.admission or $row.pair_id != $ref.admission_pair)
     then error("source identity disagreement") else . end
   | {arm: (input_filename | split("/")[-1] | split(".")[0]),
      order: $ref.governor_order, pair: .pair_id, admission, full: .arm.full}]
| group_by([.arm, .order, .pair])
| map(if (map(.admission) | sort) != ["defer", "take"] then error("unpaired source") else
        {arm: .[0].arm, governor_order: .[0].order, won: all(.[]; .full == true)} end)
| group_by([.arm, .governor_order])
| map({arm: .[0].arm, governor_order: .[0].governor_order, pairs: length,
       won: (map(select(.won)) | length)})
| if length != 10 or any(.[]; .pairs != 8) then error("incomplete analogue") else . end
