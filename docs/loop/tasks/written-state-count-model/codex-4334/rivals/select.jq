def starts_word($row):
    (.features[0] as $class | [2,4,6,8] | index($class) != null)
    or .pos == $row.user_bounds[0]
    or $row.prefix[.pos-1] == 223;

($truth | INDEX(.case_id)) as $keys
| . as $row
| $keys[.case_id].sites as $sites
| if $sites == null or $row.question_start == null then error("missing exact row identity or boundary") else . end
| .candidates |= map(select(
    if $mask == "word" then starts_word($row)
    elif $mask == "passage" then .pos < $row.question_start
    elif $mask == "passage-word" then starts_word($row) and .pos < $row.question_start
    elif $mask == "supplied" then . as $c | any($sites[]; .pos == $c.pos and .id == $c.id)
    else error("unknown mask") end
))
| .candidates as $kept
| if all($sites[]; . as $site | any($kept[]; .pos == $site.pos and .id == $site.id))
  then . else error("removed an annotated site") end
