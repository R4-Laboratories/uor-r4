def insist($ok; $message): if $ok then . else error($message) end;
def denominator($title; $family):
  $selection[0][$family][] | select(.title==$title) | .pairs;
def negative_groups($rows):
  ["unanswerable","false_premise","ambiguous","answerable",null] | map(. as $group |
    ($rows | map(select(.negative_group == $group))) as $members |
    {group:$group,concordant_or_unresolved:($members|length),
     semantic:($members|map(select(.accepted))|length),
     qualified:($members|map(select(.qualified))|length)});

($old[0].titles | INDEX([.family,.article]|@json)) as $baseline |
insist(($pairs|length)==113 and ($baseline|length)==18; "invalid population in coverage report") |
($expanded[0] | .titles |= map(
  . as $entry | $baseline[[$entry.family,$entry.article]|@json] as $prior |
  ($pairs | map(select(.article == $entry.article and .family == $entry.family))) as $rows |
  insist($prior != null and ($rows|length) == .selected; "missing registered article") |
  . + {mechanical_pairs:denominator($entry.article;$entry.family),original:{
         selected:$prior.selected,semantic:$prior.semantic,qualified:$prior.qualified},
       additional_selected:(.selected-$prior.selected)} +
  (if .family=="answerability" then {negative_groups:negative_groups($rows)} else {} end)
)) + {original_selected:67,additional_selected:46,expanded_selected:113,
      answerability_negative_groups:negative_groups($pairs | map(select(.family=="answerability")))}
