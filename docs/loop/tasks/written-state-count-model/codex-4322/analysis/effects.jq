def effects:
  {gained_exact:map(select(.exact and (.baseline_exact|not))|.pair_id),
   lost_exact:map(select(.baseline_exact and (.exact|not))|.pair_id),
   gained_directed:map(select(.directed and (.baseline_directed|not))|.pair_id),
   lost_directed:map(select(.baseline_directed and (.directed|not))|.pair_id)};
($base[0].pairs|INDEX(.[];.pair_id)) as $refs |
if ($refs|keys) != ([.pairs[].pair_id]|sort) then error("baseline pair census mismatch") else . end |
[.pairs[] | . as $pair | $refs[$pair.pair_id] as $ref |
 if [$pair.facts,$pair.wording,$pair.contrast] != [$ref.facts,$ref.wording,$ref.contrast]
 then error("baseline reference mismatch") else . end |
 . + {baseline_exact:$ref.exact,baseline_directed:$ref.directed}] as $pairs |
{overall:($pairs|effects),contrasts:($pairs|group_by(.contrast)|
 map({key:(.[0].contrast|tostring),value:effects})|from_entries)}
