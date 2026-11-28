# Invoke with jq -nc --slurpfile gold GOLD.jsonl -f rivals.jq DECISIONS.jsonl
def diagnose($arm;$obs;$truth):
  ($obs.candidates|map(select(.pos==$truth.source.pos))|first // null) as $gold |
  ($obs.candidates|map(select(.pos!=$truth.source.pos))) as $others |
  ($others|map(.score)|max) as $top |
  ($others|map(select(.score==$top))|first // null) as $rival |
  {case_id:$truth.case_id,cohort:$truth.cohort,split:$truth.split,delay:$truth.source_delay,
   arm:$arm,gold:$truth.source.pos,winner:$obs.winner_pos,winning_ties:$obs.best_ties,
   available:($gold!=null),correct:($obs.winner_pos==$truth.source.pos),
   no_read_score:$obs.no_read_score,gold_score:$gold.score,strongest_rival_score:$top,
   strongest_rival_pos:$rival.pos,
   strict_rank:(if $gold then 1+([$others[]|select(.score>$gold.score)]|length) else null end),
   loses_to_threshold:(if $gold then $gold.score<=$obs.no_read_score else null end),
   strongest_rival_in_query:(if $rival then $rival.pos >= $obs.query_bounds[0] and $rival.pos < $obs.query_bounds[1] else null end),
   strongest_rival_same_id:(if $rival then $rival.id==$truth.source.id else null end),
   shared_feature_vector:(if $gold then any($others[]; .features==$gold.features) else null end)};
($gold|INDEX(.case_id)) as $truth |
inputs as $row |
$truth[$row.case_id] as $t | select($t.supported) |
["old_framed","expanded_frozen","expanded_fitted"][] as $arm |
diagnose($arm;$row[$arm];$t)
