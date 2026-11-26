def ensure($yes;$msg): if $yes then . else error($msg) end;
def admissible: .regression.source==4 and .near.correct==24;
def orderkey: [(-.train.source),(-.train.local),.train.missing_margins,.train.worst_deficit,.train.sum_deficit,
  (-.train.full),(-.train.strict_third),.ordinal];
def ranked: sort_by(if admissible then [0,0,0]+orderkey else
  [1,(-(.regression.source+.near.correct)),(-.regression.source)]+orderkey end);
def top($group): [$group[]] | ranked | if length>0 then .[0].id else null end;
ensure(length>=2 and (map(.id)|unique|length)==length and any(.[];.id=="original") and any(.[];.id=="median");
 "candidate identifiers or baselines") |
ensure(all(.[];.train.pairs==32 and .regression.pairs==4 and .near.pairs==24 and
 (.regression.source_ids|unique|length)==.regression.source and (.near.wins|unique|length)==.near.correct);
 "incomplete election populations") |
ranked as $all |
{selection:(if ($all[0]|admissible) then "admissible" else "diagnostic" end),
 selected:$all[0].id,admissible_ids:[$all[]|select(admissible)|.id],candidates:length,
 best_initial:top([.[]|select(.kind=="initial")]),best_learned:top([.[]|select(.kind=="learned")]),
 best_recovery:top([.[]|select(.kind=="recovery")]),
 ranking:[$all[]|{id,kind,ordinal,admissible:admissible,regression:.regression.source,near:.near.correct,
 train:.train.source,local:.train.local,missing_margins:.train.missing_margins,worst_deficit:.train.worst_deficit,
 sum_deficit:.train.sum_deficit,full:.train.full,third:.train.strict_third}]}
