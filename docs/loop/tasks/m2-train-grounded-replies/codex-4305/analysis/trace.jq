def tally: group_by(.)|map({key:.[0],value:length})|from_entries;
def comparison($one;$two):
  if $one.eligible==0 then "target absent"
  elif $two.eligible==0 then "rival absent"
  elif $one.best<$two.best then "target lower"
  elif $one.best>$two.best then "rival lower" else "tie" end;
[$rows[] as $r | $routes[$r.row] as $obs |
  if $r.row!=$obs.row or $r.ties!=$obs.best_assignments then error("row or route read mismatch") else . end |
  ($r.sign|if .==1 then 0 else 1 end) as $side | $obs.ranks[$side] as $rank |
  $rank[$r.expected_noun.i] as $requested | $rank[$r.duplicate.i] as $rival |
  [$r.valid[] as $i|select($i!=$r.expected_noun.i and $i!=$r.duplicate.i and $rank[$i].eligible>0)|$rank[$i].best] as $others |
  {row:$r.row,pair:$r.pair,selector:$r.selector,signed:$r.signed,noun:$r.noun,kind:$r.kind,governor:$r.governor,
   local_requested:$requested.best,local_duplicate:$rival.best,other_argument_min:($others|min),
   local_vs_duplicate:comparison($requested;$rival),
   local_vs_other:(if $requested.eligible==0 then "target absent" elif $others==[] then "no eligible other"
      elif $requested.best<($others|min) then "target lower" elif $requested.best>($others|min) then "other lower" else "tie" end),
   native_g:$r.native.g,native_n:$r.native.n,selected_cost:$r.cost,forced_lexical:$obs.forced_lexical}
] as $v |
{rows:($v|length),local_vs_duplicate:($v|map(.local_vs_duplicate)|tally),local_vs_other:($v|map(.local_vs_other)|tally),
 local_duplicate_pair_wins:($v|group_by(.pair)|map(select(all(.[];.local_vs_duplicate=="target lower")))|length),
 unique_wrong_source:($v|map(select((.signed|not) and .kind!="tie" and .kind!="absent")|.kind)|tally),
 requested_locally_lower_but_signed_wrong:($v|map(select(.local_vs_duplicate=="target lower" and (.signed|not)))|length),
 rows_original:$v}
