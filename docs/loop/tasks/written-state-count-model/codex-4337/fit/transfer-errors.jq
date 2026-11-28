def bank_totals($w;$keys):
  if ($keys|length)!=16 then error("bank width") else
    [$keys[] | $w[.]] as $terms | [($terms[:8]|add),($terms[8:]|add)] end;
def site($feature;$span):
  if $span==null then {keys:$feature.no_read_keys,score:$feature.scores.selection.no_read_score}
  else first($feature.candidates|to_entries[]|select(.value.start==$span.start and .value.end==$span.end)
    | {keys:.value.keys,score:$feature.scores.candidates[.key]}) end;
def location($s;$g;$t):
  if $s==$g then "exact"
  elif $g==null then "unsupported-selected-span"
  elif $s==null then "unsupported-route-on-supported"
  elif $s.start==$g.start then "correct-start-wrong-end"
  elif $s.end==$g.end then "correct-end-wrong-start"
  elif $s.start<$g.end and $g.start<$s.end then "overlap-only"
  elif any($t.sites[]?; .start==$s.start and .end==$s.end) then "exact-other-fact-site"
  else "disjoint-other-interval" end;
($truth | INDEX(.case_id)) as $ts |
($features | map(select(.arm==$arm)) | INDEX(.case_id)) as $fs |
($words | split(" ") | map(select(length>0)|tonumber)) as $w |
if ($w|length)!=4096 then error("word count") else . end |
[.cases[] | . as $r | $fs[.case_id] as $f | $ts[.case_id] as $t |
  site($f;.gold) as $gold | site($f;.strongest_gold_rival.span) as $rival |
  bank_totals($w;$gold.keys) as $gs | bank_totals($w;$rival.keys) as $rs |
  if ($gs|add)!=$gold.score or ($rs|add)!=$rival.score or
     ($gs|add)-($rs|add)!=$r.minimum_gold_margin then error("cold rival scores differ") else . end |
  {case_id,cohort:$t.tape,supported:(.gold!=null),location:location(.selected;.gold;$t),
   margin:.minimum_gold_margin, shared_margin:($gs[0]-$rs[0]),route_margin:($gs[1]-$rs[1]),
   correct_interval:.gold,selected_interval:.selected,strongest_rival:.strongest_gold_rival.span}] as $rows |
{arm:$arm,rows:$rows,breakdown:($rows|group_by(.cohort,.supported)|map({
  cohort:.[0].cohort,supported:.[0].supported,count:length,
  location:(group_by(.location)|map({category:.[0].location,count:length})),
  margin_signs:(group_by((.shared_margin>0),(.route_margin>0))|map(
     {shared_banks_prefer_gold:(.[0].shared_margin>0),route_banks_prefer_gold:(.[0].route_margin>0),count:length}))}))}
