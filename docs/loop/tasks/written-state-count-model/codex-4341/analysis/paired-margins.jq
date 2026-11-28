# Descriptive readback of a verified single-arm panel. A null address means NoRead.
def side($case; $other; $index):
  $index[$case.id] as $row |
  if ($row == null or $row.opening_sha256 != $case.opening_sha256) then error("missing authenticated prompt") else . end |
  ([$row.features[]|select(.arm==$arm)]|if length==1 then .[0] else error("arm multiplicity") end) as $f |
  if ($f.scores.candidates|length)!=($f.candidates|length) then error("score arity") else . end |
  ([range($f.candidates|length) as $i | {span:$f.candidates[$i], score:$f.scores.candidates[$i]}]
    + [{span:null,score:$f.scores.selection.no_read_score}]) as $choices |
  ($case.target.span//null) as $wanted | ($other.target.span//null) as $mate |
  if $wanted==$mate then error("identical target actions") else . end |
  ([$choices[]|select(.span==$wanted)|.score]) as $gold |
  ([$choices[]|select(.span==$mate)|.score]) as $rival |
  if ($gold|length)!=1 or ($rival|length)!=1 then error("missing gold or mate choice") else . end |
  ([$choices[]|select(.span!=$wanted)|.score]|max) as $best |
  {id:$case.id, margin_over_mate:($gold[0]-$rival[0]), margin_over_all:($gold[0]-$best),
   selected_kind:(if $f.scores.selection.span==$wanted then "gold"
      elif $f.scores.selection.span==$mate then "mate"
      elif $f.scores.selection.span==null then "other_no_read" else "other_interval" end)};
def pairs:
  if $panel=="validation" then
    $roster[0].titles[] as $t | $t.selected[] | select(.qualification.stage=="qualified") |
    {title:$t.name,family:$t.family,cases:.qualification.opening.cases}
  elif $panel=="train" then
    $roster[0].training[]|select(.mixture==$mix)|.sources[]|select((.openings|length)==2) |
    [.openings[].cases[0]] as $cases | {title,family:(if all($cases[];.target.kind=="source") then "two_answer" else "answerability" end),$cases}
  else error("bad panel") end;
(.capture|INDEX(.request_id)) as $index |
[pairs | .cases as $c | [side($c[0];$c[1];$index),side($c[1];$c[0];$index)] as $s |
 {title,family,sides:$s,strict_against_mate:all($s[];.margin_over_mate>0),strict_against_all:all($s[];.margin_over_all>0),
  question_effect:($s[0].margin_over_mate+$s[1].margin_over_mate)}] as $rows |
if ($rows|length)==0 then error("no eligible pairs") else . end |
{mixture:$mix,arm:$arm,panel:$panel,by_family:($rows|group_by(.family)|map({family:.[0].family,pairs:length,
 strict_against_mate:(map(select(.strict_against_mate))|length), strict_against_all:(map(select(.strict_against_all))|length),
 lost_to_other_or_tie:(map(select(.strict_against_mate and (.strict_against_all|not)))|length),
 question_effect_signs:(group_by(if .question_effect>0 then "positive" elif .question_effect<0 then "negative" else "zero" end)|
  map({sign:(if .[0].question_effect>0 then "positive" elif .[0].question_effect<0 then "negative" else "zero" end),pairs:length}))})),rows:$rows}
