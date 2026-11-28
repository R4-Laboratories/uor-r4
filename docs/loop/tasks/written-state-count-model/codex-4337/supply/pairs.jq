def target($q):
  {start:$q.answer.first_position,end:($q.answer.first_position+($q.answer.source_ids|length))};
def covers($feature;$span):
  any($feature.candidates[]; .start==$span.start and .end==$span.end);
($features|map(select(.arm=="geometry"))|INDEX(.case_id)) as $f |
[group_by(.passage_sha256)[] | . as $group |
  range(0;length) as $i | range($i+1;length) as $j |
  $group[$i] as $a | $group[$j] as $b |
  target($a) as $x | target($b) as $y |
  select($a.article==$b.article and $a.paragraph==$b.paragraph and $a.question!=$b.question) |
  select($x.end<=$y.start or $y.end<=$x.start) |
  select($a.answer.source_ids!=$b.answer.source_ids) |
  $f[$a.case_id] as $af | $f[$b.case_id] as $bf |
  if $af==null or $bf==null then error("missing blind candidate capture") else . end |
  select(($af.candidates|map([.start,.end]))==($bf.candidates|map([.start,.end]))) |
  select(covers($af;$x) and covers($bf;$y)) |
  select(all([$a,$b][]; .canonical_prefix[$x.start:$x.end]==$a.answer.source_ids and
       .canonical_prefix[$y.start:$y.end]==$b.answer.source_ids)) |
  {article:$a.article,passage:$a.passage_sha256,left:$a.case_id,right:$b.case_id,
   left_offset:$a.answer.response_lexical_offset,right_offset:$b.answer.response_lexical_offset}]
| sort_by(.article,.passage,.left,.right)
