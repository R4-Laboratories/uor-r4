# Compare the positive occurrence to that same original address in the paired negative.
# Both labels are read only after scoring. The human pair shares one complete paragraph.
($gold | INDEX(.case_id)) as $truth |
[inputs as $row | $truth[$row.case_id] as $t | select($t.cohort=="squad" or ($t.cohort=="authored" and ($t.paired_ids|length)==1 and $truth[$t.paired_ids[0]].supported!=$t.supported)) |
  (if $t.supported then $t else $truth[$t.paired_ids[0]] end) as $positive |
  ($row.expanded_fitted.candidates|map(select(.pos==$positive.source.pos))|first) as $site |
  if $site.id!=$positive.source.id then error("original paragraph address differs") else
  {case:$row.case_id,cohort:$t.cohort,group:$t.contrast_group,split:$t.split,delay:$positive.source_delay,
   positive:$t.supported,features:$site.features,score:$site.score} end] |
group_by(.group) | map(
 if length!=2 then error("incomplete pair") else
 (map(select(.positive))|first) as $p | (map(select(.positive|not))|first) as $n |
 {group:$p.group,cohort:$p.cohort,split:$p.split,delay:$p.delay,unchanged:($p.features==$n.features),
  changed:[range(16) as $i|select($p.features[$i]!=$n.features[$i])|$i],
  positive_higher:($p.score>$n.score),same_score:($p.score==$n.score)} end) |
 group_by([.cohort,.split,.delay])|map({cohort:.[0].cohort,split:.[0].split,delay:.[0].delay,pairs:length,
  identical_features:(map(select(.unchanged))|length),identical_scores:(map(select(.same_score))|length),
  higher_supported_score:(map(select(.positive_higher))|length)})
