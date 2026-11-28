def digest($n): ("0"*64 + ($n|tostring))[-64:];
def article($title;$n;$count;$alternates):
  {title:$title,passages:$count,pairs:($count+$alternates),selected:([$count,4]|min),
   candidates:[range(1;$count+1) as $rank |
     ($n+$rank) as $id |
     {rank:$rank,disposition:(if $rank<=4 then "review" else "outside_article_quota" end),
      pair:{article:$title,passage_sha256:digest($id),left:"L\($id)",right:"R\($id)"}},
     (if $rank==1 then range($alternates) as $alt |
       {rank:$rank,disposition:"alternate_within_passage",
        pair:{article:$title,passage_sha256:digest($id),left:"L\($id)",right:"Z\($id)-\($alt)"}}
      else empty end)]};
[1,2,7,9,10,18,30] as $two |
[["1973_oil_crisis",14],["Harvard_University",17],["Islamism",27],
 ["Other_A",24],["Other_B",24],["Other_C",24],["Other_D",24],
 ["Other_E",24],["Other_F",24],["Other_G",24],["Other_H",30]] as $absence |
{schema:"natural-reciprocal-validation-selection/1",quota_per_article:4,
 two_answer:[range($two|length) as $i | article("Two_\($i)";($i*100);$two[$i];(if $i==6 then 110 else 0 end))],
 answerability:[range($absence|length) as $i | article($absence[$i][0];(1000+$i*100);$absence[$i][1];0)]}
