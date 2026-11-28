def insist($ok;$why): if $ok then . else error($why) end;
def text: type=="string" and test("[^[:space:]]");
def grade($valid): . as $value | $valid|index($value)!=null;
def decision($case;$ledger):
  $ledger[$case.pair_id] as $d |
  insist($d.passage_sha256==$case.passage_sha256 and $d.left_case_id==$case.left.case_id and
    $d.right_case_id==$case.right.case_id;"review cites different source or question") |
  ["supported","ambiguous","unsupported","contradicted"] as $grades |
  insist(($d.left_grade | grade($grades)) and
    ($d.right_grade | grade(if $case.family=="two_answer" then $grades
       else ["unanswerable","false_premise","ambiguous","answerable"] end)) and
    ([$d.left_reason,$d.right_reason,$d.distinct_reason] | all(.[];text)) and
    ($d.distinct|type)=="boolean";"invalid or incomplete review") | $d;

($source|map(.pair_id)|sort) as $ids |
($source|group_by(.family)|map([.[0].family,length])|sort) as $families |
insist(($ids|unique|length)==($ids|length); "duplicate validation pair identity") |
if $ARGS.named.registered == null then
  insist(($ids|length)==67 and $families==[["answerability",44],["two_answer",23]];
    "unregistered validation denominator")
else
  ($source|map({article,passage_sha256,family,rank,pair_id,left:.left.case_id,right:.right.case_id})) as $rows |
  $ARGS.named.registered as $roster |
  insist(($ids|length)==113 and $families==[["answerability",90],["two_answer",23]] and
    ($rows[:67]|sort_by(.pair_id)) == ($roster[:67]|sort_by(.pair_id)) and $rows[67:] == $roster[67:];
    "unregistered expanded validation roster")
end |
insist(all([$first,$second][]; (map(.pair_id)|sort)==$ids);"incomplete or duplicate review roster") |
($first|INDEX(.pair_id)) as $a | ($second|INDEX(.pair_id)) as $b |
$source[] as $row | decision($row;$a) as $left | decision($row;$b) as $right |
[
  if $left.left_grade!="supported" or $right.left_grade!="supported" then "unsupported_or_disputed_positive" else empty end,
  if $left.right_grade!=$right.right_grade then "second_question_grade_disagreement" else empty end,
  if $row.family=="two_answer" and ($left.right_grade!="supported" or $right.right_grade!="supported")
    then "second_source_not_licensed" else empty end,
  if $row.family=="answerability" and
    ((["unanswerable","false_premise"]|index($left.right_grade))==null or
     (["unanswerable","false_premise"]|index($right.right_grade))==null)
    then "candidate_negative_answerable_or_ambiguous" else empty end,
  if ($left.distinct and $right.distinct)|not then "contrast_not_uniquely_licensed" else empty end
] as $reasons |
{pair_id:$row.pair_id,family:$row.family,article:$row.article,passage_sha256:$row.passage_sha256,
 left_case_id:$row.left.case_id,right_case_id:$row.right.case_id,
 accepted:($reasons|length==0),reasons:$reasons,
 negative_group:(if $row.family=="answerability" and $left.right_grade==$right.right_grade
                 then $left.right_grade else null end)}
