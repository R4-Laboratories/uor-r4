# Inputs: official SQuAD 2.0 TRAIN JSON, selected pairs, screened questions.
def selected($by_case;$pair;$case;$article):
  $by_case[$case] as $row |
  if $row == null or $row.article != $pair.article or $row.passage_sha256 != $pair.passage
     or ($row.paragraph|type) != "number" then error("selected case identity mismatch")
  else
    $article.paragraphs[$row.paragraph] as $paragraph |
    ($paragraph.qas | map(select(.id == $row.source_id))) as $official |
    $row.answer as $answer |
    if ($official|length) != 1 or $official[0].is_impossible
       or $official[0].question != $row.question
       or (any($official[0].answers[]; .answer_start == $answer.char_start and .text == $answer.text)|not)
       or $paragraph.context[$answer.char_start:($answer.char_start+($answer.text|length))] != $answer.text
       or $row.text != ($paragraph.context + " " + $row.question)
    then error("selected source span or prompt mismatch: " + $case)
    else {paragraph:$row.paragraph, case_id:$case, source_id:$row.source_id,
          question:$row.question, answer:$answer}
    end
  end;

($picked | map(.left,.right)) as $ids |
($questions | map(select(.case_id as $id | $ids|index($id))) | INDEX(.case_id)) as $by_case |
.data as $articles |
$picked[] as $pair |
($articles | map(select(.title == $pair.article))) as $article |
if ($article|length) != 1 then error("selected article lookup mismatch")
else $article[0] end as $source |
selected($by_case;$pair;$pair.left;$source) as $left |
selected($by_case;$pair;$pair.right;$source) as $right |
if $left.paragraph != $right.paragraph then error("reciprocal cases refer to different paragraphs") else . end |
$source.paragraphs[$left.paragraph] as $paragraph |
($paragraph.qas | map(select(.is_impossible == true)) | sort_by(.id)) as $impossible |
{article:$pair.article, passage_sha256:$pair.passage, paragraph:$left.paragraph,
 context:$paragraph.context, left:$left, right:$right,
 impossible_candidates:($impossible|length), impossible:($impossible|first // null)}
