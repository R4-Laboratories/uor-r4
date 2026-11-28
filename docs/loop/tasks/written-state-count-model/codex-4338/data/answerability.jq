def require($ok;$msg): if $ok then . else error($msg) end;
($truth|INDEX(.case_id)) as $ts | ($prompts|INDEX(.case_id)) as $qs |
(.data|INDEX(.title)) as $articles |
$truth | map(select(.is_impossible==false)) | sort_by(.case_id) | to_entries[] |
.key as $index | .value as $pos | $ts[$pos.paired_with] as $neg |
$articles[$pos.article].paragraphs[$pos.paragraph] as $p |
($p.qas|INDEX(.id)) as $official | $pos.gold_source as $answer |
require($neg.paired_with==$pos.case_id and $neg.is_impossible==true and
  $neg.article==$pos.article and $neg.paragraph==$pos.paragraph and $neg.passage_sha256==$pos.passage_sha256;"unpaired passage") |
require($official[$neg.case_id].is_impossible==true and $official[$neg.case_id].answers==[] and
  $official[$neg.case_id].question==$neg.question and $official[$pos.case_id].question==$pos.question and
  $official[$pos.case_id].is_impossible==false and any($official[$pos.case_id].answers[];
   .answer_start==$answer.char_start and .text==$answer.text);"official QA identity") |
require($p.context[$answer.char_start:$answer.char_start+($answer.text|length)]==$answer.text and
  $pos.prefix[$answer.first_position:$answer.first_position+($answer.source_ids|length)]==$answer.source_ids and
  $qs[$pos.case_id].prompt==($p.context+" "+$pos.question) and
  $qs[$neg.case_id].prompt==($p.context+" "+$neg.question);"original bytes/prompt") |
{index:$index,article:$pos.article,paragraph:$pos.paragraph,passage_sha256:$pos.passage_sha256,context:$p.context,
 left:{case_id:$pos.case_id,question:$pos.question,answer:$answer},
 right:{case_id:$neg.case_id,question:$neg.question}}
