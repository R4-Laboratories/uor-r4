def check($ok; $message): if $ok then . else error($message) end;
def char_at($text; $offset):
  reduce ($text | explode[]) as $cp ({bytes:0, chars:0};
    if .bytes < $offset then .bytes += ([$cp] | implode | utf8bytelength) | .chars += 1 else . end)
  | if .bytes == $offset then .chars else error("invalid UTF-8 window boundary") end;
def annotation($text; $ann; $official):
  $ann.char_start as $i | ($i + ($ann.text | length)) as $j |
  ($i | type) == "number" and $i == ($i | floor) and $i >= 0 and
  ($ann.text | type == "string" and length > 0) and $text[$i:$j] == $ann.text and
  $ann.bytes == [($text[:$i]|utf8bytelength), ($text[:$j]|utf8bytelength)] and
  any($official[]; .answer_start == $i and .text == $ann.text);
def original_annotation($entry):
  $entry.label + {char_start:$entry.original_annotation_char,bytes:$entry.original_annotation_bytes};
def client_opening($context; $side):
  $side.prompt == ($context + " " + $side.question) and
  ($side.prefix | type == "array" and length > 0 and length <= 180) and
  ($side.user_bounds | type == "array" and length == 2) and
  $side.user_bounds[0] >= 0 and $side.user_bounds[1] <= ($side.prefix | length) and
  $side.user_bounds[0] < $side.user_bounds[1];
def positive_label($context; $side; $annotations):
  $side.answer as $a |
  annotation($context; $a; $annotations) and
  $a.whole_span_ids == true and ($a.source_ids | type == "array" and length > 0 and length <= 8) and
  $a.first_id == $a.source_ids[0] and $a.first_position >= $side.user_bounds[0] and
  ($a.first_position + ($a.source_ids|length)) <= $side.user_bounds[1] and
  $side.prefix[$a.first_position:($a.first_position + ($a.source_ids|length))] == $a.source_ids and
  ($a.response_lexical_offset == 0 or $a.response_lexical_offset == 1) and
  $a.response_ids[$a.response_lexical_offset:] == $a.source_ids;
def find_question($paragraph; $id):
  [$paragraph.qas[] | select(.id == $id)] |
  if length == 1 then .[0] else error("missing or duplicate official question") end;
def reciprocal_side($pick; $indexed; $id; $article):
  $indexed[$id] as $src |
  check($src.article == $pick.article and $src.passage_sha256 == $pick.passage_sha256 and
    ($src.paragraph | type) == "number"; "selected reciprocal source identity mismatch") |
  $article.paragraphs[$src.paragraph] as $original |
  find_question($original; $src.source_id) as $q |
  {case_id:$src.case_id,source_id:$src.source_id,question:$src.question,prompt:$src.text,
   prefix:$src.canonical_prefix,user_bounds:$src.user_bounds,answer:$src.answer,
   published_impossible:false,original_annotation_bytes:$src.answer.bytes,
   original_annotation_char:$src.answer.char_start} |
  check(($q.is_impossible == false) and $q.question == .question and
    client_opening($original.context; .) and positive_label($original.context; .; $q.answers);
    "reciprocal opening or official annotation mismatch");
def reciprocal($pick; $indexed; $article):
  ($indexed[$pick.left].paragraph) as $paragraph |
  check($paragraph == $indexed[$pick.right].paragraph; "reciprocal rows use different original paragraphs") |
  $article.paragraphs[$paragraph].context as $context |
  $pick + {paragraph:$paragraph,context:$context,full_paragraph:true,
    window_bytes:[0,($context|utf8bytelength)],
    left:reciprocal_side($pick; $indexed; $pick.left; $article),
    right:reciprocal_side($pick; $indexed; $pick.right; $article)};
def dev_side($entry; $positive):
  {case_id:$entry.published_id,source_id:$entry.published_id,question:$entry.question,
   prompt:$entry.prompt,prefix:$entry.prefix,user_bounds:$entry.user_bounds,
   answer:(if $positive then $entry.label else null end),
   published_impossible:$entry.published_impossible,
   original_annotation_bytes:$entry.original_annotation_bytes,
   original_annotation_char:$entry.original_annotation_char} +
  if $positive then {} else {distractor:($entry.label | {text,bytes,char_start})} end;
def window_annotations($site; $start):
  $site | map(select(.answer_start >= $start) | .answer_start -= $start);
def answerability($pick; $indexed; $article):
  $indexed[$pick.passage_sha256] as $src |
  check($src.split == "validation" and $src.article == $pick.article and
    $src.positive.published_id == $pick.left and $src.negative.published_id == $pick.right;
    "selected answerability source identity mismatch") |
  $article.paragraphs[$src.paragraph] as $original |
  $src.window_bytes as [$begin,$end] |
  char_at($original.context;$begin) as $i | char_at($original.context;$end) as $j |
  $original.context[$i:$j] as $context |
  check($begin < $end and $src.shared_passage == $context and
    $src.full_paragraph == ($context == $original.context); "answerability window drift") |
  find_question($original;$pick.left) as $positive |
  find_question($original;$pick.right) as $negative |
  check($positive.is_impossible == false and $negative.is_impossible == true and
    ($negative.answers | length) == 0 and
    $src.positive.published_impossible == false and $src.negative.published_impossible == true and
    $positive.question == $src.positive.question and $negative.question == $src.negative.question;
    "official answerability question mismatch") |
  check(all([$src.positive,$src.negative][]; . as $entry |
    .original_annotation_bytes == (.label.bytes | map(. + $begin)) and
    .original_annotation_char == (.label.char_start + $i) and
    .prefix[.user_bounds[0]:(.user_bounds[0] + ($src.shared_passage_ids|length))] == $src.shared_passage_ids and
    client_opening($context;$entry)); "answerability opening or full-source coordinate mismatch") |
  ([$positive.answers[] |
    select(.answer_start == $src.positive.original_annotation_char and .text == $src.positive.label.text)] | length) as $agreement |
  $pick + {paragraph:$src.paragraph,context:$context,full_paragraph:$src.full_paragraph,
    window_bytes:$src.window_bytes,left:dev_side($src.positive;true),right:dev_side($src.negative;false)} |
  check($agreement >= 2 and $agreement * 2 > ($positive.answers | length) and
    $src.positive.published_annotation_agreement == $agreement and
    annotation($original.context;original_annotation($src.positive);$positive.answers) and
    annotation($original.context;original_annotation($src.negative);$negative.plausible_answers) and
    annotation($context;$src.negative.label;window_annotations($negative.plausible_answers;$i)) and
    positive_label($context;.left;window_annotations($positive.answers;$i));
    "answerability annotation or consensus mismatch");

check(.version == "v2.0"; "unexpected official SQuAD version") |
. as $dataset |
[$picks[] | select(.family == $family)] as $selected |
(if $family == "two_answer" then
  check(($source | length) == (($selected | length) * 2) and
    ($source | map(.case_id) | unique | length) == ($source | length); "reciprocal source ledger coverage mismatch") |
  ($source | INDEX(.case_id))
elif $family == "answerability" then
  check(($source | length) == ($selected | length) and
    ($source | map(.passage_sha256) | unique | length) == ($source | length); "answerability source ledger coverage mismatch") |
  ($source | INDEX(.passage_sha256))
else error("unregistered family")
end) as $indexed |
$selected[] as $pick |
[$dataset.data[] | select(.title == $pick.article)] as $articles |
check(($articles|length) == 1; "missing or duplicate selected article") |
if $family == "two_answer" then reciprocal($pick;$indexed;$articles[0])
else answerability($pick;$indexed;$articles[0])
end
