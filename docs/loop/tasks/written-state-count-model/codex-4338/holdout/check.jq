# Inputs are the unchanged official DEV JSON and the private roster JSONL.
def check($ok; $error): if $ok then . else error($error) end;
def char_index_at($text; $bytes):
  reduce ($text | explode[]) as $cp ({bytes:0, chars:0};
    if .bytes < $bytes
    then .bytes += ([$cp] | implode | utf8bytelength) | .chars += 1
    else .
    end)
  | if .bytes == $bytes then .chars else error("invalid UTF-8 byte boundary") end;
def assert_annotation($source; $entry; $span):
  $entry.original_annotation_char as $ch
  | ($source.context[0:$ch] | utf8bytelength) as $b
  | check(($span | any(.answer_start == $ch and .text == $entry.label.text)); "original annotation missing")
  | check($entry.original_annotation_bytes == [$b, ($b + ($entry.label.text|utf8bytelength))]; "original annotation bytes drift")
  | check($source.context[$ch:($ch+($entry.label.text|length))] == $entry.label.text; "original span text drift");

(.data | INDEX(.title)) as $by_title
| [ $roster[] as $row
    | ($by_title[$row.article].paragraphs[$row.paragraph] // error("missing official passage")) as $source
    | ($source.qas | INDEX(.id)) as $by_question
    | $row.window_bytes as [$begin, $end]
    | $source.context as $context
    | char_index_at($context; $begin) as $start
    | char_index_at($context; $end) as $stop
    | $context[$start:$stop] as $window
    | $row
    | check($row.shared_passage == $window; "window bytes drift")
    | check($row.full_paragraph == ($window == $context); "full passage flag drift")
    | check($row.positive.published_impossible == false and $row.negative.published_impossible == true; "reversed roster polarity")
    | . as $original
    | reduce ["positive", "negative"][] as $polarity (.;
        $original[$polarity] as $entry
        | ($by_question[$entry.published_id] // error("missing official question")) as $question
        | check($question.question == $entry.question and $question.is_impossible == $entry.published_impossible; "official question drift")
        | check($entry.prompt == ($window + " " + $question.question); "prompt byte drift")
        | check(($entry.prefix | length) <= 180; "client prefix outside budget")
        | check($entry.prefix[$entry.user_bounds[0]:($entry.user_bounds[0]+($row.shared_passage_ids|length))] == $row.shared_passage_ids; "shared token IDs diverge")
        | check($entry.original_annotation_bytes == ($entry.label.bytes | map(. + $begin)); "window annotation byte origin drift")
        | check($entry.original_annotation_char == ($entry.label.char_start + $start); "window annotation character origin drift")
        | if $polarity == "positive" then
            check($entry.published_annotation_agreement >= 2 and $entry.published_annotation_agreement * 2 > ($question.answers|length); "positive majority missing")
            | check(($question.answers | map(select(.answer_start == $entry.original_annotation_char and .text == $entry.label.text)) | length) == $entry.published_annotation_agreement; "positive agreement count drift")
            | check(($entry.label.source_ids|length) >= 1 and ($entry.label.source_ids|length) <= 8 and $entry.label.whole_span_ids; "positive outside exact copy budget")
            | check($entry.prefix[$entry.label.first_position:($entry.label.first_position+($entry.label.source_ids|length))] == $entry.label.source_ids; "positive original addresses drift")
            | assert_annotation($source; $entry; $question.answers)
          else
            check(($question.answers|length) == 0; "impossible question has gold")
            | assert_annotation($source; $entry; $question.plausible_answers)
          end
      )
    | {split,article,passage_sha256, positive: .positive.published_id, negative: .negative.published_id}
  ] as $verified
| check(($verified | map(.passage_sha256) | unique | length) == ($verified | length); "duplicate passage")
| check(($verified | map([.positive,.negative]) | flatten | unique | length) == ($verified | length)*2; "duplicate question")
| check(($verified | group_by(.article) | all(map(.split)|unique|length == 1)); "article crosses split")
| $verified
| group_by(.split)
| map({key: .[0].split, value: {pairs: length, articles: (map(.article)|unique|length), verified_questions: (length * 2)}})
| from_entries
