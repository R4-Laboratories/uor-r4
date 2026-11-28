def strict_extension($short; $long):
  ($short.text | length) as $size
  | ($long.text | startswith($short.text)) and ($long.text[$size:$size+1] | test("\\s"));

([$train[] | select(.gold_source != null) | .article] | unique) as $fit_titles
| ($train | INDEX(.case_id)) as $sealed_by_question
| [
    .data[] | .title as $article | select($fit_titles | index($article))
    | .paragraphs | to_entries[] | .key as $paragraph | .value.context as $context
    | [
        .value.qas[] | select(.is_impossible == false) | .id as $id | .question as $question
        | .answers[]
        | select($context[.answer_start:(.answer_start + (.text | length))] == .text)
        | {id:$id, question:$question, start:.answer_start, text,
           sealed:($sealed_by_question[$id] != null)}
      ] | group_by(.start)[] as $same_start
    | $same_start[] as $short | $same_start[] as $long
    | select($short.id != $long.id and $short.question != $long.question and strict_extension($short; $long))
    | {article:$article, paragraph:$paragraph, context:$context, short:$short, long:$long}
  ]
| unique_by([.short.id,.long.id])[]
