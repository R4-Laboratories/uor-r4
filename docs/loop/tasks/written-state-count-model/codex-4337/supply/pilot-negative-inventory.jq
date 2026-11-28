# Inputs: official SQuAD 2.0 TRAIN JSON and frozen TRAIN-only metadata.
($train_passages | split("\n") | map(select(length>0)) | map(split("\t") | {rank:.[0],passage:.[1],article:(.[2]|fromjson)})) as $train |
($train | map(.article) | unique) as $titles |
($pairs[0] | map(select(.article as $article | $titles|index($article))) | group_by(.passage) | map(first)) as $train_pairs |
($train_pairs | map(.left)) as $ids |
($questions | map(select(.case_id as $id | $ids|index($id))) | INDEX(.case_id)) as $by_case |
($train_pairs | INDEX(.passage)) as $by_passage |
(.data | map(select(.title as $title | $titles|index($title))) | INDEX(.title)) as $sources |
$train[] as $row |
$by_case[$by_passage[$row.passage].left] as $case |
if $case == null or $case.article != $row.article then error("TRAIN case identity mismatch") else . end |
$sources[$row.article].paragraphs[$case.paragraph] as $paragraph |
if (any($paragraph.qas[]; .id == $case.source_id and .question == $case.question)|not)
then error("TRAIN official paragraph identity mismatch") else . end |
($paragraph.qas | map(select(.is_impossible == true)) | sort_by(.id)) as $impossible |
{article:$row.article,passage_sha256:$row.passage,paragraph:$case.paragraph,
 context:$paragraph.context,impossible_candidates:($impossible|length),impossible:($impossible|first // null)}
