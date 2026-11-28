def require($ok; $message): if $ok then . else error($message) end;
def tsv: split("\n") | map(select(length > 0) | split("\t"));
def digest: type == "string" and test("^[0-9a-f]{64}$");
def nonempty: type == "string" and test("[^[:space:]]");
def grade_counts: group_by(.) | map({key: .[0], value: length}) | from_entries;

def verify_span($context):
  . as $a | $a.char_start as $i | ($i + ($a.text | length)) as $j |
  $i >= 0 and ($a.text | nonempty) and $context[$i:$j] == $a.text and
  $a.bytes == [($context[:$i] | utf8bytelength), ($context[:$j] | utf8bytelength)] and
  $a.first_position > 0 and ($a.source_ids | length) > 0 and $a.first_id == $a.source_ids[0] and
  $a.whole_span_ids == true and ($a.response_lexical_offset == 0 or $a.response_lexical_offset == 1) and
  $a.response_ids[$a.response_lexical_offset:] == $a.source_ids;

def verify_row($row; $decision; $index; $right_grades):
  ["supported", "ambiguous", "contradicted", "unsupported"] as $left_grades |
  require(($row.passage_sha256 | digest) and $decision.index == $index and
    $decision.passage_sha256 == $row.passage_sha256; "missing or wrong review for index \($index)") |
  require(($left_grades | index($decision.left_grade)) != null and
    ($right_grades | index($decision.right_grade)) != null and
    ($decision.left_reason | nonempty) and ($decision.right_reason | nonempty); "invalid review at index \($index)") |
  require($row.left.case_id != $row.right.case_id and $row.left.question != $row.right.question and
    all([$row.left, $row.right][]; (.case_id | nonempty) and (.question | nonempty)); "invalid question pair at index \($index)") |
  require(($row.left.answer | verify_span($row.context)); "left source span mismatch at index \($index)");

def make_case($pair; $side; $other):
  $pair.row as $row | $row[$side] as $case | $row[$other].case_id as $swap |
  $case.answer as $span | $span.first_position as $start |
  {
    input: {case_id: $case.case_id, text: ($row.context + " " + $case.question)},
    gold: {case_id: $case.case_id, start: $start,
      end: (if $start == null then null else $start + ($span.source_ids | length) end),
      group: ("natural-" + $pair.kind + "-" + $row.passage_sha256), swapped_case_id: $swap},
    truth: {case_id: $case.case_id, swapped_case_id: $swap, tape: $pair.kind,
      article: $row.article, paragraph: $row.paragraph, passage_sha256: $row.passage_sha256,
      source_id: ($case.source_id // $case.case_id), question: $case.question,
      source_ids: $span.source_ids, canonical_word_ids: $span.response_ids,
      expected_ids: (if $span == null then null else $span.response_ids + [1] end),
      original_bytes: $span.bytes, original_char: $span.char_start, span_text: $span.text,
      response_lexical_offset: $span.response_lexical_offset}
  };

def previous_grade: if startswith("unsupported_") then "unsupported" else . end;
def normalized_answer: ascii_downcase | gsub("[[:punct:][:space:]]+"; "");
def review($kind; $row; $decision; $previous; $distinct_review; $prior_included; $reasons):
  {kind: $kind, row: $row, decision: $decision, pilot: $previous, distinct_review: $distinct_review,
   prior_fit_positive: $prior_included, exclusion_reasons: $reasons};

def summary:
  map(. + {accepted: (.exclusion_reasons == [])}) |
  {pairs_reviewed: length, accepted_pairs: (map(select(.accepted)) | length),
   excluded_pairs: (map(select(.accepted | not)) | length),
   left_grades: (map(.decision.left_grade) | grade_counts),
   right_grades: (map(.decision.right_grade) | grade_counts),
   supported_reciprocal_pairs: (map(select(.kind == "positive-positive" and .decision.left_grade == "supported" and .decision.right_grade == "supported")) | length),
   supported_unanswerable_pairs: (map(select(.kind == "answerability" and .decision.left_grade == "supported" and .decision.right_grade == "unanswerable")) | length),
   exclusion_reasons: (map(.exclusion_reasons[]) | grade_counts),
   earlier_fit_positive_reuse: {reviewed: (map(select(.prior_fit_positive)) | length),
     accepted: (map(select(.prior_fit_positive and .accepted)) | length)}};

($split | tsv) as $partition |
($partition | map({key: (.[2] | fromjson), value: .[0]}) | from_entries) as $split_by_article |
require(($partition | map(.[0]) | grade_counts) == {train: 24, validation: 7, locked: 7} and
  ($split_by_article | length) == 38; "invalid frozen article split") |
require(($positive | length) == 149 and ($picks | length) == 149 and ($answer | length) == 130;
  "unexpected source denominators") |
require(($positive | map(.article) | unique | length) == 24 and
  all($positive[]; $split_by_article[.article] == "train"); "positive source escaped frozen TRAIN articles") |
require(all($answer[]; $split_by_article[.article] == null);
  "answerability source overlaps a frozen positive article") |
require((($positive + $answer) | map(.passage_sha256) | unique | length) == 279 and
  (($positive + $answer) | map(.left.case_id, .right.case_id) | unique | length) == 558;
  "duplicate source passage or case") |
require(($even | map(.index) | sort) == [range(0;149;2)] and
  ($odd | map(.index) | sort) == [range(1;149;2)] and
  ($a | map(.index) | sort) == [range(0;44)] and
  ($b | map(.index) | sort) == [range(44;87)] and
  ($c | map(.index) | sort) == [range(87;130)]; "incomplete or duplicated review indexes") |
require(($distinct | map(.index) | unique | length) == ($distinct | length) and
  all($distinct[]; . as $d | (.index | type) == "number" and .index >= 0 and .index < 149 and
    .index == (.index | floor) and .passage_sha256 == $positive[$d.index].passage_sha256 and (.reason | nonempty));
  "invalid distinct-answer review") |
(($even + $odd) | INDEX(.index | tostring)) as $positive_reviews |
(($a + $b + $c) | INDEX(.index | tostring)) as $answer_reviews |
($distinct | INDEX(.index | tostring)) as $distinct_reviews |
($pilot_picks | INDEX(.passage)) as $prior_picks |
($pilot | tsv | .[1:] | map({passage_sha256: .[0], left_grade: .[1], left_reason: .[2],
  right_grade: .[3], right_reason: .[4]})) as $previous |
($previous | INDEX(.passage_sha256)) as $previous_reviews |
require(($previous_reviews | length) == 8 and ($prior_picks | length) == 8 and
  all($previous[]; . as $review | $prior_picks[$review.passage_sha256] as $pick |
    any($positive[]; .passage_sha256 == $review.passage_sha256 and .left.case_id == $pick.left and .right.case_id == $pick.right));
  "pilot review does not bind to the same ordered cases") |
require(($old_exclusions | length) == 1 and ($old_exclusions[0] | unique | length) == 13;
  "prior fit exclusion inventory mismatch") |
($old_exclusions[0] | INDEX(.)) as $prior_exclusions |
($old_fit | INDEX(.case_id)) as $prior_included |

([$positive | to_entries[] | .key as $index | .value as $row |
  $positive_reviews[$index | tostring] as $decision |
  $distinct_reviews[$index | tostring] as $distinct_review |
  $previous_reviews[$row.passage_sha256] as $previous_decision |
  $picks[$index] as $pick |
  verify_row($row; $decision; $index; ["supported", "ambiguous", "contradicted", "unsupported"]) |
  require($row.article == $pick.article and $row.passage_sha256 == $pick.passage and
    $row.left.case_id == $pick.left and $row.right.case_id == $pick.right; "positive pick changed at index \($index)") |
  require(($row.right.answer | verify_span($row.context)); "right source span mismatch at index \($index)") |
  $row.left.answer as $left | $row.right.answer as $right |
  require(($left.source_ids | length) <= 8 and ($right.source_ids | length) <= 8 and
    ($left.first_position + ($left.source_ids | length) <= $right.first_position or
     $right.first_position + ($right.source_ids | length) <= $left.first_position) and
    $left.source_ids != $right.source_ids; "positive pair lost distinct native intervals at index \($index)") |
  review("positive-positive"; $row; $decision; $previous_decision; $distinct_review; false;
    [if $decision.left_grade != "supported" then "left_not_supported" else empty end,
     if $decision.right_grade != "supported" then "right_not_supported" else empty end,
     if $previous_decision != null and
       (($previous_decision.left_grade | previous_grade) != $decision.left_grade or
        ($previous_decision.right_grade | previous_grade) != $decision.right_grade)
       then "pilot_grade_disagreement" else empty end,
     if ($left.text | normalized_answer) == ($right.text | normalized_answer) then "same_normalized_answer" else empty end,
     if $distinct_review != null then "distinct_answer_review_excluded" else empty end])
 ]) as $reciprocal |
([$answer | to_entries[] | .key as $index | .value as $row |
  $answer_reviews[$index | tostring] as $decision |
  verify_row($row; $decision; $index; ["unanswerable", "false_premise", "answerable", "ambiguous"]) |
  require($row.index == $index and $row.right.answer == null; "answerability source polarity or index changed") |
  review("answerability"; $row; $decision; null; null; ($prior_included[$row.left.case_id] != null);
    [if $decision.left_grade != "supported" then "left_not_supported" else empty end,
     if $decision.right_grade != "unanswerable" then "right_not_genuinely_unanswerable" else empty end,
     if $prior_exclusions[$row.left.case_id] != null or $prior_exclusions[$row.right.case_id] != null
       then "prior_fit_exclusion" else empty end,
     if ($row.left.answer.source_ids | length) > 8 then "positive_span_exceeds_native_interval_limit" else empty end])
 ]) as $answerability |
($reciprocal + $answerability) as $reviews |
([$reviews[] | select(.exclusion_reasons == []) | make_case(.; "left"; "right"), make_case(.; "right"; "left")]) as $rows |
{
  rows: $rows,
  excluded: [$reviews[] | select(.exclusion_reasons != []) |
    del(.row) + {index: .decision.index, passage_sha256: .decision.passage_sha256}],
  evidence: {schema: "uor-r4/natural-source-qualification/1", split: {train: 24, validation: 7, locked: 7},
    reciprocal: ($reciprocal | summary), answerability: ($answerability | summary),
    pilot_pairs_reconciled: ($previous | length), input_queries: ($rows | length),
    source_queries: ($rows | map(select(.gold.start != null)) | length),
    no_read_queries: ($rows | map(select(.gold.start == null)) | length),
    accepted_passages: ($rows | map(.truth.passage_sha256) | unique | length),
    accepted_articles: ($rows | map(.truth.article) | unique | length)}
}
