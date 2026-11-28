def require($ok; $reason): if $ok then . else error($reason) end;
def surface: ascii_downcase | [scan("[[:alpha:]]+|[0-9]+")];
def object_set: map({key: ., value:true}) | from_entries;

def family($source_words; $function_words):
  reduce .[] as $word ([];
    (if ($word|test("^[0-9]+$")) then "<num>"
     elif $source_words[$word] and ($function_words[$word]|not) then "<src>"
     else $word end) as $piece |
    if ($piece=="<src>" or $piece=="<num>") and .[-1]==$piece then .
    else . + [$piece] end
  ) | join(" ");

def exact_answer($context):
  if type != "object" or (.text|type) != "string" or (.answer_start|type) != "number"
  then null
  else
    .text as $text | .answer_start as $a | ($a+($text|length)) as $b |
    if $a<0 or $a!=($a|floor) or ($text|test("[^[:space:]]")|not) or $context[$a:$b]!=$text
    then null
    else {
      original_char:[$a,$b], original_bytes:[($context[0:$a]|utf8bytelength),($context[0:$b]|utf8bytelength)],
      span_text:$text
    } end
  end;

def examine($context; $source_words; $function_words; $prior):
  . as $q |
  if .is_impossible then {status:"impossible", annotations:0}
  else
    (.answers | map(exact_answer($context))) as $answers |
    ($answers|unique) as $exact |
    (.question|surface) as $words |
    {annotations:($answers|length), exact_listed:($exact|map(select(.!=null))|length)} +
    (if ($answers|length)==0 then {status:"no-annotation"}
     elif any($answers[]; .==null) then {status:"invalid-offset"}
     elif ($exact|length)!=1 then {status:"span-disagreement"}
     elif ($words|length)==0 then {status:"empty-normalized-question"}
     else {status:"eligible", case:({case_id:$q.id, question_source_id:$q.id, question:$q.question,
       annotations:($answers|length),
       wording:$words, family:($words|family($source_words;$function_words)),
       prior:($prior[$q.id] // {qualified:false,natural:false,historical:false})}+$exact[0])
     } end)
  end;

def unique_wording: map(.wording) | unique;
def spans:
  group_by(.original_bytes) | map({bytes:.[0].original_bytes,
    cases:(group_by(.wording)|map(min_by(.case_id))|sort_by(.case_id))});

def choose($count):
  . as $span |
  if $count==1 then $span.cases[] | [.] else
    range(0;($span.cases|length)) as $i |
    range($i+1;($span.cases|length)) as $j | [$span.cases[$i],$span.cases[$j]]
  end;

def disjoint($left; $right):
  $left.bytes[1] <= $right.bytes[0] or $right.bytes[1] <= $left.bytes[0];

def two_targets($first_count; $second_count):
  . as $spans |
  $spans[] as $first | $spans[] as $second |
  select(disjoint($first;$second) and ($first_count!=$second_count or $first.bytes<$second.bytes)) |
  ($first|choose($first_count)) as $left | ($second|choose($second_count)) as $right |
  select((($left+$right)|unique_wording|length)==($first_count+$second_count)) |
  [$left,$right];

def shape_choices:
  . as $spans | {
    quad:([$spans|two_targets(2;2)]|first),
    triple:([$spans|two_targets(2;1)]|first),
    switch:([$spans|two_targets(1;1)]|first),
    "same-target":([$spans[]|choose(2)|[.]]|first)
  };

def count_by(key): group_by(key) | map({key:(.[0]|key),value:length}) | from_entries;
def counts_for($all; $picked; $split; $shapes):
  ($all|map(select(.split==$split))) as $rows |
  [$rows[].observations[]] as $questions |
  ($picked|map(select(.split==$split))) as $selected |
  [$selected[].families[][]] as $picked_cases |
  [$questions[]|select(.status=="eligible").case] as $eligible |
  {articles:($articles[0]|map(select(.split==$split))|length), paragraphs:($rows|length),
   duplicate_paragraphs:($rows|map(select(.duplicate))|length), answerable_questions:($questions|map(select(.status!="impossible"))|length),
   question_status:($questions|count_by(.status)),
   listed_answer_annotations:([$questions[].annotations]|add // 0),
   records_with_multiple_listed_answers:($questions|map(select(.annotations>1))|length),
   records_with_multiple_exact_listed_spans:($questions|map(select(.exact_listed>1))|length),
   independently_identified_annotators:null,
   exact_eligible_passages:($rows|map(select(any(.observations[]; .status=="eligible")))|length),
   evidence_shape:([$shapes[] as $shape|{key:$shape,value:($rows|map(select(.shapes[$shape]!=null))|length)}]|from_entries),
   assigned_shape:([$shapes[] as $shape|{key:$shape,value:($rows|map(select(.shape==$shape))|length)}]|from_entries),
   selected_shape:([$shapes[] as $shape|{key:$shape,value:($selected|map(select(.stratum==$shape))|length)}]|from_entries),
   selected_passages:($selected|length), selected_articles:($selected|map(.article)|unique|length),
   selected_source_qa_records:($picked_cases|length),
   selected_listed_answer_annotations:([$picked_cases[].annotations]|add // 0),
   eligible_wording_families:($eligible|map(.family)|unique|length),
   selected_wording_families:($picked_cases|map(.family)|unique|length),
   previous_exposure:{
     qualified_exact:($eligible|map(select(.prior.qualified))|length),
     fitted_natural_exact:($eligible|map(select(.prior.natural))|length),
     fitted_original_via_qualification_join_exact:($eligible|map(select(.prior.historical))|length),
     selected_qualified_exact:($picked_cases|map(select(.prior.qualified))|length),
     selected_fitted_natural_exact:($picked_cases|map(select(.prior.natural))|length),
     selected_fitted_original_via_qualification_join_exact:($picked_cases|map(select(.prior.historical))|length)
   }};

[inputs] as $source |
($hashes|split("\n")|map(select(length>0)|split("\t"))) as $hash_rows |
($hash_rows|map(.[0])|count_by(.)) as $passage_duplicates |
($prior|group_by(.source_id)|map({key:.[0].source_id, value:.})|from_entries) as $previous |
($registered[0].function_words|object_set) as $function_words |
$registered[0].shape_order as $shapes |
$registered[0].sample_per_shape_per_split as $cap |
require(($hash_rows|length)==($source|length) and all($hash_rows[]; length==2 and all(.[]; test("^[0-9a-f]{64}$")));
  "missing or malformed passage digest") |
[$source[].qas[].id] as $ids |
require(($ids|length)==($ids|unique|length) and all($ids[]; type=="string" and test("^[0-9a-f]{24}$"));
  "source QA IDs must be distinct and canonical") |
[$source|to_entries[] | .value as $row | $hash_rows[.key] as [$digest,$rank] |
  ($row.context|surface|unique|object_set) as $source_words |
  ($row.qas | map(.id as $id |
     [($previous[$id] // [])[] | select(.article==$row.article and .paragraph==$row.paragraph and .passage_sha256==$digest)] as $matches |
     examine($row.context;$source_words;$function_words;
       {($id):{qualified:($matches|length>0), natural:any($matches[]; .natural), historical:any($matches[]; .historical)}})
   )) as $observations |
  ($passage_duplicates[$digest]>1) as $duplicate |
  (if $duplicate then {} else [$observations[]|select(.status=="eligible").case]|spans|shape_choices end) as $choices |
  ([$shapes[] as $shape | select($choices[$shape]!=null) | $shape]|first) as $shape |
  ($row|del(.qas))+{passage_sha256:$digest, passage_rank:$rank, passage_bytes:($row.context|utf8bytelength),
     duplicate:$duplicate, observations:$observations, shapes:$choices, shape:$shape}
 ] | sort_by(.passage_rank,.article,.paragraph) |
. as $population |
[$population[]|.split as $split|.observations[]|select(.status=="eligible")|.case+{split:$split}] as $eligible |
([$eligible[]|select(.split=="screen").family]|unique|object_set) as $screen_families |
([$eligible[]|select(.prior.natural or .prior.historical).family]|unique|object_set) as $previous_fitted_families |
[$population[]|select(.shape!=null)] | group_by(.split,.shape) |
map(sort_by(.passage_rank,.article,.paragraph)[:$cap]) | add // [] |
map(. as $row | {
  group_id:("codex-4342-"+.passage_sha256), split, article, paragraph, passage_sha256, passage_rank, passage_bytes,
  stratum:.shape, context, families:.shapes[$row.shape]
}) | sort_by(.split,.passage_rank,.article,.paragraph) |
. as $selected |
{summary:{
  schema:"uor-r4/directed-relation-train-supply-inventory/1",
  screen:counts_for($population;$selected;"screen";$shapes),
  construction:counts_for($population;$selected;"construction";$shapes),
  family_quarantine:{
    reserved_skeletons:($screen_families|length),
    construction_eligible_cases_overlapping_screen:($eligible|map(select(.split=="construction" and $screen_families[.family]))|length),
    construction_eligible_skeletons_overlapping_screen:([$eligible[]|select(.split=="construction" and $screen_families[.family]).family]|unique|length),
    recorded_previous_fitted_skeletons:($previous_fitted_families|length),
    eligible_screen_cases_overlapping_previous_fitted:($eligible|map(select(.split=="screen" and $previous_fitted_families[.family]))|length),
    selected_screen_cases_overlapping_previous_fitted:([$selected[]|select(.split=="screen").families[][]|select($previous_fitted_families[.family])]|length)
  }},
  selected:$selected,
  passages:[$population[]|{article,paragraph,split,passage_sha256,passage_rank,duplicate,shape,
    available_shapes:[.shapes|to_entries[]|select(.value!=null).key],
    question_status:(.observations|count_by(.status))}],
  wording_quarantine:[$eligible[]|select(.split=="construction" and $screen_families[.family])|{case_id,family}]
}
