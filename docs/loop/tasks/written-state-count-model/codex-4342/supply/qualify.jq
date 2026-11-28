def require($ok; $reason): if $ok then . else error($reason) end;

def review: split("\n") | map(select(length>0)|split("\t") |
  require(length==5; "expected five fields in reviewer ledger") |
  {group_id:.[0],support:.[1],equivalence:.[2],contrast:.[3],reason:.[4]});

def check($group):
  ($group.families|map(length)) as $lengths |
  ($lengths|add) as $prompts |
  ($lengths|map(select(.>1))|length) as $repeats |
  require(.group_id==$group.group_id and (.reason|test("[^[:space:]]")) and
    (.support|test("^[sua]{\($prompts)}$")) and
    (if $repeats==0 then .equivalence=="-" else (.equivalence|test("^[eda]{\($repeats)}$")) end) and
    (if ($lengths|length)==1 then .contrast=="paraphrase"
     else (.contrast|IN("same-fact-inverse","target-switch","same-request","ambiguous")) end);
    "invalid reviewer judgment for \($group.group_id)");

def supported: .support|test("^s+$");
def equivalent: .equivalence=="-" or (.equivalence|test("^e+$"));
def changed_target: .contrast|IN("same-fact-inverse","target-switch","paraphrase");
def histogram(key): group_by(key)|map({key:(.[0]|key),value:length})|from_entries;

($selected|map(select(.split=="screen"))) as $panel |
($review_a|review) as $a | ($review_b|review) as $b |
require(($panel|map(.group_id))==($a|map(.group_id)) and ($panel|map(.group_id))==($b|map(.group_id));
  "both returns must cover exactly the preregistered screen in its original order") |
[$panel|to_entries[] | .value as $group |
  ($a[.key]|check($group)) as $decision_a |
  ($b[.key]|check($group)) as $decision_b |
  {group:$group,rows:[$group.families[][]],$decision_a,$decision_b,
   exclusions:[
     (if ($decision_a|supported|not) then "review_a_endpoint" else empty end),
     (if ($decision_b|supported|not) then "review_b_endpoint" else empty end),
     (if $decision_a.support!=$decision_b.support then "endpoint_review_disagreement" else empty end),
     (if ($decision_a|equivalent|not) then "review_a_equivalence" else empty end),
     (if ($decision_b|equivalent|not) then "review_b_equivalence" else empty end),
     (if $decision_a.equivalence!=$decision_b.equivalence then "equivalence_review_disagreement" else empty end),
     (if ($decision_a|changed_target|not) then "review_a_contrast" else empty end),
     (if ($decision_b|changed_target|not) then "review_b_contrast" else empty end),
     (if $decision_a.contrast!=$decision_b.contrast then "contrast_review_disagreement" else empty end)
   ]}
] as $reviewed |
($reviewed|map(select(.exclusions==[]))) as $accepted |
($reviewed|map(select(.exclusions!=[]))) as $rejected |
{summary:{schema:"uor-r4/directed-relation-source-qualification/1",split:"screen",source_only_reviewers:2,
  selected_groups:($panel|length), selected_articles:($panel|map(.article)|unique|length),
  selected_questions:([$reviewed[].rows[]]|length),
  accepted_groups:($accepted|length), accepted_articles:($accepted|map(.group.article)|unique|length),
  accepted_questions:([$accepted[].rows[]]|length), rejected_groups:($rejected|length),
  shapes:[(["quad","triple","switch","same-target"][]) as $shape |
    ($reviewed|map(select(.group.stratum==$shape))) as $earlier |
    ($accepted|map(select(.group.stratum==$shape))) as $later |
    {shape:$shape,selected:($earlier|length),accepted:($later|length),
     accepted_contrast:($later|histogram(.decision_a.contrast)),
     titles:($later|histogram(.group.article))}],
  exclusion_reasons:([$rejected[].exclusions[]]|histogram(.)),
  accepted_exact_natural_fit_cases:([$accepted[].rows[]|select(.prior.natural)]|length),
  accepted_exact_original_fit_cases_via_qualification_join:([$accepted[].rows[]|select(.prior.historical)]|length),
  accepted_groups_containing_any_exact_fit_case:($accepted|map(select(any(.rows[]; .prior.natural or .prior.historical)))|length)},
  accepted:$accepted,rejected:$rejected}
