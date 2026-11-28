def insist($ok; $message): if $ok then . else error($message) end;
def identify($family):
  .pair + {family:$family,rank,pair_id:($family + "-" + .pair.passage_sha256)};
def original:
  to_entries[] | select(.key == "two_answer" or .key == "answerability") |
  .key as $family | .value[].candidates[] | select(.disposition == "review") | identify($family);
def unique_identity($rows):
  ($rows|map(.pair_id)|unique|length)==($rows|length) and
  ($rows|map(.passage_sha256)|unique|length)==($rows|length) and
  ($rows|map(.left,.right)|unique|length)==2*($rows|length);

{"1973_oil_crisis":10,"Harvard_University":13,"Islamism":23} as $counts |
insist(.schema == "natural-reciprocal-validation-selection/1" and .quota_per_article == 4 and
       ([.two_answer[].passages] | sort) == [1,2,7,9,10,18,30] and
       ([.two_answer[].pairs] | add) == 187 and
       (.answerability|length)==11 and ([.answerability[].pairs] | add)==256 and
       all(.answerability[];
           .title as $title |
           .selected == 4 and .passages == .pairs and .pairs == (.candidates|length) and
           ([.candidates[].rank] == [range(1;.passages+1)]) and
           all(.candidates[];
               .pair.article == $title and (.disposition ==
                   (if .rank <= 4 then "review" else "outside_article_quota" end))));
       "prior validation candidate manifest shape mismatch") |
[original] as $prior |
insist($prior == $old and ($old|length) == 67 and unique_identity($old) and
       (($old | group_by(.family) | map([.[0].family,length])) == [["answerability",44],["two_answer",23]]);
       "prior 67 review identities differ from the fixed roster") |
[.answerability[] | select(.title as $title | $counts[$title] != null) |
 .candidates[] | select(.disposition == "outside_article_quota") | identify("answerability")] as $extra |
insist(($extra | group_by(.article) | map({key:.[0].article,value:length}) | from_entries) == $counts and
       ($extra | length)==46 and unique_identity($old + $extra);
       "expanded validation identities are incomplete or collide") |
$extra[]
