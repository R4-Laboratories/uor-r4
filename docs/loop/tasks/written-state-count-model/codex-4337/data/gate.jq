def flagged: ["5a5ab76c9c0277001abe714b","5a5ad4db9c0277001abe71ce","5a7b67b621c2de001afe9fd6","5ad4c5ed5b96ef001a10a009","5ad4f0d05b96ef001a10a6f2"];
def prompt($text): {case_id,text:$text};
def authored_group:
  if .family == "locker" then
    if .fact_order==0 and .value_assignment==[0,1] and .statement_type=="denied" and .changed_side==0 then "locker-denied-0"
    elif .fact_order==1 and .value_assignment==[0,0] and .statement_type=="question" and .changed_side==1 then "locker-question-duplicate-1"
    elif .fact_order==0 and .value_assignment==[0,0] and .statement_type=="asserted" then "locker-neutral-duplicate-0"
    elif .fact_order==1 and .value_assignment==[0,0] and .statement_type=="asserted" then "locker-neutral-duplicate-1"
    else null end
  elif .family == "fabric" then
    if .fact_order==0 and .value_assignment==[1,0] and .statement_type=="question" and .changed_side==1 then "fabric-question-0"
    elif .fact_order==1 and .value_assignment==[1,0] and .statement_type=="denied" and .changed_side==0 then "fabric-denied-1"
    else null end
  else null end;
($supply_prompts|INDEX(.case_id)) as $human
| ($protected_prompts|INDEX(.case_id)) as $protected
| (
  $supply_truth[]
  | select(.case_id as $id|$chosen[0]|index($id))
  | select((.case_id as $id|flagged|index($id)|not) and (.paired_with as $id|flagged|index($id)|not))
  | select(.is_impossible==false and .gold_source.whole_span_ids and .gold_source.response_lexical_offset==0 and (.gold_source.source_ids|length)<=8)
  | {split:"train",input:prompt($human[.case_id].prompt), gold:{case_id,start:.gold_source.first_position,end:(.gold_source.first_position + (.gold_source.source_ids|length)),group:("human-"+.article),swapped_case_id:null}, truth:{case_id,tape:"SQuAD2.0 TRAIN via #4329",article,paragraph,passage_sha256,question,source_ids:.gold_source.source_ids,canonical_word_ids:.gold_source.response_ids,original_bytes:.gold_source.bytes,original_char:.gold_source.char_start,expected_ids:(.gold_source.response_ids+[1]),user_bounds,span_text:.gold_source.text}}
),
(
  $authored[]
  | authored_group as $group | select($group!=null)
  | ("gate-"+$group+"-"+(.asked|tostring)) as $id
  | ("gate-"+$group+"-"+((1-.asked)|tostring)) as $swap
  | (.prompt|rtrimstr(" Reply with the word and a period. If the text doesn\u0027t tell you, reply unknown.")) as $text
  | if ($text|endswith("?")|not) then error("authored suffix") else . end
  | {split:"train",input:{case_id:$id,text:$text},gold:{case_id:$id,start:.source_pos,end:(if .source_pos==null then null else .source_pos+1 end),group:$group,swapped_case_id:$swap},truth:{case_id:$id,tape:"author-labelled #4329 original facts and requests, trimmed answer instruction",family,asked,fact_order,statement_type,changed_side,site_ids,site_positions,same_value:(.site_ids[0]==.site_ids[1]),source_ids:(if .supported then [.site_ids[.asked]] else [] end),expected_ids:(if .supported then [.site_ids[.asked],1] else null end)}}
),
(
  $protected_truth[]
  | select(.case_id|test("^protected-c02[1-4]$"))
  | {split:"protected",input:prompt($protected[.case_id].prompt),gold:{case_id,start:.source.pos,end:(.source.pos+1),group:.contrast_group,swapped_case_id:.paired_ids[0]},truth:{case_id,tape:"protected #4330 stock address anchors",source_ids:[.source.id],expected_ids,same_id_rival}}
)
