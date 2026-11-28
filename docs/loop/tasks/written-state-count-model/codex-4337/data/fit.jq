def g: [.family,.statement_type, (.fact_order|tostring), (.value_assignment|map(tostring)|join("")), (.changed_side|tostring)]|join("-");
def prompt($text): {case_id,text:$text};
def to_query:
  rtrimstr(" Reply with the word and a period. If the text doesn\u0027t tell you, reply unknown.")
  | if contains("Tell me what drink was inside the thermos packed by ") then
      sub("Tell me what drink was inside the thermos packed by (?<name>[^.]+)\\.$";"What drink was inside the thermos packed by \(.name)?")
    elif contains("Read me the label from the ticket that ") then
      sub("Read me the label from the ticket that (?<name>[^.]+) brought\\.$";"What is the label from the ticket that \(.name) brought?")
    else . end;
($supply_prompts|INDEX(.case_id)) as $human
| (
  $supply_truth[]
  | select((.case_id as $id|$exclude[0]|index($id)|not) and (.paired_with as $id|$exclude[0]|index($id)|not))
  | select(.is_impossible==false and .gold_source.whole_span_ids and (.gold_source.source_ids|length)<=8)
  | {input:prompt($human[.case_id].prompt), gold:{case_id,start:.gold_source.first_position,end:(.gold_source.first_position + (.gold_source.source_ids|length)),group:("human-"+.article),swapped_case_id:null}, truth:{case_id,tape:"SQuAD2.0 TRAIN via #4329",article,paragraph,passage_sha256,question,source_ids:.gold_source.source_ids,canonical_prefix:.prefix,canonical_word_ids:.gold_source.response_ids,original_bytes:.gold_source.bytes,original_char:.gold_source.char_start,expected_ids:(.gold_source.response_ids+[1]),user_bounds,span_text:.gold_source.text,response_lexical_offset:.gold_source.response_lexical_offset,claim_scope:"published positive with full mechanical alignment, known disputed pairs and uncertain targeted examples removed"}}
),
(
  $authored[]
  | g as $group
  | ("fit-"+$group+"-"+(.asked|tostring)) as $id
  | ("fit-"+$group+"-"+((1-.asked)|tostring)) as $swap
  | (.prompt|to_query) as $text
  | if ($text|endswith("?")|not) then error("authored final request must be interrogative") else . end
  | {input:{case_id:$id,text:$text},gold:{case_id:$id,start:.source_pos,end:(if .source_pos==null then null else .source_pos+1 end),group:$group,swapped_case_id:$swap},truth:{case_id:$id,tape:"author-labelled #4329 original facts and requests",family,asked,fact_order,statement_type,changed_side,site_ids,site_positions,same_value:(.site_ids[0]==.site_ids[1]),source_ids:(if .supported then [.site_ids[.asked]] else null end),expected_ids:(if .supported then [.site_ids[.asked],1] else null end),response_lexical_offset:(if .supported then 0 else null end)}}
),
(
  ($extra_truth|INDEX(.case_id)) as $t
  | ($extra_input|INDEX(.case_id)) as $p
  | $extra_gold[]
  | {input:$p[.case_id],gold:.,truth:($t[.case_id]+{tape:"new author-labelled TRAIN closed world"})}
)
