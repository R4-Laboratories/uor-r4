def require(p; m): if p then . else error(m) end;
def counter(predicate): map(select(predicate))|length;
def person_pair:
  sort_by(.asked) | require(map(.asked)==[0,1]; "asked-person pair incomplete")
  | require(.[0].site_positions==.[1].site_positions and .[0].site_ids==.[1].site_ids;
            "question change moved underlying factual positions")
  | if .[0].changed_side==null then
      require(all(.[];.supported) and (map(.source_pos)|unique|length)==2; "affirmation pair has no source switch")
    else require(counter(.supported)==1; "negative statement leaked the answer to question identity") end;
require((map(.prefix)|unique|length)==length and all(.[]; (.prefix|length)<=188 and
  (.site_positions|unique|length)==2 and
  (.site_ids==[.prefix[.site_positions[0]],.prefix[.site_positions[1]]]) and
  (.supported==(.asked!=.changed_side)) and
  (if .supported then .source_pos==.site_positions[.asked] and .expected_ids[0]==.prefix[.source_pos]
   else .source_pos==null and .expected_ids[0]==2823 end));"invalid address, label, expected ID or duplicate prefix")
| (group_by([.family,.fact_order,.value_assignment,.changed_side,.statement_type])|map(person_pair)) as $pairs
| {split:(map(.split)|unique), families:(map(.family)|unique), rows:length,
  positives:counter(.supported),no_read:counter(.supported|not),
  asserted_to_other_person_or_denied_pairs:([$pairs[]|select(.[0].changed_side!=null)]|length),
  both_asserted_question_switches:([$pairs[]|select(.[0].changed_side==null)]|length),
  same_id_distinct_source_questions:([$pairs[]|select(.[0].changed_side==null and .[0].site_ids[0]==.[0].site_ids[1])]|length),
  states:(group_by(.statement_type)|map({kind:.[0].statement_type,positive:counter(.supported),no_read:counter(.supported|not)}))}
