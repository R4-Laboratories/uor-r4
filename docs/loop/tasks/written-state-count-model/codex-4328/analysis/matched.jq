# Inputs are sealed prompt strings and graded diagnostic rows joined back to
# the actual ordinary authenticated client's first two commits by case ID.
def expect($p; $s): if $p then . else error($s) end;
def observation:
  {case_id, group, family, fenced, fact_order, question_target,
   requested: .gold.pos, requested_id: .gold.id, selected: .reader.winner_pos,
   reader_gate, copied: .arms.learned.copied, ids: .arms.learned.committed_ids};
def pair:
  {cases: map(.case_id),
   requested: map(.requested), selected: map(.selected), first_ids: map(.ids[0]),
   source_correct_on_both: all(.[]; .requested == .selected),
   authenticated_on_both: all(.[]; .requested == .selected and .copied),
   first_id_correct_on_both: all(.[]; .requested_id == .ids[0])};
def summary($p):
  {pairs: ($p|length),
   both_sources: ([$p[]|select(.source_correct_on_both)]|length),
   both_authenticated: ([$p[]|select(.authenticated_on_both)]|length),
   both_first_ids: ([$p[]|select(.first_id_correct_on_both)]|length),
   pairs_with_different_selected_positions: ([$p[]|select(.selected[0] != .selected[1])]|length)};
($prompts|map({key:.case_id,value:.prompt})|from_entries) as $text
| expect(length > 0 and length == ($text|length) and
         (map(.case_id)|sort) == ($text|keys); "case census differs from raw prompts")
| map(select(.supported)|observation) as $rows
| ($rows | group_by([.group,.family,.fact_order]) | map(sort_by(.question_target)
    | expect(length == 2 and (map(.question_target)|unique|length)==2 and
             .[0].requested != .[1].requested; "invalid same-facts question swap") | pair)) as $questions
| ($rows | group_by([.group,.family,.question_target]) | map(sort_by(.fact_order)
    | expect(map(.fact_order)==[0,1] and .[0].requested_id == .[1].requested_id;
             "invalid fact-order swap") | pair)) as $facts
| ($rows | group_by($text[.case_id]|gsub("\n\n";" ")) | map(select(length>1)
    | sort_by(.fenced) | expect(length==2 and map(.fenced)==[false,true] and
      .[0].requested_id==.[1].requested_id; "invalid whitespace-only pair")
    | {natural:.[0],fenced:.[1]})) as $fence
| {question_changes:summary($questions), fact_order_changes:summary($facts),
   whitespace_only:$fence, question_pairs:$questions, fact_order_pairs:$facts}
