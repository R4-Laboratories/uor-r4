# Post-hoc audit of the existing native_word_ending::before_instruction token contract.
# The three carriers reuse precisely these features in native_source_choice::score.
def instruction:
  . as $p | length as $n
  | if $p[-6:] != [201,35,560,652,714,28] then {reason:"no_assistant_marker"}
    else ($n-6) as $end
      | ([range(0;$end-1)|select($p[.:.+2]==[201,201])]|last) as $fence
      | if $fence==null then {reason:"no_fact_query_fence"}
        else ([range($fence+2;$end)|select($p[.]==33)]|last) as $question
        | if $question==null then {reason:"no_question_id_after_fence"}
          else {reason:"located",start:($question+1),end:$end,tokens:($end-$question-1)}
          end
        end
    end;
($truth|map({key:.case_id,value:(.prompt_ids|instruction)})|from_entries) as $scope
| .rows | map(select(.release!=null) | . + {instruction:$scope[.case_id]}
  | if .instruction.reason!="located" and .release.score!=0 then error("unlocated instruction scored nonzero: "+.case_id) else . end)
| group_by([.arm,.instruction.reason]) | map({
   arm:.[0].arm, scope:.[0].instruction.reason, rows:length,
   verified:(map(select(.verified_boundary))|length),
   negative:(map(select(.release.score<0))|length),zero:(map(select(.release.score==0))|length),
   positive:(map(select(.release.score>0))|length), ids:map(.case_id)})
