def count_where(f): map(select(f))|length;
def spans($row;$cut):
   [$row.bound[]|select(.query[0]>=$cut and .fact[1]<=$cut)];
def assess($row;$key;$cut):
   spans($row;$cut) as $bounds
   | [$bounds[].ranges[]] | unique as $ranges
   | [$bounds[].selected|select(.!=null)]|unique as $chosen
   | {bound_count:($bounds|length),range_count:($ranges|length),chosen:$chosen,
      starts:[$ranges[][0]]|unique,has_gold:any($ranges[];.[0]==$key.target.pos),
      selected_gold:($key.target!=null and ($chosen|index($key.target.pos)!=null))};
($truth|INDEX(.case_id)) as $keys
| map(. as $r | $keys[.case_id] as $k
    | (if $k.target then [$r.copies[]|select(.pos==$k.target.pos and .id==$k.target.id)] else [] end) as $copies
    | if $k==null or ($k.target!=null and (($copies|length)!=1 or $copies[0].bytes[0]!=$k.target.bytes[0])) then error("unverified source join") else . end
    | {id:$r.case_id,kind:$k.kind,supported:($k.target!=null),gold:$k.target.pos,
       gold_copy_complete:$copies[0].complete,release:$copies[0].release,
       exact_span:($k.target!=null and $copies[0].complete and $copies[0].bytes==$k.target.bytes),
       native:$r.native,
       tail:assess($r;$k;$r.tail_cut),separator:assess($r;$k;$r.separator)}) as $joined
| if ($joined|length)!=($truth|length) or ($joined|map(.id)|unique|length)!=($truth|length) then error("incomplete joined panel") else . end
| {rows:($joined|length),supported:($joined|count_where(.supported)),unsupported:($joined|count_where(.supported|not)),
   models:[range(0;3) as $i | {sha:$joined[0].native[$i].artifact,source_bank:$joined[0].native[$i].bound_source_bank,
      requested_framing:$joined[0].native[$i].requested_framing,
      paired_entry:($joined|count_where(.native[$i].paired_decision!=null)),
      role_requests:($joined|count_where(.native[$i].paired_decision.role_words[0]!=null and .native[$i].paired_decision.role_words[0]!=4294967295)),
      supporting_fact:($joined|count_where(.native[$i].paired_decision.supporting_facts>0)),
      selected:($joined|count_where(.native[$i].paired_decision.choice.source!=null)),
      correct:($joined|count_where(.supported and .native[$i].paired_decision.choice.source==.gold)),
      false_admissions:($joined|count_where((.supported|not) and .native[$i].paired_decision.choice.source!=null))}],
   oracle_binder_enumeration:["tail","separator"]|map(. as $view | {
      cut:$view,rows_with_legal_binding:($joined|count_where(.[$view].bound_count>0)),
      rows_with_candidates:($joined|count_where(.[$view].range_count>0)),
      supported_gold_covered:($joined|count_where(.supported and .[$view].has_gold)),
      supported_selected_in_any_binding:($joined|count_where(.supported and .[$view].selected_gold)),
      unsupported_any_candidates:($joined|count_where((.supported|not) and .[$view].range_count>0)),
      unsupported_any_selections:($joined|count_where((.supported|not) and ((.[$view].chosen|length)>0)))}),
   supplied_copy:{complete:($joined|count_where(.supported and .gold_copy_complete)),
       exact:($joined|count_where(.exact_span)),
       cap:($joined|count_where(.supported and .release==3)),
       completed_but_mismatched:($joined|count_where(.supported and .gold_copy_complete and (.exact_span|not)))},
   cases:[$joined[]|del(.native)]}
