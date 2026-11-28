# Link checked integer inequalities to the raw full-code and mask observations.
($caps|INDEX(.case_id)) as $r | ($keys|group_by(.pair_id)|INDEX(.[0].pair_id)) as $by_pair |
[.outcomes[]|select(.arm=="actual" and .margin==1 and .scope=="annotated")|.pairs[] |
  . as $result | $by_pair[.pair] as $pair | $r[$pair[0].case_id] as $first |
  $r[$pair[1].case_id] as $second |
  {pair,kind,feasible:(.verdict.result=="Feasible"),
   identical_query_code:($first.query_code==$second.query_code),
   query_codes:[$first.query_code,$second.query_code],
   identical_aligned_mask_difference:all(.contrast.aligned_difference[];.==0),
   edges:.contrast.constraints, excess:.verdict.excess,
   target_positions:[$pair[].target.pos],
   sites:[$pair[0].sites[] as $s |
     {pos:$s.pos,id:$s.id,
      observations:[($first,$second) | .candidates[] | select(.pos==$s.pos) |
        {code:.recurrence_code,mask:.real_mask,base_score:.score}]}]}]
