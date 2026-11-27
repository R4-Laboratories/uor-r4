# Retrospective diagnostic on emitted GPU scores; no threshold is fitted.
def require(test; why): if test then . else error(why) end;
def id: [.pair_id,.side]|@json;
def summarize:
 . as $all | map(select(.[1]-.[0]>=2)|[.[0]+1,.[1]-1]) as $intervals |
 ([$intervals[][0]]|unique|map(. as $point | [$intervals[]|select(.[0]<=$point and $point<=.[1])]|length)|max//0) as $bound |
 {pairs:length, strict_switch:([.[]|select(.[0]<0 and .[1]>0)]|length),
  inverted:([.[]|select(.[0]>0 and .[1]<0)]|length),
  both_stop:([.[]|select(.[0]<0 and .[1]<0)]|length),
  both_copy:([.[]|select(.[0]>0 and .[1]>0)]|length),
  tied_member:([.[]|select(.[0]==0 or .[1]==0)]|length),
  gap_negative:([.[]|select(.[1]<.[0])]|length),
  gap_zero:([.[]|select(.[1]==.[0])]|length),
  individually_separable:($intervals|length), common_threshold_upper_bound:$bound};
($keys|INDEX(.[];id)) as $k |
[$rows[]|. as $r | $k[id] as $truth | require($truth!=null;"missing truth") |
 if $truth.selector_swap_id!=null then {kind:"selector",family:.family,group:$truth.selector_swap_id,
    count:$truth.expected_word_count,score:.arms.learned.steps[2].action_score}
 elif .contrast==1 then .arms.first.steps as $s |
   require($s[2].selected_source=="verified_copy" and $s[2].refusal==0 and $s[2].requested_pos==.positions[1] and $s[2].requested_id==.ids[1] and $s[3].stage==1;"B was not authenticated") |
   {kind:"direct_after_B",family:.family,group:.pair_id,count:$truth.expected_word_count,score:$s[3].action_score}
 else empty end]
| group_by(.kind) | map({key:.[0].kind,value:(group_by(.group)|map(sort_by(.count)|require(length==2 and .[0].count+1==.[1].count;"unmatched prompt")|map(.score))|summarize)})|from_entries
