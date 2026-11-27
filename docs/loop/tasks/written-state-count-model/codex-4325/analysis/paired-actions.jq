# Descriptive audit of already emitted actions. A stage-1 comparison conditions on supplied stage 0 only.
def acts($arm):
 (.[$arm].ids) as $emitted |
 (if $arm=="learned" then 2 else 3 end) as $idx |
 if $arm!="learned" and (.[$arm].b_reached|not) then "unreached"
 elif $emitted[$idx]==1 then "stop"
 elif $emitted[$idx]==.ids[$idx-1] then "copy" else "other" end;
def classify($arm):
 sort_by(.side) |
 if length!=2 or (map(.side)|sort)!=[0,1] then error("paired policy mismatch") else . end |
 [.[0]|acts($arm)] + [.[1]|acts($arm)] |
 if .==["stop","copy"] then "correct_flip" elif .==["copy","stop"] then "inverted"
 elif .==["stop","stop"] then "both_stop" elif .==["copy","copy"] then "both_copy" else "other" end;
def hist($arm):map(classify($arm))|group_by(.)|map({key:.[0],value:length})|from_entries;
{scoped_selector_at_period:(map(select(.one!=null))|group_by([.group,.one])|hist("learned")),
 direct_two_three_after_authenticated_b:(map(select(.one==null and .contrast==1))|group_by(.pair_id)|hist("after_supplied_first"))}
