# Compare against the exact original successful trajectories by pair and side.
def key: [.pair_id,.side]|@json;
def exact: . as $row | .arms.learned | .terminal and ([.steps[].committed_id]==$row.expected);
(INDEX($prior[];key)) as $original |
[$measured[] | . as $row | $original[$row|key] as $old |
 if $old==null or
    ($old|{pair_id,side,contrast,family,positions,ids,prefix,expected}) !=
    ($row|{pair_id,side,contrast,family,positions,ids,prefix,expected}) or
    ($old|exact|not)
 then error("protected row is not an original exact result")
 else {pair_id,side,exact:exact} end] | group_by(.pair_id) |
if length!=50 or any(.[]; ([.[].side]|sort)!=[0,1])
then error("protected census")
else {original_exact_pairs:length,
      retained_exact_pairs:(map(select(all(.[];.exact)))|length),
      lost_exact_pair_ids:(map(select(any(.[];.exact|not))|.[0].pair_id))}
end
