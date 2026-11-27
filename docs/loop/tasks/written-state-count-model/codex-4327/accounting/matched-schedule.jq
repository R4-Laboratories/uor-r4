# Compare each occurrence at its original within-epoch/stage offset. Equal totals
# alone would not detect swapping one rehearsal or target identity for another.
def require(ok; why): if ok then . else error(why) end;
[$prior | to_entries[] | .key as $source_slot | .value | {$source_slot, epoch,stage:.joint_stage, train_index, pair_id,side,copy:.joint_copy}]
| group_by([.epoch,.stage]) | map(sort_by(.source_slot)|to_entries | map(.value + {offset:.key}|del(.source_slot))) | add as $old |
[$groups[]|. as $g | [.first,.second][] | select(. != null) | {
  epoch:$g.epoch,stage:$g.stage,train_index,pair_id,side,copy,offset:.original_offset}] as $new |
($old|sort_by(.epoch,.stage,.offset)) as $left |
($new|sort_by(.epoch,.stage,.offset)) as $right |
require(($left|length)==6144 and $left==$right; "exposure, original offset or target changed") |
require(all($groups[]; .second==null or (.first.copy != .second.copy and .first.train_index != .second.train_index)); "not opposed") |
{exposures:($right|length), opportunities:(($right|length)*16), original_offsets_reproduce_all_identities_and_labels:true,
 by_stage:($right|group_by(.stage)|map({stage:.[0].stage, exposures:length,copy:(map(select(.copy))|length),stop:(map(select(.copy|not))|length)})),
 by_group:($groups|group_by([.stage,.kind])|map({stage:.[0].stage,kind:.[0].kind,groups:length,members:([.[]|1+(if .second==null then 0 else 1 end)]|add)}))}
