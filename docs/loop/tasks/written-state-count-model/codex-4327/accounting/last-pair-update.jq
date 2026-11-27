# Compare a pair's final matched update with its scores after all later updates.
# Both time points come from the same GPU trajectory and use strict score signs.
def require(p; why): if p then . else error(why) end;
def strict(field): all(.members[]; .[field]>0);
def n(p): map(select(p)) | length;
def movement($before; $after):
 if $before then (if $after then "kept" else "lost" end)
 else (if $after then "gained" else "missed_both" end) end;
def hist: group_by(.) | map({key:.[0],value:length}) | from_entries;
require(($scores|length)==768 and ($trace|length)==($schedule|length); "tape size mismatch") |
[$trace[] | . as $g | $schedule[.group] as $slot |
 require(.stage==$slot.stage and .kind==$slot.kind and
  ([.members[].train_index]==[$slot.first,$slot.second|select(.!=null)|.train_index]);"changed group identity") |
 select((.members|length)==2) | .members |= map(. as $m |
    require((.copy|type)=="boolean" and ($scores[.train_index][$g.stage]|type)=="number"; "missing final GPU score") |
    . + {final_margin:((if .copy then 1 else -1 end)*$scores[.train_index][$g.stage])}) |
 {group,kind,stage, members, key:([.members[].train_index]|sort)}]
| group_by([.kind,.key]) | map(max_by(.group)) | group_by(.kind) | map({
 kind:.[0].kind, pairs:length, before_last:n(strict("signed_margin_before_group")),
 after_last:n(strict("signed_margin_after_group")), final:n(strict("final_margin")),
 last_to_final:(map(movement(strict("signed_margin_after_group");strict("final_margin")))|hist),
 examples_of_erasure:(map(select(strict("signed_margin_after_group") and (strict("final_margin")|not)))|.[0:1]|
 map({group, members:[.members[]|{train_index,copy,signed_margin_before_group,signed_margin_after_group,final_margin}]}))
})
