# Independently join the raw event rows to the frozen panel. jq -n --slurpfile p PANEL --slurpfile t RAW -f audit.jq
def must(ok; why): if ok then . else error(why) end;
def key: [.pair_id, .side] | @json;
def choice($score;$native): if $score<0 then 1 elif $score>0 then 16 else $native end;
def slot($result;$expected):
  [range($expected | length) | select($result[.] != $expected[.])][0];
def n(f): map(select(f)) | length;
def summary: . as $rows | {
  rows:length, stop_rows:n(.side==0), period_rows:n(.side==1),
  copied:n(.ids[0]==.target), token_two_correct:n(.ids[1]==.expected[1]),
  exact:n(.exact), exact_stop:n(.exact and .side==0), exact_period:n(.exact and .side==1),
  exact_pairs:(group_by(.pair_id)|n(length==2 and all(.[];.exact))),
  second_pairs:(group_by(.pair_id)|n(length==2 and all(.[];.ids[1]==.expected[1]))),
  first_wrong:(group_by(.first_wrong)|map({key:(.[0].first_wrong|tostring),value:length})|from_entries),
  retained_native_exact:n(.exact and .native_exact), gained_vs_native:n(.exact and (.native_exact|not)),
  lost_vs_native:n((.exact|not) and .native_exact),
  modal_wrong:(map(select(.first_wrong!=null))|group_by([.first_wrong,.expected[.first_wrong-1],.ids[.first_wrong-1]])|map({position:.[0].first_wrong,expected:(.[0]|.expected[.first_wrong-1]),actual:(.[0]|.ids[.first_wrong-1]),n:length})|sort_by(-.n)|.[:8])
};
must(($p|length)>0 and ($p|length)==($t|length);"panel cardinality") |
(INDEX($p[];key)) as $keys | (INDEX($t[];key)) as $events |
must(($keys|length)==($p|length) and ($events|keys)==($keys|keys);"duplicate or missing row") |
[$t[] as $o | ($keys[$o|key]) as $r |
 ($events[[ $r.pair_id, (1-$r.side)]|@json]) as $other |
 must($o.prefix==$r.prefix and $o.score==$other.peer_score and $o.peer_score==$other.score;"prefix or swapped score") |
 must(($o.arms | keys) == ["learned","native","oracle","swapped","wrong_oracle"];"arm inventory") |
 $o.arms.native as $native | $native.steps[0] as $first |
 must($first.copied and $first.read_pos==$r.gold_pos and $first.claim_pos==$r.gold_pos and $first.committed_id==$r.target and $first.committed_id==$r.prefix[$r.gold_pos];"frozen opening") |
 ([$native.steps[].committed_id]==$r.expected and $native.terminal) as $n_exact |
 $o.arms | to_entries[] | .key as $name | .value as $arm |
 $arm.steps[1].native_proposal as $proposal | [$arm.steps[].committed_id] as $ids |
 must($first==$arm.steps[0] and $ids[0]!=1 and $proposal==$native.steps[1].native_proposal;"opening or native proposal differs") |
 must(($ids|length)>=2 and ($ids|length)<=3 and ($arm.terminal==($ids[-1]==1)) and ($ids[0:-1]|index(1))==null;"EOS overrun or invalid length") |
 must(($ids|length)==3 or $ids[1]==1;"missing native third emission") |
 must(($arm.steps[1:]|all(.[];.copied==false)) and (($ids|length)!=3 or ($arm.steps[2].native_proposal==$ids[2]));"rearmed or forced third action") |
 must($ids[1]==(if $name=="native" then $proposal elif $name=="learned" then choice($o.score;$proposal) elif $name=="swapped" then choice($other.score;$proposal) elif $name=="oracle" then $r.expected[1] elif $r.expected[1]==1 then 16 else 1 end);"action disagrees with intervention") |
 (slot($ids;$r.expected)) as $wrong |
 {arm:$name, pair_id:$r.pair_id, side:$r.side,family:$r.family, target:$r.target, expected:$r.expected, ids:$ids, score:$o.score,peer_score:$other.score, native_exact:$n_exact,exact:($ids==$r.expected and $arm.terminal),first_wrong:(if $wrong==null then null else 1+$wrong end)}] as $results |
 {rows:($p|length),arms:($results|group_by(.arm)|map({key:.[0].arm,value:summary})|from_entries),families:($results|group_by(.family)|map({key:.[0].family,value:(group_by(.arm)|map({key:.[0].arm,value:summary})|from_entries)})|from_entries),
  states:($results|map(select(.arm=="learned"))|{negative:n(.score<0),positive:n(.score>0),zero:n(.score==0),opposite_signed_pairs:(group_by(.pair_id)|n((map(.score>0)|unique|length)==2 and all(.[];.score!=0))),pairs_stop_negative_period_positive:(group_by(.pair_id)|n(all(.[]; if .side==0 then .score<0 else .score>0 end)))})}
