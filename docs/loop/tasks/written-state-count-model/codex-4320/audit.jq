# Independently aggregate the actual self-fed GPU or client event rows and exact EOS.
# jq -n --slurpfile p panel/rows.jsonl --slurpfile t measured/observations.jsonl -f audit.jq
def ensure(ok; why): if ok then . else error(why) end;
def key: [.pair_id,.side]|@json;
def n(f):map(select(f))|length;
def mismatch($ids;$want):[range($want|length)|select($ids[.]!=$want[.])][0] | if .==null then null else .+1 end;
def summarize:
 {rows:length,pairs:(map(.pair_id)|unique|length),word_and_period:n(.ids[0]==.target and .ids[1]==16),
  third_correct:n(.ids[2]==.expected[2]),third_correct_stop:n(.side==0 and .ids[2]==.expected[2]),
  third_correct_continue:n(.side==1 and .ids[2]==.expected[2]),third_correct_pairs:(group_by(.pair_id)|n(length==2 and all(.[];.ids[2]==.expected[2]))),
  exact:n(.exact),exact_stop:n(.exact and .side==0),exact_continue:n(.exact and .side==1),
  exact_pairs:(group_by(.pair_id)|n(length==2 and all(.[];.exact))),ordinary_eos:n(.terminal and .last_source=="native"),
  first_wrong:(group_by(.first_wrong)|map({key:(.[0].first_wrong|tostring),value:length})|from_entries),
  retained_native:n(.exact and .native_exact),gained_vs_native:n(.exact and (.native_exact|not)),lost_vs_native:n((.exact|not) and .native_exact),
  actual_third:(group_by(.ids[2])|map({id:.[0].ids[2],rows:length})|sort_by(-.rows)),
  fourth_after_supplied_word:(map(select(.side==1 and .ids[2]==.expected[2]))|group_by(.ids[3])|map({id:.[0].ids[3],rows:length})|sort_by(-.rows)),
  wrong_examples:(map(select(.first_wrong!=null))|group_by([.first_wrong,.expected[.first_wrong-1],.ids[.first_wrong-1]])|map({position:.[0].first_wrong,expected:(.[0]|.expected[.first_wrong-1]),actual:(.[0]|.ids[.first_wrong-1]),n:length})|sort_by(-.n)|.[:12])};
ensure(($p|length)>0 and ($p|length)==($t|length);"wrong cardinality") |
(INDEX($p[];key)) as $keys | (INDEX($t[];key)) as $events |
ensure(($keys|length)==($p|length) and ($events|keys)==($keys|keys);"duplicate or missing row") |
[$t[] as $o | $keys[$o|key] as $r | $keys[[$r.pair_id,1-$r.side]|@json] as $peer |
 ensure($o.prefix==$r.prefix and $r.gold_pos==$peer.gold_pos and $r.target==$peer.target and $r.next_pos==$peer.next_pos and $r.next_id==$peer.next_id and $r.family==$peer.family;"pair or tape mismatch") |
 ensure($r.gold_pos!=$r.next_pos and $r.prefix[$r.gold_pos]==$r.target and $r.prefix[$r.next_pos]==$r.next_id and $r.target!=$r.next_id;"source binding") |
 ensure($r.expected==(if $r.side==0 then [$r.target,16,1] else [$r.target,16,$r.next_id,1] end);"reference") |
 ensure(($o.arms|keys)==["correct","native","supplied_next","swapped"];"missing arm") |
 $o.arms.native as $native | ($native.steps|map(.committed_id)) as $native_ids |
 $o.arms|to_entries[] | .key as $a | .value as $arm | ($arm.steps|map(.committed_id)) as $ids |
 ensure($arm.steps[:2]==$native.steps[:2] and $ids[0]==$r.target and $ids[1]==16 and $arm.steps[0].copied and $arm.steps[0].read_pos==$r.gold_pos and $arm.steps[0].claim_pos==$r.gold_pos;"opening or period changed") |
 ensure($arm.steps[0].selected_source=="copy" and $arm.steps[0].source_id==$r.target and
  $arm.steps[1].selected_source=="supplied_period" and
  $arm.steps[2].selected_source==(if $a=="supplied_next" then $a elif ($a=="correct" and $r.side==0) or ($a=="swapped" and $peer.side==0) then $a else "native" end) and
  (($ids|length)!=4 or $arm.steps[3].selected_source=="native");"source provenance") |
 ensure(($ids|length)>=3 and ($ids|length)<=4 and $arm.terminal==($ids[-1]==1) and ($ids[:-1]|index(1))==null;"terminal status or EOS overrun") |
 ensure($ids[2]==(if $a=="native" then $native_ids[2] elif $a=="correct" then (if $r.side==0 then 1 else $native_ids[2] end) elif $a=="swapped" then (if $peer.side==0 then 1 else $native_ids[2] end) else $r.expected[2] end);"completion intervention") |
 ensure($arm.steps[2].native_proposal==$native_ids[2] and (($ids|length)==4 or $ids[2]==1) and (($ids|length)!=4 or $ids[3]==$arm.steps[3].native_proposal);"native proposal changed or missing/forced successor") |
 ensure($arm.first_mismatch==mismatch($ids;$r.expected);"misreported mismatch") |
 {arm:$a,pair_id:$r.pair_id,family:$r.family,side:$r.side,target:$r.target,expected:$r.expected,ids:$ids,first_wrong:$arm.first_mismatch,
  terminal:$arm.terminal,last_source:$arm.steps[-1].selected_source,exact:($ids==$r.expected and $arm.terminal),native_exact:($native_ids==$r.expected and $native.terminal)}] as $rows |
{arms:($rows|group_by(.arm)|map({key:.[0].arm,value:summarize})|from_entries),families:($rows|group_by(.family)|map({key:.[0].family,value:(group_by(.arm)|map({key:.[0].arm,value:summarize})|from_entries)})|from_entries)}
