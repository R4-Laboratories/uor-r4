# Independent scorer of original token IDs, conditional action flips and complete replies.
def ck($ok;$why): if $ok then . else error($why) end;
def key:[.pair_id,.side]|@json;
def sign($s):if $s<0 then "eos" elif $s>0 then "copy" else "native" end;
def ct(f):map(select(f))|length;
def hist(f): group_by(f)|map({key:(.[0]|f|tostring),value:length})|from_entries;
def wrong($ids;$ref): ([range(0;([$ids|length,$ref|length]|max)) as $i|select($ids[$i]!=$ref[$i])|$i+1]|first);
def results:
 . as $rows | {rows:length,exact:ct(.exact), exact_short:ct(.side==0 and .exact),exact_long:ct(.side==1 and .exact),
 exact_pairs:(group_by(.pair_id)|ct(all(.[];.exact))), reached_both:(group_by(.pair_id)|ct(all(.[];.reached))),
 reached_both_active:(group_by(.pair_id)|ct(all(.[];.active))),gained_vs_native:ct(.exact and (.native_exact|not)),lost_vs_native:ct((.exact|not) and .native_exact),
 natural_eos:ct(.terminal and .end_source=="native"),selected_eos:ct(.terminal and .end_source=="selected_eos"),open_cap:ct(.terminal|not),
 first_wrong_short:(map(select(.side==0))|hist(.first_wrong)),first_wrong_long:(map(select(.side==1))|hist(.first_wrong)),
 wrong_transitions:(map(select(.first_wrong!=null))|group_by([.side,.first_wrong,.wanted,.actual])|map({side:.[0].side,position:.[0].first_wrong,wanted:.[0].wanted,actual:.[0].actual,n:length})|sort_by(-.n))};
def flips:
 group_by(.pair_id)|map(sort_by(.side)|.[0] as $a|.[1] as $b|
 {actions:[$a.choice,$b.choice],score_difference:($b.score-$a.score),correct:($a.choice=="eos" and $b.choice=="copy"),backwards:($a.choice=="copy" and $b.choice=="eos")})|
 {pairs:length, correct:ct(.correct),inverted:ct(.backwards),any_sign_flip:ct(.correct or .backwards),any_abstention:ct(.actions|index("native")),
   score_order:hist(.score_difference|if .>0 then "ordered" elif .==0 then "tied" else "reversed" end),
   decisions:(group_by(.actions)|map({choices:.[0].actions,n:length}))};
def summarize: {by_arm:(group_by(.arm)|map({key:.[0].arm,value:results})|from_entries),conditional:(map(select(.arm=="learned"))|flips)};
ck(($p|length)>0 and ($p|length)==($t|length);"cardinality") |
(INDEX($p[];key)) as $refs | (INDEX($t[];key)) as $obs |
ck(($p|length)==($refs|length) and ($obs|keys)==($refs|keys);"missing or duplicate") |
[$t[] as $o | $refs[$o|key] as $r | $refs[[$r.pair_id,1-$r.side]|@json] as $other |
 ck($r.prefix==$o.prefix and $r.ids==$other.ids and $r.positions==$other.positions and $r.contrast==$other.contrast and $r.family==$other.family and ($r.prefix!=$other.prefix);"pair") |
 ck($r.expected==([$r.ids[0],16]+$r.ids[1:1+$r.contrast+$r.side]+[1]) and all(range(4);. as $i|$r.prefix[$r.positions[$i]]==$r.ids[$i]) and ($r.ids|unique|length)==4 and ($r.positions|unique|length)==4;"reference") |
 ck($o.common.stage==$r.contrast and $o.common.own_choice==sign($o.common.own_score) and $o.common.peer_score==$obs[$other|key].common.own_score and $o.common.peer_choice==sign($o.common.peer_score);"independent counterfactual probe") |
 ck(($o.arms|keys)==["correct","learned","native","opposite","swapped"];"arm names") |
 $o.arms.native as $n | ($n.steps|map(.committed_id)) as $native |
 $o.arms|to_entries[] | .key as $a|.value as $v|($v.steps|map(.committed_id)) as $ids |
 ck($v.steps[:2]==$n.steps[:2] and $ids[:2]==[$r.ids[0],16] and $v.steps[0].copied and $v.steps[0].source_id==$r.ids[0] and $v.steps[0].claim_pos==$r.positions[0] and $v.steps[0].read_pos==$r.positions[0] and $v.steps[1].selected_source=="supplied_period";"fixed opening") |
 ck($v.terminal==($ids[-1]==1) and ($ids[:-1]|index(1))==null and (($ids|length)==5 or $v.terminal) and ($ids|length)<=5;"terminal or cap") |
 ck(all($v.steps[2:][]; if .selected_source=="selected_eos" then .committed_id==1 elif .selected_source=="verified_copy" then .stage as $j|.requested_pos==$r.positions[$j+1] and .requested_id==$r.ids[$j+1] and .committed_id==$r.ids[$j+1] elif .selected_source=="native" then .committed_id==.native_proposal else false end);"release provenance") |
 ($r.contrast+2) as $c| ($ids[:$c]==$r.expected[:$c] and ($ids|length)>$c) as $reach |
 ck(if $reach and $v.steps[$c].stage==$r.contrast and ($a=="learned" or $a=="swapped") then ($v.steps[$c].action_score==(if $a=="learned" then $o.common.own_score else $o.common.peer_score end)) else true end;"actual versus conditional") |
 ck(if $a=="correct" then ($ids==$r.expected and $v.terminal) elif $a=="opposite" then ($ids==$other.expected and $v.terminal) else true end;"oracle not exact for assigned instruction") |
 (wrong($ids;$r.expected)) as $w | ck($w==$v.first_mismatch;"mismatch") |
 {arm:$a,pair_id:$r.pair_id,family:$r.family,side:$r.side,contrast:$r.contrast,terminal:$v.terminal,end_source:$v.steps[-1].selected_source,exact:($ids==$r.expected and $v.terminal),native_exact:($native==$r.expected and $n.terminal),
 first_wrong:$w,actual:(if $w then $ids[$w-1] else null end),wanted:(if $w then $r.expected[$w-1] else null end),
 reached:$reach,active:($reach and $v.steps[$c].stage==$r.contrast),score:$o.common.own_score,choice:$o.common.own_choice}] as $rows |
 {overall:($rows|summarize),contrasts:($rows|group_by(.contrast)|map({key:(.[0].contrast|tostring),value:summarize})|from_entries),families:($rows|group_by([.contrast,.family])|map({contrast:.[0].contrast,family:.[0].family,actions:(map(select(.arm=="learned"))|flips),learned:(map(select(.arm=="learned"))|results|del(.wrong_transitions))}))}
