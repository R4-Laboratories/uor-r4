# Usage: jq -n --slurpfile p PANEL --slurpfile t OBS -f audit-transitions.jq
def check($condition; $why): if $condition then . else error($why) end;
def incount($xs; f): [$xs[] | select(f)] | length;
def goodfirst: .first.committed_id == .gold;
def goodtwo: goodfirst and .successor != null and .successor.committed_id == .second;
def metrics($xs; f):
 {rows:incount($xs;f),
  pairs:([$xs | group_by([.group,.assignment])[] | select(length==2 and all(.[];f))]|length),
  groups:([$xs | group_by(.group)[] | select(length==4 and all(.[];f))]|length)};
check(($p|length)>0 and ($p|length)==($t|length);"length mismatch") |
[range($p|length) as $i | $p[$i] as $r | $t[$i] as $obs |
 check($r.pair_id==$obs.pair_id and $r.prefix==$obs.prefix;"misbound pair or raw prefix") |
 check(([$obs.arms|keys[]]|length)==8;"missing arm") |
 check(($obs.arms.learned.first.native_proposal==$obs.arms.generate.first.committed_id)
       and ($obs.arms.learned.first.read_pos==$obs.arms.generate.first.read_pos);"learned reader or native proposal changed") |
 check(([$obs.arms[]|select(.successor!=null)|.successor.copied]|all(.==false));"second copy") |
 check(([$obs.arms[]]|group_by(.first.committed_id)|all(.[]; ([.[].successor]|unique|length)==1));"inconsistent second event given identical committed ID") |
 check(($obs.arms.correct.first.native_proposal==$obs.arms.missing.first.committed_id and
        $obs.arms.correct.first.read_pos==$obs.arms.missing.first.read_pos and
        $obs.arms.correct.first.gate_id==$obs.arms.missing.first.gate_id);"gold-address native writer control differs") |
 $obs.arms|to_entries[]|.key as $arm|.value as $a |
 check((($a.first.committed_id==1)==($a.successor==null));"continued through EOS or missing successor") |
 check((if $a.first.copied then $a.first.read_pos!=null and $a.first.read_pos==$a.first.claim_pos and
       $r.prefix[$a.first.read_pos]==$a.first.committed_id and $a.first.source_admitted and
       $a.first.source_distance==$a.first.minimum_distance else $a.first.committed_id==$a.first.native_proposal end);"invalid write evidence") |
 check((if ($arm=="missing" or $arm=="invalid" or $arm=="same_id_wrong_occurrence") then ($a.first.copied|not) else true end);"corrupt address accepted") |
 {cohort:$r.cohort,group:$r.group_id,assignment:$r.assignment,side:$r.side,stratum:$r.stratum,arm:$arm,
  gold:$r.target,second:$r.expected_second,first:$a.first,successor:$a.successor}] as $rows |
 {all_prompts:($p|length),cohorts:($p|group_by(.cohort)|map({key:.[0].cohort,value:length})|from_entries),
  controlled: ([$rows[]|select(.cohort=="controlled")]|group_by(.arm)|map(. as $x|
    {key:.[0].arm, value:{first:metrics($x;goodfirst),two:metrics($x;goodtwo),terminal_at_first:incount($x;.first.committed_id==1), with_successor:incount($x;.successor!=null),copied:incount($x;.first.copied)}})|from_entries)}
