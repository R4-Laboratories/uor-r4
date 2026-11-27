def assert($ok;$s): if $ok then . else error($s) end;
def arm($r;$a): $r.arms[$a];
(length) as $n |
(map(select(.pair_id=="probe-261"))[0]) as $lex |
(map(select(.pair_id=="probe-262"))[0]) as $alias |
(map(select(.pair_id=="instrument-same-token-other-address"))[0]) as $dup |
(map(select(.pair_id=="finite-unscoped"))[0]) as $bad |
(map(select(.pair_id=="finite-empty"))[0]) as $empty |
[.[]|select(.pair_id=="f2b1f2d83ca48e0066ac8c26" or .pair_id=="94524602b35e0d22e9bb8164")] as $eos |
assert($n==9 and ($eos|length)==4;"finite coverage") |
assert(($lex.arms.correct.first|.copied and .committed_id==261 and .native_proposal==261);"lexical positive") |
assert(($alias.arms.correct.first|.copied and .committed_id==262 and .native_proposal==6);"non-EOS alias") |
assert(($eos|all(.[]; (.arms.correct.first|.copied and .native_proposal==1) and .arms.missing.successor==null and .arms.correct.successor!=null and (.arms.correct.first.committed_id==1808 or .arms.correct.first.committed_id==3984) and .arms.correct.first.committed_id!=.arms.swapped.first.committed_id));"collision or self-feed") |
assert(($dup.arms.same_id_wrong_occurrence.first|.source_id==1808 and .claim_pos!=.read_pos and (.copied|not) and .committed_id==1);"duplicate identity") |
assert(([$empty,$bad]|all(.[]; (.arms.correct.first.copied|not) and (.arms.learned.first.copied|not)));"missing or unsupported source") |
{status:"pass",rows:$n,lexical_positive:261,non_eos_alias:[262,6],eos_collision:[1808,3984],same_id_wrong_address_rejected:true,empty_and_malformed_refused:true}
