# Compare independent instruction types, separating punctuation variants and fact replication.
def sign: if .>0 then "copy" elif .<0 then "eos" else "native" end;
def counts(f):group_by(f)|map({value:(.[0]|f),pairs:length});
(INDEX($k[];[.pair_id,.side]|@json)) as $keys |
[$t|group_by(.pair_id)[]|sort_by(.side) as $pair |
 $pair[0] as $a|$pair[1] as $b|$keys[[$a.pair_id,0]|@json] as $ka|$keys[[$a.pair_id,1]|@json] as $kb|
 {contrast:$a.contrast,family:$a.family,variant:$ka.variant,
   marked:($ka.directive|endswith(".")),short_instruction:$ka.directive,long_instruction:$kb.directive,
   scores:[$a.common.own_score,$b.common.own_score],choices:[$a.common.own_score,$b.common.own_score]|map(sign),
   learned_exact_pair:($a.arms.learned.terminal and $b.arms.learned.terminal and $a.arms.learned.first_mismatch==null and $b.arms.learned.first_mismatch==null)}] |
group_by([.contrast,.family,.variant,.marked])|map({contrast:.[0].contrast,family:.[0].family,variant:.[0].variant,marked:.[0].marked,pairs:length,
 short_instruction:.[0].short_instruction,long_instruction:.[0].long_instruction,
 score_pairs:counts(.scores),choices:counts(.choices),complete_pairs:(map(select(.learned_exact_pair))|length)})
