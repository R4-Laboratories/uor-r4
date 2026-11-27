# Bind the diagnostic to the original causal prefix, the exact parent weight words,
# and the frozen GPU scores. Never infer addresses from a decoded sentence.
def require($ok;$why): if $ok then . else error($why) end;
def identity: [.pair_id,.side]|@json;
def h: group_by(.)|map({key:.[0],value:length})|from_entries;
def addterms: (map(.weight)|add)//0;
def bare: rtrimstr(".");
def support:
  [.visits[] |
    (select(.root_term!=null)|{bucket:.root_bucket,weight:.root_term}),
    (select(.adjacency_term!=null)|{bucket:.adjacency_bucket,weight:.adjacency_term})] | sort_by(.bucket);
($panel|INDEX(.[];identity)) as $src |
($keys|INDEX(.[];identity)) as $names |
require(($panel|length)>0 and ($panel|length)==($rows|length) and ($keys|length)==($rows|length) and ($src|length)==($rows|length) and ($names|length)==($rows|length);"duplicate or missing input") |
[$rows[] as $r | $src[$r|identity] as $p | $r|support as $terms |
 require($p!=null and $p.family==$r.family and $p.contrast==$r.contrast and $p.prefix[$r.instruction_start:-6]==[$r.visits[].id] and ($terms|length)>0 and ($terms|unique_by(.bucket)|length)==($terms|length) and $r.bias+($terms|addterms)==$r.baseline_score and $r.baseline_score==$r.scores[0] and $r.residual_score==0;"slice, support or score mismatch") |
 {pair_id:$r.pair_id,side:$r.side,family:$r.family,axis:($names[$r|identity].pair_axis // "consumed"),wording:($names[$r|identity].directive|bare),ids:[$r.visits[].id],score:$r.scores[0],peer:$r.peer_scores[0],terms:$terms}] |
group_by(.pair_id)|map(sort_by(.side) | require(length==2 and .[0].side==0 and .[1].side==1 and .[0].peer==.[1].score and .[1].peer==.[0].score;"invalid matched pair") |
 .[0] as $s | .[1] as $c | ($s.terms-$c.terms) as $left | ($c.terms-$s.terms) as $right |
 require(($right|addterms)-($left|addterms)==$c.score-$s.score;"unreconstructed pair difference") |
 {pair_id:$s.pair_id,family:$s.family,axis:$s.axis,wording:[$s.wording,$c.wording],identical_slice:($s.ids==$c.ids),
  equal_length:(($s.ids|length)==($c.ids|length)),same_token_bag:(($s.ids|sort)==($c.ids|sort)),
  scores:[$s.score,$c.score],exclusive_counts:[($left|length),($right|length)],exclusive_sums:[($left|addterms),($right|addterms)],
  nonzero_difference_terms:([$left[],$right[]]|map(select(.weight!=0))|length),
  class:(if $s.ids==$c.ids then "identical_slice" elif $left==[] and $right==[] then "identical_support"
   elif all(($left[],$right[]);.weight==0) then "all_zero"
   elif $s.score==$c.score then "cancellation" elif $s.score<0 and $c.score>0 then "correct_flip"
   elif $c.score<$s.score then "reverse_order" else "ordered_without_flip" end)}) as $pairs |
{counts:($pairs|map(.class)|h),bare_wordings:($pairs|map(.wording)|unique|length),
 wordings:($pairs|group_by(.wording)|map({family:.[0].family,wording:.[0].wording,counts:(map(.class)|h)})),
 pairs:$pairs}
