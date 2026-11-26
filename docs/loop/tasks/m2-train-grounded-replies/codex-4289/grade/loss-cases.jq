def ck($ok;$reason): if $ok then . else error($reason) end;
def intv: [.start,.end];
def addr: {bytes:(.bytes|intv),tokens:(.tokens|intv)};
def word($text;$bytes): $text[$bytes[0]:$bytes[1]];
def object($text;$a): if $a==null then null else ($a+{word:word($text;$a.bytes)}) end;
def kind($t;$target;$other;$neutral;$v):
  if $v==null then "none" elif $v==$neutral then "neutral_same_spelling_occurrence"
  elif $v==$other then "other_signed_noun" elif word($t;$v.bytes)==word($t;$target.bytes) then "other_same_spelling"
  else "other_word" end;
($reference[0]) as $refs |
ck(($refs|length)==256 and ($coords|length)==256;"missing case reference") |
inputs as $p | $coords[$p.row] as $c | $refs[$p.row] as $r | $r.turns[1] as $text |
ck($p.sign==0 or $p.sign==1;"invalid signed edge") |
(if $p.sign==0 then $r.governors.affirmative else $r.governors.excluded end) as $g |
($r.native_scope[1-$p.sign] | {bytes:[(.bytes.start-1),(.bytes.end-1)],tokens:(.tokens|intv)}) as $other |
{bytes:($r.neutral_rival_bytes|intv),tokens:($r.neutral_rival.tokens|intv)} as $neutral |
ck($r.case_id==$c.id and $c.ordinal==$p.row and $c.prompt_sha256==$r.structural_prompt_sha256
 and ($text|length)==($text|utf8bytelength) and $p.governor==($c.words[$p.governor_word]|addr)
 and $p.gold==($c.sites[$p.gold_site]|addr) and $p.governor.bytes==($g.bytes|intv) and
 $p.governor.tokens==$g.tokens and $p.gold.bytes==($g.noun|intv) and $p.gold.tokens==($g.native_noun|intv);
 "unauthenticated governor or target address") |
($p.candidates|INDEX(.site|tostring)) as $sites |
($sites[$p.before.rival_site|tostring].address) as $first |
($sites[$p.after.rival_site|tostring].address) as $last |
ck(all($p.candidates[]; . as $s | $s.address==($c.sites[$s.site]|addr) and $s.site!=$p.gold_site) and
  ($p.candidates|map(.site)|length)==($p.candidates|map(.site)|unique|length);"duplicated or invalid wrong source") |
{row:$p.row,sign:$p.sign,id:$r.case_id,pair:$r.pair_id,family:$r.family,distance:$r.distance,admission:$r.admission,
 eligible:$p.eligible_for_fit,gold_in_slot:$p.gold_in_slot,governor_admitted:$p.governor_admitted,rivals_in_pool:$p.eligible_rivals,
 governor:object($text;$p.governor),gold:object($text;$p.gold),before:$p.before.outcome,after:$p.after.outcome,
 old_margin:$p.before.margin,new_margin:$p.after.margin,old_hinge:$p.before.loss,new_hinge:$p.after.loss,
 old_kind:kind($text;$p.gold;$other;$neutral;$first),new_kind:kind($text;$p.gold;$other;$neutral;$last),
 old_rival:object($text;$first),new_rival:object($text;$last),old_tie_count:$p.before.rival_ties,new_tie_count:$p.after.rival_ties,
 current_pivot_equals_old:($p.before.rival_site==$p.after.rival_site),old_pivot_after:($p.initial_rivals_after[0].after.margin),
 beats_all_old_rivals:($p.eligible_for_fit and ($p.initial_rivals_after|length)>0 and all($p.initial_rivals_after[];.after.margin>0)),
 chosen_step:$p.chosen_step,transition:$p.flip}
