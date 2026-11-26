# Join measured candidate costs and joint choices to the original prompt only after collection.
def require($ok;$message): if $ok then . else error($message) end;
def interval: [.start,.end];
def address: {bytes:(.bytes|interval),tokens:(.tokens|interval)};
def original_word($text;$addr):
  $addr.bytes as [$a,$b] |
  def at($k): if ($text|length)==($text|utf8bytelength) then $k else
    first(range(($text|length)+1) as $i|select(($text[:$i]|utf8bytelength)==$k)|$i) end;
  [at($a),at($b)] as $ends |
  require($a>=0 and $b>$a and $b<=($text|utf8bytelength) and ($ends|length)==2;"invalid source byte interval") |
  $text[$ends[0]:$ends[1]];
def view($text;$coord;$id):
  if $id==null or $id<0 then null else
    ($coord.sites[$id]|address) as $a |
    {site:$id,word:original_word($text;$a),bytes:$a.bytes,bpe:$a.tokens} end;
def cost_view($text;$coord): . as $x | view($text;$coord;.site)+{cost:$x.cost[1],center:$x.transport_center};
def mask_site($id): if $id==null or $id<0 then null else $id end;

require(($panel|length)==48 and ($raw|length)==256 and ($rank|length)==48 and ($oldrank|length)==48 and
   ($ref[0]|length)==256 and ($coord|length)==256 and ($joined|length)==256 and ($tsv|length)==256 and
   ($panel|map(.row))==($rank|map(.row)) and ($rank|map(.row))==($oldrank|map(.row)) and
   ($rank|map(.row)|unique|length)==48;"physical row enrollment changed") |
range(48) as $j | $panel[$j] as $p | $p.row as $i | $p.failed_sign as $sg |
($tsv[$i]|split("\t")|map(tonumber)) as $mask |
$ref[0][$i] as $original | $coord[$i] as $c | $original.turns[1] as $text |
$raw[$i] as $read | $joined[$i] as $jrow | $rank[$j] as $r | $oldrank[$j] as $before |
($p.links[$sg].gold_site) as $gold | $p.links[$sg].governor_word as $word |
(if $arm=="winner" then mask_site($p.winner_site) elif $arm=="control" then mask_site($p.control_site) else null end) as $omit |
require($p.case_id==$original.id and $p.pair_id==$original.pair_id and $c.ordinal==$i and
   $c.id==$original.id and $c.prompt_sha256==$original.structural_prompt_sha256 and $read.row==$i and
   $jrow.case_id==$original.id and ($sg==0 or $sg==1) and
   ($mask|length)==3 and $mask[0]==$sg and $mask[1]==$p.winner_site and
   mask_site($mask[2])==mask_site($p.control_site) and
   $r.sign==$sg and $r.word==$word and $r.governor==($c.words[$word]|address) and $r.governor==$before.governor and
   $r.requested_site==$omit and $r.omitted_site==$omit and $before.omitted_site==null and
   ($before.candidates|map(.site))==($before.candidates|map(.site)|unique) and
   all($before.candidates[]; .noun==($c.sites[.site]|address) and (.cost|length)==2 and all(.cost[];type=="number") and (.transport_center|type)=="number") and
   ($before.candidates|map(.cost[0])|unique|length)==1 and
   $r.candidates==[$before.candidates[]|select(.site!=$omit)] and
   ($before.candidates|length)-($r.candidates|length)==(if $omit==null then 0 else 1 end);
   "wrong original role/address, mask or unmasked candidate costs") |
([$r.candidates[]|select(.site==$gold)]) as $match |
require(($match|length)==1 and $omit!=$gold and ($r.candidates|length)>=2;"gold disappeared from eligible list") |
$match[0].cost[1] as $gc |
([$r.candidates[].cost[1]]|min) as $minimum |
([$r.candidates[]|select(.site!=$gold)|.cost[1]]|min) as $wrong |
([$r.candidates[].site] | length) as $n |
{row:$i,case_id:$p.case_id,pair_id:$p.pair_id,admission:$original.admission,family:$original.family,
 failed_sign:$sg,governor_word:$word,governor:{word:original_word($text;$r.governor),bytes:$r.governor.bytes,bpe:$r.governor.tokens},
 old_winner:view($text;$c;$p.winner_site),matched_control:view($text;$c;$p.control_site),omitted:view($text;$c;$omit),
 gold:($match[0]|cost_view($text;$c)),eligible:$n,
 gold_rank:(1+([$r.candidates[]|select(.cost[1]<$gc)]|length)),
 gold_ties:([$r.candidates[]|select(.cost[1]==$gc)]|length),
 signed_winners:[$r.candidates[]|select(.cost[1]==$minimum)|cost_view($text;$c)],
 strongest_remaining_wrong:[$r.candidates[]|select(.site!=$gold and .cost[1]==$wrong)|cost_view($text;$c)],
 joint_selected:(if $read.tuple.sites==null then null else [$read.tuple.sites[]|view($text;$c;.)] end),
 joint_governors:[$read.links[]|{word,governor}], joint_links:[$read.links[]|{word,site}],
 joint_ties:$read.tuple.ties,heading_correct:($read.tuple.addresses!=null and $read.tuple.addresses[2].tokens==($original.native_scope[2].tokens|interval)
   and $read.tuple.addresses[2].bytes==[$original.native_scope[2].bytes.start-1,$original.native_scope[2].bytes.end-1]),
 governors_correct:$jrow.arm.both_governors,exact_tuple:$jrow.arm.exact,full:$jrow.arm.full,first_source_error:$jrow.arm.first_source_error}
