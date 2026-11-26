# Read the already measured neutral ranks at the original governor.
def require($ok;$why): if $ok then . else error($why) end;
def interval: [.start,.end];
def address: {bytes:(.bytes|interval),tokens:(.tokens|interval)};
def packed: {bytes,tokens};
def surface($text;$span):
  $span as [$lo,$hi] |
  def at($byte):
    if ($text|length)==($text|utf8bytelength) then $byte else
      first(range(($text|length)+1) as $k | select(($text[:$k]|utf8bytelength)==$byte) | $k)
    end;
  [at($lo),at($hi)] |
  require(length==2 and $lo>=0 and $hi>$lo and $hi<=($text|utf8bytelength);"invalid original byte boundaries") |
  $text[.[0]:.[1]];
def span_between($left;$mid;$right):
  ($left[1]<=$mid[0] and $mid[1]<=$right[0]) or ($right[1]<=$mid[0] and $mid[1]<=$left[0]);
def gap($a;$b): [0,$a[0]-$b[1],$b[0]-$a[1]] | max;
def distances($gov;$noun): {bytes:gap($gov.bytes;$noun.bytes),bpe:gap($gov.tokens;$noun.tokens)};
def rank_record($text;$gov):
  {site,word:surface($text;.noun.bytes),bytes:.noun.bytes,bpe:.noun.tokens,
   cost:.cost[1],center:.transport_center,governor_gap:distances($gov;.noun)};
def relation($delta): if $delta<0 then "closer" elif $delta==0 then "same distance" else "farther" end;
def relative($gov;$gold):
  . as $candidate |
  . + {cost_minus_gold:(.cost-$gold.cost),
       between_bytes:span_between($gov.bytes;.bytes;$gold.bytes),
       between_bpe:span_between($gov.tokens;.bpe;$gold.bpe),
       compared_with_gold_gap:{
         bytes:relation($candidate.governor_gap.bytes-$gold.governor_gap.bytes),
         bpe:relation($candidate.governor_gap.bpe-$gold.governor_gap.bpe)},
       same_center_as_gold:(.center==$gold.center)};
def histogram(f):
  group_by(f)|map({key:(.[0]|f|tostring),value:length})|from_entries;
def ranges(f): map(f)|{min:min,max:max};
def facts:
  . as $rows | [$rows[].outrankers[]] as $all |
  {prompts:length,pairs:(map(.pair_id)|unique|length),strict_outrankers:($all|length),
   gold_rank:histogram(.gold_rank),gold_cost:ranges(.gold.cost),gold_centers:histogram(.gold.center),
   outranker_cost:($all|ranges(.cost)),cost_minus_gold:($all|ranges(.cost_minus_gold)),
   outranker_centers:($all|histogram(.center)),
   outranker_center_to_gold:($all|histogram(.same_center_as_gold)),
   between_original_spans:($all|histogram([.between_bytes,.between_bpe])),
   governor_gap_relative_to_gold:($all|histogram([.compared_with_gold_gap.bytes,.compared_with_gold_gap.bpe])),
   outranker_words:($all|group_by(.word)|map({word:.[0].word,occurrences:length,cost:ranges(.cost),
                       cost_minus_gold:ranges(.cost_minus_gold),centers:histogram(.center),
                       byte_gap:ranges(.governor_gap.bytes),bpe_gap:ranges(.governor_gap.bpe)})),
   control_counts:histogram(if .control==null then "missing"
       elif .control.cost_minus_gold<0 then "beats gold"
       elif .control.cost_minus_gold==0 then "ties gold"
       else "loses to gold" end),
   control_cost_minus_gold:([.[]|.control|select(.!=null)]|ranges(.cost_minus_gold))};

require(($panel|length)==48 and ($rank|length)==48 and ($grade|length)==48 and
    ($reference[0]|length)==256 and ($coordinates|length)==256 and
    ($panel|map(.row))==($rank|map(.row)) and ($rank|map(.row))==($grade|map(.row)) and
    ($panel|map(.row)|unique|length)==48;
    "incomplete or shifted neutral trace") |
[range(48) as $j |
  $panel[$j] as $p | $rank[$j] as $r | $grade[$j] as $g |
  $p.row as $i | $p.links[$p.failed_sign] as $link |
  $reference[0][$i] as $ref | $coordinates[$i] as $coord | $ref.turns[1] as $text |
  require($ref.id==$p.case_id and $coord.id==$p.case_id and $g.case_id==$p.case_id and
     $coord.ordinal==$i and $ref.pair_id==$p.pair_id and $ref.family==$p.family and
     $ref.structural_prompt_sha256==$p.prompt_sha256 and $p.prompt_sha256==$coord.prompt_sha256 and
     $ref.admission==$p.admission and $r.sign==$p.failed_sign and $r.sign==$link.sign and
     $r.word==$link.governor_word and $r.governor==($link.governor|packed) and
     $r.governor==($coord.words[$r.word]|address) and $r.omitted_site==null and $r.requested_site==null and
     ($r.candidates|map(.site))==$p.eligible_sites and ($p.eligible_sites|unique)==$p.eligible_sites and
     all($r.candidates[]; .site>=0 and .site<($coord.sites|length) and
       .noun==($coord.sites[.site]|address) and (.cost|length)==2 and (.transport_center|type)=="number") and
     ($r.candidates|map(.cost[0])|unique|length)==1;
     "neutral rank or original identity disagrees") |
  ($r.candidates|map(select(.site==$link.gold_site))) as $matched_gold |
  require(($matched_gold|length)==1 and $matched_gold[0].noun==($link.gold|packed);"gold address missing") |
  ($matched_gold[0]|rank_record($text;$r.governor)) as $gold |
  ([$r.candidates[] | select(.cost[1]<$gold.cost) |
      rank_record($text;$r.governor) | relative($r.governor;$gold) | .+{native_joint_winner:(.site==$p.winner_site)}] |
      sort_by([.cost,.site])) as $outrankers |
  ([$r.candidates[] | select(.site==$p.control_site)]) as $matched_control |
  require($gold.word==$link.gold.surface and ($gold|del(.governor_gap))==$g.gold and
      (1+($outrankers|length))==$g.gold_rank and
      ([$r.candidates[]|select(.cost[1]==$gold.cost)]|length)==$g.gold_ties and
      ($outrankers|any(.native_joint_winner)) and
      (if $p.control_site==-1 then ($matched_control|length)==0
       else ($matched_control|length)==1 and $matched_control[0].noun==($p.control|packed) end);
      "gold, winner, control or grade rank disagrees") |
  {row:$i,case_id:$p.case_id,pair_id:$p.pair_id,family:$p.family,admission:$p.admission,
   sign:$r.sign,governor:$link.governor,gold:$gold,gold_rank:$g.gold_rank,gold_ties:$g.gold_ties,
   control:($matched_control[0] | if .==null then null else
       rank_record($text;$r.governor)|relative($r.governor;$gold) end),
   outrankers:$outrankers}
] as $rows |
{method:"strictly lower recorded integer transport cost at the fixed original governor",
 census:($rows|facts),by_family:($rows|group_by(.family)|map({family:.[0].family}+facts)),
 rows:$rows}
