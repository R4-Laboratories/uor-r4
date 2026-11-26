# Join the cold physical reads only after inference. Every address is checked in two coordinate systems.
def insist($ok;$msg): if $ok then . else error($msg) end;
def span: {start:.[0],end:.[1]};
def address: if .==null then null else {bytes:(.bytes|span),tokens:(.tokens|span)} end;
def valid_index($i;$n): ($i|type)=="number" and $i==($i|floor) and $i>=0 and $i<$n;
def rank($arm;$device;$c;$p):
    $device as $d | $d.tuple as $t |
    insist($d.row==$c.ordinal and ($d.links|length)==2 and
        ($d.arm==(if ($arm|IN("corrupt_noun","flipped_sign")) then "learned"
          elif $arm=="correct_link" then "correct" elif $arm=="exchanged_link" then "exchanged"
          else $arm end)); "physical ordinal or arm differs") |
    ($d.links|map(. as $link | ($link.governor|address) as $g | ($link.noun|address) as $n |
        insist((if $link.word==null then $g==null else valid_index($link.word;($c.words|length)) and $c.words[$link.word]==$g end)
          and (if $link.site==null then $n==null else valid_index($link.site;($c.sites|length)) and $c.sites[$link.site]==$n end);
          "winner word or noun address does not resolve") | {g:$g,n:$n,word_ties,site_ties})) as $links |
    ($t.addresses | if .==null then null else map(address) end) as $choice |
    insist((if $t.ordinal==null then $choice==null and $t.sites==null and $t.ties!=1
       else valid_index($t.ordinal;($c.tuples|length)) and $t.ties==1 and
         $c.tuples[$t.ordinal]==$t.sites and ($t.sites|map($c.sites[.]))==$choice end);
       "picked tuple differs from original proposal or tie") |
    ([$p.native_scope[] | {bytes:{start:(.bytes.start-1),end:(.bytes.end-1)},tokens}]) as $gold |
    ([$p.governors.affirmative,$p.governors.excluded] | map({bytes,tokens:(.tokens|span)})) as $governors |
    ($links|map(.g)) as $wg | ($links|map(.n)) as $wn |
    insist($arm=="native" or $choice==null or
       (all($wg[]; .!=null) and $wg[0]!=$wg[1] and $wn==$choice[:2]);
       "non-native winning links disagree with the chosen original noun addresses") |
    ($wg==$governors) as $bothg | ($wn==$gold[:2]) as $bothn |
    ($choice==$gold) as $scope |
    (if $choice==null then null else ($choice|map(.bytes)) end) as $bytes |
    (if $choice==null then null else ($choice|map(.tokens)) end) as $tokens |
    {selection:$bytes,tokens:$tokens,bank_cost:$t.bank,governors:$wg,nouns:$wn,
     ties:$t.ties,eligible:$t.eligible,best:$t.best,rival:$t.rival,
     both_governors:$bothg,both_linked_nouns:$bothn,both_links:($bothg and $bothn),
     signs:[range(2) as $s | {governor:($wg[$s]==$governors[$s]),noun:($wn[$s]==$gold[$s]),
          governor_ties:$links[$s].word_ties,noun_ties:$links[$s].site_ties}],
     exact:$scope,full:($scope and $bothg and $bothn),
     earlier:($choice!=null and $choice[:2]==$gold[:2]),
     swapped:($choice==[$gold[1],$gold[0],$gold[2]]),
     neutral:($choice!=null and any($choice[:2][]; .tokens==$p.neutral_rival.tokens)),
     first_source_error:(if $scope and $bothg and $bothn then "none"
        elif $scope and $arm=="native" then "none"
        elif $arm!="native" and ($bothg|not) then "signed_governor"
        elif $t.eligible==0 then "no_eligible_tuple"
        elif $choice==null then "joint_tie"
        elif $choice[0]!=$gold[0] then "affirmed_noun"
        elif $choice[1]!=$gold[1] then "excluded_noun"
        elif $choice[2]!=$gold[2] then "heading"
        else "link_witness" end)};

($reference[0]) as $refs |
{native:$native,unfitted:$unfitted,learned:$learned,governor_gap:$gap,correct_link:$correct,
 exchanged_link:$exchange,corrupt_noun:$corrupt,flipped_sign:$flipped} as $ranks |
insist(($refs|length)==256 and ($coords|length)==256 and all($ranks[];length==256) and
  ($refs|map(.id)|unique|length)==256;"incomplete cohort or arm") |
range(256) as $i | $refs[$i] as $r | $coords[$i] as $c |
insist($i==$c.ordinal and $r.id==$c.id and $r.structural_prompt_sha256==$c.prompt_sha256;
       "graded text differs from prepared current turn") |
([ $ranks|keys[] as $a | {key:$a,value:rank($a;$ranks[$a][$i];$c;$r)} ]|from_entries) as $arms |
{case_id:$r.id,pair_id:$r.pair_id,admission:$r.admission,family:$r.family,distance:$r.distance,
 prompt_sha256:$c.prompt_sha256,gold:[$r.gold_binding.affirmed,$r.gold_binding.excluded,$r.gold_binding.referenced],
 gold_tokens:($r.native_scope |map(.tokens)), arms:$arms}
