# Pass the sealed independently authored panel and the device's physical JSONL picks.
def arms: ["native","governor_only","nouns_only","correct_link","exchanged_link"];
def insist($ok; $why): if $ok then . else error($why) end;
def swapped: [.[1], .[0], .[2]];
def address: type == "object" and (.start|type) == "number" and .end > .start;
def valid_triple: type == "array" and length == 3 and all(.[]; address);
def same_earlier($a; $b): $a[0:2] == $b[0:2];
def count_where(f): map(select(f)) | length;

def summarize_arm($arm):
  group_by(.pair) as $pairs |
  { endpoints:length, pairs:($pairs|length),
    exact_endpoints:count_where(.v[$arm].exact),
    exact_pairs:($pairs|count_where(all(.[]; .v[$arm].exact))),
    gains_over_native:($pairs|count_where(all(.[]; .v[$arm].exact) and any(.[]; .v.native.exact|not))),
    losses_to_native:($pairs|count_where(any(.[]; .v[$arm].exact|not) and all(.[]; .v.native.exact))),
    both_earlier_exact:count_where(.v[$arm].earlier),
    exact_earlier_pairs:($pairs|count_where(all(.[]; .v[$arm].earlier))),
    wrong_heading:count_where(.v[$arm].wrong_heading),
    neutral_noun_selected:count_where(.v[$arm].neutral),
    abstentions:count_where(.v[$arm].abstained),
    best_ties:count_where(.v[$arm].tied),
    rows_with_only_one_eligible_tuple:count_where(.v[$arm].eligible==1),
    eligible_tuple_range:(map(.v[$arm].eligible)|{min:min,max:max}),
    exact_with_supplied_heading:count_where(.v[$arm].given_heading),
    paired_with_supplied_heading:($pairs|count_where(all(.[]; .v[$arm].given_heading))) };
def scoreboard: . as $rows | [arms[] as $arm | {key:$arm,value:($rows|summarize_arm($arm))}] | from_entries;

($reference[0] | INDEX(.case_id // .id)) as $cases |
$rows | insist(length == 256 and (map(.case_id)|unique|length) == 256 and ($cases|length)==256;
     "panel size or ID differs") |
map(. as $r | $cases[$r.case_id] as $c |
    ($c.gold_binding | [(.affirmed // .affirmative),.excluded,(.referenced // .reference)]) as $bytes |
    ($c.native_scope | map(.tokens)) as $tokens |
    $c.neutral_rival.tokens as $neutral |
    insist($c != null and $r.pair_id==($c.pair_id // $c.admission_pair) and $r.admission==$c.admission
        and $r.family==$c.family and $r.distance==$c.distance; "mismatched panel row: "+$r.case_id) |
    insist(($bytes|valid_triple) and ($tokens|valid_triple) and ($neutral|address)
        and all($tokens[]; .!=$neutral) and $r.gold==$bytes and $r.exchanged==($bytes|swapped);
        "source reference differs: "+$r.case_id) |
    {id:$r.case_id,pair:$r.pair_id,family:$r.family,distance:$r.distance,admission:$r.admission,
     v:([arms[] as $a | $r.arms[$a] as $p | $r.conditional_gold_heading[$a] as $q |
         insist($p!=null and $q!=null; "missing arm or diagnostic on "+$r.case_id) |
         insist((($p.selection==null) or (($p.selection|valid_triple) and ($p.tokens|valid_triple) and $p.ties==1))
          and (($p.selection == $bytes) == ($p.tokens == $tokens)); "byte/token disagreement on "+$r.case_id) |
         insist($p.selection==null or
           (if $a=="correct_link" then same_earlier($p.selection;$bytes)
            elif $a=="exchanged_link" then same_earlier($p.selection;($bytes|swapped))
            elif $a=="nouns_only" then ($p.selection[0:2]|sort)==($bytes[0:2]|sort)
            else true end); "intervention returned an unavailable earlier endpoint on "+$r.case_id) |
         {key:$a,value:{exact:($p.selection==$bytes and $p.tokens==$tokens),
           earlier:($p.selection!=null and same_earlier($p.selection;$bytes) and same_earlier($p.tokens;$tokens)),
           wrong_heading:($p.selection!=null and $p.tokens[2]!=$tokens[2]),
           neutral:($p.selection!=null and any($p.tokens[0:2][]; .==$neutral)),
           abstained:($p.selection==null), tied:($p.ties>1),eligible:$p.eligible,
           given_heading:($q.selection==$bytes and $q.tokens==$tokens)}}] | from_entries) }) |
insist((group_by(.pair)|length)==128 and
  all(group_by(.pair)[]; length==2 and (map(.admission)|sort)==["defer","take"]
      and (map(.family)|unique|length)==1 and (map(.distance)|unique|length)==1)
  and (group_by(.family)|length)==4 and
  all(group_by([.family,.distance])[]; length==32) and (map(.distance)|unique)==["far","near"];
  "unbalanced opposed pairs or distance/family crossing") |
scoreboard as $all |
(group_by(.distance)|map({key:.[0].distance,value:scoreboard})|from_entries) as $d |
(group_by(.family)|map({key:.[0].family,value:scoreboard})|from_entries) as $f |
{pairs:128,endpoints:256,reference_sha256:$reference_sha256,endpoints_sha256:$endpoints_sha256,
 arms:$all,by_distance:$d,by_family:$f,
 gate:{net_over_native:($all.correct_link.exact_pairs-$all.native.exact_pairs),
       net_over_exchanged:($all.correct_link.exact_pairs-$all.exchanged_link.exact_pairs),
       gains_in_both_distances:all($d[]; .correct_link.exact_pairs > .native.exact_pairs),
       no_family_loses:all($f[]; .correct_link.exact_pairs >= .native.exact_pairs)}} |
.pass=(.gate.net_over_native>=8 and .gate.net_over_exchanged>=8 and .gate.gains_in_both_distances and .gate.no_family_loses)
