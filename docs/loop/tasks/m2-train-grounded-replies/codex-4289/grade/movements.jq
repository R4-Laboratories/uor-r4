# Describe movements after the registered gate, from the original occurrence records.
def terms($text; $spans):
  [$spans[] | . as $span | $text[$span.start:$span.end]];
def words($text; $arm): {terms:terms($text;$arm.selection),original_bpe:$arm.tokens};
def all_won($arm): all(.[];.arms[$arm].exact);
def mark:
  if all_won("baseline") then (if all_won("learned") then "kept" else "lost" end)
  else (if all_won("learned") then "gained" else "neither" end) end;
def bucket: group_by(.)|map({key:.[0],value:length})|from_entries;
def change_rows($index):
  [.[] | select(.change=="gained" or .change=="lost") | . as $p | .rows[] | . as $r |
    (if $p.change=="lost" then .arms.learned else .arms.baseline end) as $f |
    $index[.case_id].turns[1] as $text |
    ([range(3) as $n | select($f.selection[$n]!=$r.gold[$n] or $f.tokens[$n]!=$r.gold_tokens[$n])|$n]) as $wrong |
    {family,distance,change:$p.change,admission,first:$f.first_source_error,wrong_slots:$wrong,
     goal:(if ($wrong|length)>0 then terms($text;[$r.gold[$wrong[0]]])[0] else null end),
     rival:(if ($wrong|length)>0 then terms($text;[$f.selection[$wrong[0]]])[0] else null end),
     governor_addresses_held:(.arms.learned.governors==.arms.baseline.governors)}] |
   group_by(.)|map(.[0]+{prompts:length});
($reference[0]|INDEX(.id)) as $index |
($held | group_by(.pair_id) |
  map({pair:.[0].pair_id,family:.[0].family,distance:.[0].distance,change:mark,rows:.})) as $pairs |
($edges | map(select(.eligible and .before=="loss"))) as $losses |
if ($pairs|length)!=128 or ($held|length)!=256 or ($edges|length)!=512 then error("incomplete evidence") else
{losses:{strict:($losses|length),original_rival_terms:($losses|map(.old_rival.word)|bucket),
   original_rival_kinds:($losses|map(.old_kind)|bucket),changed_strongest:($losses|map(select(.current_pivot_equals_old|not))|length),
   old_margin_range:($losses|map(.old_margin)|[min,max]),new_margin_range:($losses|map(.new_margin)|[min,max])},
 cells:($pairs|group_by(.family,.distance)|map({family:.[0].family,distance:.[0].distance,movement:(map(.change)|bucket)})),
 changed_source_counts:($pairs|change_rows($index)),
 lost_near:[$pairs[]|select(.distance=="near" and .change=="lost")|.pair],
 examples:($pairs|map(select(.change=="gained" or .change=="lost"))|group_by(.family,.distance,.change)|
  map(sort_by(.pair)|first | {pair,family,distance,change,rows:(.rows|map(. as $row|$index[.case_id].turns[1] as $t |
     {id:.case_id,admission,goal:terms($t;.gold),goal_bpe:.gold_tokens,before:words($t;.arms.baseline),after:words($t;.arms.learned)}))}))}
end
