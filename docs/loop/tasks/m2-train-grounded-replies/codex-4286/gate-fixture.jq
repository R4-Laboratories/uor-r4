def loc($i): {start:$i,end:($i+1)};
def swap: [.[1],.[0],.[2]];
def pick($s): {selection:$s,tokens:$s,ties:1,eligible:3};
[range(256) as $n | ($n/2|floor) as $i |
 ([loc(10),loc(20),loc(50)] | if $n%2==0 then . else swap end) as $g |
 {id:($n|tostring),pair:($i|tostring),family:(($i/32|floor)|tostring),
  distance:(if $i%2==0 then "near" else "far" end),admission:(if $n%2==0 then "take" else "defer" end),
  g:$g,x:($g|swap)}] |
if $part=="reference" then
  map({case_id:.id,pair_id:.pair,family,distance,admission,
       gold_binding:{affirmed:.g[0],excluded:.g[1],referenced:.g[2]},
       native_scope:[.g[]|{tokens:.}],neutral_rival:{tokens:loc(33)}})
else
  .[] | . as $c |
  (if $part=="fail" then .g
   elif $part=="damaged" and .id=="3" then [loc(11),.g[1],.g[2]]
   else .x end) as $x |
  {case_id:.id,pair_id:.pair,family,distance,admission,gold:.g,exchanged:.x,
   arms:{native:pick($x),governor_only:pick($x),nouns_only:pick(.x),
     correct_link:pick(.g),exchanged_link:pick(.x)}} |
  .conditional_gold_heading=.arms |
  if $part=="damaged" and $c.id=="3" then .arms.correct_link.tokens[0]=loc(777) else . end
end
