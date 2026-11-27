def census($key): group_by(.[$key])|map({key:(.[0][$key]|tostring),value:length})|from_entries;
def pairs($name): group_by(.pair)|map(select(all(.[];.[$name]))|.[0].pair)|sort;
def both($name): pairs($name)|length;
def countrows($name): map(select(.[$name]))|length;
def stats:
  {pairs:(group_by(.pair)|length), rows:length,
   signed:both("signed"),full:both("full"),distinct_signed:both("distinct_signed"),
   signed_rows:countrows("signed"),full_rows:countrows("full"),governor_rows:countrows("governor"),noun_rows:countrows("noun"),
   same_governor_rows:countrows("repeated_governor"),opposite_rows:countrows("opposite"),heading_rows:countrows("heading"),
   tied:map(select(.ties>1))|length,absent:map(select(.ties==0))|length,
   source_kinds:census("kind")};
if any(group_by(.pair)[];length!=2) then error("incomplete enrolled pair") else . end |
if $part!="near" and any(group_by(.pair)[];map(.selector)|sort!=["first","later"]) then error("unpaired first/later wording") else . end |
stats + {arm:$arm,split:$part,wins:pairs("signed"),full_wins:pairs("full"),strict_wins:pairs("distinct_signed"),
  ids:(map(.pair)|unique),
  by_family:(group_by(.family)|map(.[0].family as $family|stats+{family:$family})),
  by_sign:(group_by(.sign)|map(.[0].sign as $sign|if $part=="near" then {sign:$sign,rows:length,signed_rows:countrows("signed")}
    else stats+{sign:$sign} end)),
  by_order:(group_by(.order)|map(.[0].order as $order|if $part=="near" then {order:$order,rows:length}
    else stats+{order:$order} end)),
  first_later:(if $part=="near" then null else
    group_by(.pair)|map((sort_by(.selector)|map(.signed)) as $s |
      if $s==[true,true] then "both" elif $s==[true,false] then "first_only"
      elif $s==[false,true] then "later_only" else "neither" end)|
      group_by(.)|map({key:.[0],value:length})|from_entries end)}
