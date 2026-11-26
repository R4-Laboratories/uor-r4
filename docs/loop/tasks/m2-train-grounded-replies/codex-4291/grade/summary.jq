def need($ok;$why):if $ok then . else error($why) end;
def keys_for(f): map(select(f)|.pair)|sort;
def describe($p):
 {exact:($p|keys_for(.exact)),full:($p|keys_for(.full)),earlier:($p|keys_for(.earlier)),
  prompts:($p|map(.exact_rows)|add),full_prompts:($p|map(.full_rows)|add),
  governor_prompts:($p|map(.both_governors_rows)|add),neutral_prompts:($p|map(.neutral_rows)|add)};
def cells($all;$ids): $all|with_entries(.value=describe([.value[]|select(.pair as $k|$ids|has($k))]));
($arms[0]) as $all |
($all.baseline|map({pair,family,distance,ids})) as $design |
need(($design|length)==128 and ($design|map(.pair)|unique|length)==128 and all($all[];map({pair,family,distance,ids})==$design);
     "arm identities differ") |
{all:cells($all;($design|INDEX(.pair))),
 distance:($design|group_by(.distance)|map({key:.[0].distance,value:cells($all;INDEX(.pair))})|from_entries),
 family:($design|group_by(.family)|map({key:.[0].family,value:cells($all;INDEX(.pair))})|from_entries),
 family_distance:($design|group_by([.family,.distance])|map({key:(.[0].family+"/"+.[0].distance),value:cells($all;INDEX(.pair))})|from_entries)}
