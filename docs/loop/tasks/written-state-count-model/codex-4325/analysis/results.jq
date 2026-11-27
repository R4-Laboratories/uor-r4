def n(p): map(select(p))|length;
def closed: all(.[];.learned.exact);
def matched: {n:length,exact:n(closed),ceiling:n(all(.[];.ceiling.exact))};
def row_stats:
 {rows:length,exact:n(.learned.exact),early:n(.learned.wrong==3),late:n(.learned.wrong==4),
  other:n(.learned.wrong!=null and .learned.wrong!=3 and .learned.wrong!=4),
  cap:n(.learned.terminal|not),b_required:n(.learned.b_required),b_reached:n(.learned.b_reached),b_right:n(.learned.b_right),
  b_right_after_supplied_first:n(.after_supplied_first.b_right),b_reached_after_supplied_first:n(.after_supplied_first.b_reached),
  exact_short:n(.side==0 and .learned.exact),exact_long:n(.side==1 and .learned.exact),
  id_pairs:(group_by(.pair_id)|matched),ceiling:n(.ceiling.exact)};
def crossed:
  group_by(.group) as $groups |
  if all($groups[]; length==4 and (map([.selector,.one])|sort)==[[1,1],[1,2],[2,1],[2,2]] and
     (map([.context,.question,.positions,.ids])|unique|length)==1)
  then . else error("invalid crossed context") end |
  {contexts:($groups|matched),equal_length_contexts:($groups|map(select((map(.length)|unique|length)==1))|matched),
   clauses:(group_by([.group,.selector])|matched),selectors:(group_by([.group,.one])|matched)};
. as $all | [.[]|select(.one!=null)] as $scoped | [.[]|select(.one==null)] as $direct |
{all:row_stats,scoped:($scoped|row_stats),direct:($direct|row_stats),
 direct_by_length:($direct|group_by(.contrast)|map({contrast:.[0].contrast,stats:row_stats})),
 family:($all|group_by(.family)|map({family:.[0].family,stats:row_stats}))} +
(if $split=="held" or $split=="quartet" or $split=="train" then
 ($scoped|map(select($split!="train" or .cohort=="fresh"))) as $full |
 {crossed:($full|crossed),crossed_families:($full|group_by(.family)|map({family:.[0].family,stats:crossed}))}
 else {} end)
