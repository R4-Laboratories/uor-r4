def names: ["learned","first","post_b","both"];
def countwhere(test): map(select(test))|length;
def exact($a): all(.[];.arms[$a].exact);
def hist: group_by(.)|map({key:(.[0]|tostring),value:length})|from_entries;
def stats($units):
  names|map(. as $a|{key:$a,value:{pairs:($units|length),exact:($units|countwhere(exact($a))),
    preserved:($units|countwhere(exact("learned") and exact($a))),
    gained:($units|countwhere((exact("learned")|not) and exact($a))),
    lost:($units|countwhere(exact("learned") and (exact($a)|not)))}})|from_entries;
def section:
 . as $r | group_by(.pair_id) as $pairs |
 if all($pairs[];(map(.side)|sort)==[0,1]) then . else error("incomplete pair") end |
 {axes:stats($pairs),rows:(names|map(. as $a|{key:$a,value:{
       rows:($r|length),exact:($r|countwhere(.arms[$a].exact)),first_wrong:([$r[]|.arms[$a].wrong]|hist),
       shifts:([$r[]|[.arms.learned.wrong,.arms[$a].wrong]|map(tostring)|join("->")]|hist),
       b_required:($r|countwhere(.arms[$a].b_required)),
       correct_b_reached:($r|countwhere(.arms[$a].reached_b)),open_cap:($r|countwhere(.arms[$a].terminal|not))}})|from_entries),
  both_required:($pairs|countwhere(exact("both") and (exact("first")|not) and (exact("post_b")|not))),
  either_suffices:($pairs|countwhere((exact("learned")|not) and exact("first") and exact("post_b")))};
. as $r | if length==0 then error("empty") else . end |
{all:section,by_axis:(group_by(.axis)|map({key:.[0].axis,value:section})|from_entries),
 by_contrast:(group_by(.contrast)|map({key:(.[0].contrast|tostring),value:section})|from_entries),
 scoped:([.[]|select(.axis!="direct-count-control" and (.one!=null))] |if length>0 then section else null end)} +
(if $split!="fresh" then {} else
 [.[]|select(.one!=null)] as $sc | ($sc|group_by(.group)) as $q |
 if all($q[];length==4 and (map([.selector,.one])|unique|length)==4 and (map([.context,.question,.positions,.ids])|unique|length)==1) and ($q|length)>0 then . else error("unmatched cells") end |
 {quartets:stats($q), selector_edges:stats($sc|group_by([.group,.one])),clause_edges:stats($sc|group_by([.group,.selector])),
 families:($sc|group_by(.family)|map({family:.[0].family,clause_edges:stats(group_by([.group,.selector])),selector_edges:stats(group_by([.group,.one])),quartets:stats(group_by(.group))}))} end)
