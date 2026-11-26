def check($ok;$msg): if $ok then . else error($msg) end;
def n(f): map(select(f))|length;
def pair_win($a): all(.[];.arms[$a].exact);
def movement($a;$b):
  group_by(.pair_id) | {gained:n(pair_win($a) and (pair_win($b)|not)),lost:n(pair_win($b) and (pair_win($a)|not)),
    kept:n(pair_win($a) and pair_win($b)),neither:n((pair_win($a) or pair_win($b))|not)};
def stats($a):
  group_by(.pair_id) as $pairs |
  {rows:length,pairs:($pairs|length),
   exact_rows:n(.arms[$a].exact),exact_pairs:($pairs|n(pair_win($a))),
   full_rows:n(.arms[$a].full),full_pairs:($pairs|n(all(.[];.arms[$a].full))),
   earlier_pairs:($pairs|n(all(.[];.arms[$a].earlier))),governor_rows:n(.arms[$a].both_governors),
   neutral_rows:n(.arms[$a].neutral),exchanged_rows:n(.arms[$a].swapped),
   ties:n(.arms[$a].ties>1),no_eligible:n(.arms[$a].eligible==0),
   errors:(group_by(.arms[$a].first_source_error)|map({key:.[0].arms[$a].first_source_error,value:length})|from_entries),
   vs_inherited:movement($a;"baseline"),vs_positive:movement($a;"positive"),
   vs_native:movement($a;"native"),vs_unfitted:movement($a;"unfitted"),
   vs_distance:movement($a;"governor_gap"),vs_correct:movement($a;"correct_link"),
   vs_exchanged:movement($a;"exchanged_link")};
def table: . as $rows | [.[0].arms|keys[] as $a|{key:$a,value:($rows|stats($a))}]|from_entries;
def cells(f): group_by(f)|map({key:(.[0]|f),value:table})|from_entries;
check(length==256 and (map(.case_id)|unique|length)==256 and (group_by(.pair_id)|length)==128 and
  all(group_by(.pair_id)[];length==2 and (map(.admission)|sort)==["defer","take"] and
      (map(.family)|unique|length)==1 and (map(.distance)|unique|length)==1); "unpaired enrollment") |
check((if $split=="train" then (map(.distance)|unique)==["far"] and (group_by(.family)|length)==8
       else (map(.distance)|unique)==["far","near"] and (group_by(.family)|length)==4 end) and
  all(group_by([.distance,.family])[];length==32); "distance/family design differs") |
{split:$split,pairs:128,endpoints:256,arms:table,by_distance:cells(.distance),
 by_family:cells(.family),by_family_and_distance:cells(.family+"/"+.distance)}
