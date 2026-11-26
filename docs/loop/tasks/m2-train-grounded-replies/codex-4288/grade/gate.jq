def insist($ok;$msg): if $ok then . else error($msg) end;
def n(f): map(select(f))|length;
def exactpair($arm): all(.[]; .arms[$arm].exact);
def move($arm;$base):
  group_by(.pair_id) | {gained:n(exactpair($arm) and (exactpair($base)|not)),
    lost:n(exactpair($base) and (exactpair($arm)|not)),
    kept:n(exactpair($arm) and exactpair($base)),neither:n((exactpair($arm) or exactpair($base))|not)};
def per_arm($a):
  . as $rows | group_by(.pair_id) as $p |
  {exact_pairs:($p|n(exactpair($a))), exact_rows:n(.arms[$a].exact),
   full_link_pairs:($p|n(all(.[];.arms[$a].full))),full_link_rows:n(.arms[$a].full),
   governor_pairs:($p|n(all(.[];.arms[$a].both_governors))),governor_rows:n(.arms[$a].both_governors),
   earlier_pairs:($p|n(all(.[];.arms[$a].earlier))),earlier_rows:n(.arms[$a].earlier),
   exchanged_rows:n(.arms[$a].swapped),neutral_rows:n(.arms[$a].neutral),
   ties:n(.arms[$a].ties>1),no_eligible:n(.arms[$a].eligible==0),
   first_source_error:(group_by(.arms[$a].first_source_error)|map({key:.[0].arms[$a].first_source_error,value:length})|from_entries),
   gained_lost_vs_native:move($a;"native"),gained_lost_vs_unfitted:move($a;"unfitted"),
   gained_lost_vs_governor_gap:move($a;"governor_gap")};
def scores: [. as $rows| .[0].arms|keys[] as $a|{key:$a,value:($rows|per_arm($a))}]|from_entries;
def cells(f): group_by(f)|map({key:(.[0]|f),value:scores})|from_entries;
insist(length==256 and (map(.case_id)|unique|length)==256 and
 (group_by(.pair_id)|length)==128 and all(group_by(.pair_id)[];
    length==2 and (map(.admission)|sort)==["defer","take"] and (map(.distance)|unique|length)==1 and (map(.family)|unique|length)==1) and
 (map(.distance)|unique)==["far","near"] and (group_by(.family)|length)==4 and
 all(group_by([.distance,.family])[];length==32);
 "wrong design or unpaired held rows") |
scores as $s | cells(.distance) as $d | cells(.family) as $f |
{pairs:128,endpoints:256,reference_sha256:$ref_sha,endpoints_sha256:$end_sha,
 arms:$s,by_distance:$d,by_family:$f,by_family_and_distance:cells(.family+"/"+.distance),
 gate:{
   native_gain:($s.learned.exact_pairs-$s.native.exact_pairs>=8),
   unfitted_gain:($s.learned.exact_pairs-$s.unfitted.exact_pairs>=8),
   both_distances:all($d[];.learned.exact_pairs>.native.exact_pairs and .learned.exact_pairs>.unfitted.exact_pairs),
   no_family_loses:all($f[];.learned.exact_pairs>=.native.exact_pairs),
   both_distances_full_links:all($d[];.learned.full_link_pairs>=1),
   control_discrimination:($s.correct_link.earlier_pairs==128 and $s.exchanged_link.exact_rows==0 and
       $s.correct_link.exact_pairs-$s.exchanged_link.exact_pairs>=8 and all($d[];.correct_link.exact_pairs>.exchanged_link.exact_pairs))
 }} | .pass=all(.gate[];.)
