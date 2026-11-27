def ids($a;$field): .[$a][$field];
def change($a;$b;$field):
  ids($a;$field) as $lhs | ids($b;$field) as $rhs |
  {from:$a,to:$b,gained:($rhs-$lhs),lost:($lhs-$rhs),shared:($rhs-($rhs-$lhs))};
INDEX(.[];.arm) |
{signed:[change("baseline";"G";"wins"),change("baseline";"N";"wins"),change("baseline";"GN";"wins"),
         change("G";"GN";"wins"),change("N";"GN";"wins"),change("GN";"GN-wrong-G";"wins"),
         change("GN";"GN-wrong-N";"wins"),change("GN";"GN-wrong-both";"wins")],
 full:[change("baseline";"G";"full_wins"),change("baseline";"N";"full_wins"),change("baseline";"GN";"full_wins")],
 strict:[change("baseline";"G";"strict_wins"),change("baseline";"N";"strict_wins"),change("baseline";"GN";"strict_wins")]}
