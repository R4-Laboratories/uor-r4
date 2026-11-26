def rank($priority):
  . as $s |
  (.local_pair_wins | add) as $local |
  (.joint_pair_wins | add) as $joint |
  (.both_pair_wins | add) as $both |
  (.both_pair_wins | min) as $floor |
  (.local_pair_worst | add) as $local_debt |
  (.joint_pair_worst | add) as $joint_debt |
  if $priority == "balanced" then
    [.invalid, -$floor, -$both, $local_debt + $joint_debt,
     -$joint, -$local, (.local_total | add) + (.joint_total | add)]
  elif $priority == "local" then
    [.invalid, -$local, $local_debt, (.local_total | add), -$floor, -$joint, $joint_debt]
  elif $priority == "joint" then
    [.invalid, -$joint, $joint_debt, (.joint_total | add), -$floor, -$local, $local_debt]
  else
    (if $priority == "affirmative" then 0 elif $priority == "excluded" then 1 else error("invalid priority") end) as $i |
    [.invalid, -.both_pair_wins[$i], .local_pair_worst[$i] + .joint_pair_worst[$i],
     -.local_pair_wins[$i], -.joint_pair_wins[$i], -$both, $local_debt + $joint_debt]
  end;

map(select(.phase == "initial")) as $initial
| map(select(.phase == "fit")) as $fit
| select(($initial | length) == 57)
| select(all($fit[];
    .priority as $priority |
    ((.after | rank($priority)) < (.before | rank($priority))) == .accepted))
| select(all(range(5; 8);
    . as $index |
    ($fit | map(select(.path == $index)))[0] as $start |
    $start.priority as $priority |
    ($initial | min_by([(.after | rank($priority)), .after.movement, .initial_index])) as $winner |
    $start.parent == $winner.name and $start.before == $winner.after))
| true
