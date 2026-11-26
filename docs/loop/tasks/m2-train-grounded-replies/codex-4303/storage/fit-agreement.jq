def empty_cost: . // [null,null,null];
def targets($row):
  $row.sites[$row.requested].fixed
  | {cost:(.best | empty_cost), eligible, ties};
def thirds($row):
  [
    $row.sites[]
    | select(.i != $row.requested and .i != $row.rival and .fixed.eligible > 0)
  ] as $eligible
  | ($eligible | map(.fixed.best) | min) as $best
  | ($eligible | map(select(.fixed.best == $best))) as $minima
  | {cost:($best | empty_cost), eligible:($eligible | map(.fixed.eligible) | add // 0),
     ties:($minima | map(.fixed.ties) | add // 0), min_sites:($minima | map(.i)),
     tuple:($minima | map(.fixed.first_tuple) | min) };
($terms | map(select(.candidate == $id))) as $fit
| if ($fit|length) > 0 and ($fit|length) == ($score|length)
     and all(range($fit|length);
        $fit[.] as $f | $score[.] as $s
        | ($f|[.row,.sign,.word,.requested,.rival]) == ($s|[.row,.sign,.word,.requested,.rival])
          and ($f.joint[0] | {cost,eligible,ties}) == targets($s)
          and ($f.joint[1] | {cost,eligible,ties,min_sites,tuple}) == thirds($s))
  then {candidate:$id, rows:($fit|length),
        strict_rows:($fit | map(select(.strict_win)) | length),
        strict_pairs:($fit | group_by(.pair) | map(select(length==2 and all(.[];.strict_win))) | length),
        opportunities:($fit|group_by(.opportunity)|map({key:.[0].opportunity,value:length})|from_entries)}
  else error("fit's strongest complete third differs from independently scored site groups")
  end
