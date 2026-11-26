def ensure($ok; $message): if $ok then . else error($message) end;
def address: {bytes:[.bytes.start,.bytes.end],tokens:[.tokens.start,.tokens.end]};
def pairs($rows):
  $rows | group_by(.id)
  | ensure(all(.[]; length == 2 and .[0].row != .[1].row); "guard physical pair multiplicity")
  | map({id:.[0].id, rows:map(.row), win:all(.[]; .win)});
def regression:
  [
    $reg[] as $g | $r_native[$g.row_index] as $n
    | (if $g.sign == 1 then 0 else 1 end) as $side
    | ensure($n.row == $g.row_index; "regression physical row mismatch")
    | {id:$g.pair_id, row:$g.row_index, win:(
        $n.tuple.ties == 1 and $n.links[$side].word == $g.governor.index
        and $n.tuple.sites[$side] == $g.requested.index
        and $n.links[$side].governor == ($g.governor | address)
        and $n.links[$side].noun == ($g.requested | address)
        and $n.tuple.addresses[$side] == ($g.requested | address))}
  ];
def near:
  [
    $gold[] as $g | $n_native[$g.row_index] as $n | $coordinates[$g.row_index] as $c
    | ensure($g.row_index == $n.row and $g.row_index == $c.ordinal
          and ($g.governors|length)==2 and ($g.sites|length)==3; "near physical row mismatch")
    | {id:$g.pair_id, row:$g.row_index, win:(
        $n.tuple.ties == 1 and ($n.links | map(.word)) == $g.governors
        and $n.tuple.sites == $g.sites
        and ($n.links | map(.governor)) == ($g.governors | map($c.words[.] | address))
        and ($n.links | map(.noun)) == ($g.sites[:2] | map($c.sites[.] | address))
        and $n.tuple.addresses == ($g.sites | map($c.sites[.] | address)))}
  ];
pairs(regression) as $r | pairs(near) as $n
| ensure(($r|length)==4 and ($n|length)==24; "guard enrollment count")
| ensure(($r|map(select(.win)|.id)) == $grade.regression_ids and
         ($n|map(select(.win)|.id)) == $grade.near_ids; "original-address guard wins disagree")
| {regression:$r, near:$n, losses:([$r[],$n[]|select(.win|not)]|length)}
