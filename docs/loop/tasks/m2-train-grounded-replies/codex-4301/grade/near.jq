[$gold[]|. as $g|$native[.row_index] as $n|
 if $n.row!=$g.row_index or ($g.governors|length)!=2 or ($g.sites|length)!=3 then error("near row identity")
 else {id:$g.pair_id,win:($n.tuple.ties==1 and ($n.links|map(.word))==$g.governors and $n.tuple.sites==$g.sites)} end] |
 group_by(.id)|map(if length!=2 then error("near pair multiplicity") else {id:.[0].id,win:all(.[];.win)} end) |
 if length!=24 then error("protected near census") else {pairs:length,correct:([.[]|select(.win)]|length),
 wins:[.[]|select(.win)|.id],losses:[.[]|select(.win|not)|.id]} end
