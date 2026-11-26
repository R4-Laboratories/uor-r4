def check($label; $pass): if $pass then true else error($label) end;
def row($rows; $i): $rows[] | select(.row == $i);
def sites: [.candidates[].site];
def competition($site):
  .candidates as $c
  | ($c[] | select(.site == $site) | .cost) as $gold
  | {rank: (1 + ([$c[] | select(.cost < $gold)] | length)),
     ties: ([$c[] | select(.cost == $gold)] | length)};

{
  original_gold_recovered: check("recovery arm";
    ($neutral[0].links | map(.site)) == [1, 3]
    and ($winner[0].links | map(.site)) == [0, 3]
    and ($control[0].links | map(.site)) == [1, 3]
    and $winner[0].tuple.sites == [0, 3, 6]
    and $winner[0].tuple.addresses[2] == {bytes:[39,44],tokens:[7,8]}
    and (row($nr; 0) | competition(0)) == {rank:2,ties:1}
    and (row($wr; 0) | competition(0)) == {rank:1,ties:1}),
  second_rival_survives: check("second rival";
    ($neutral[1].links | map(.site)) == [0, 4]
    and ($winner[1].links | map(.site)) == [0, 2]
    and ($control[1].links | map(.site)) == [0, 4]
    and (row($nr; 1) | competition(3)) == {rank:3,ties:1}
    and (row($wr; 1) | competition(3)) == {rank:2,ties:1}),
  same_surface_separate_address: check("same surface";
    ([row($nr; 0) | .candidates[] | select(.site == 1 or .site == 2)] as $two
      | ($two | map(.noun)) == [{bytes:[5,9],tokens:[1,2]},{bytes:[10,14],tokens:[2,3]}]
      and ($two | map(.noun.bytes as [$a,$b] | $prompt[$a:$b])) == ["drop","drop"])
    and (row($wr; 0) | sites) == [0, 2]
    and (row($cr; 0) | sites) == [0, 1]),
  governor_word_retained: check("governor sharing omitted address";
    ($winner[0].links[1].governor == {bytes:[5,9],tokens:[1,2]})
    and ($winner[0].links[1].governor == (row($nr; 0) | .candidates[] | select(.site == 1) | .noun))
    and all([$winner,$control][]; . as $arm |
      all($nr[]; . as $enrolled |
        $arm[$enrolled.row].independent_links[1 - $enrolled.sign]
        == $neutral[$enrolled.row].independent_links[1 - $enrolled.sign]))),
  gold_site_and_center_ties: check("gold and center ties";
    ($winner[2].tuple.ordinal == null and $winner[2].tuple.ties == 2)
    and ($winner[2].independent_links[0].site == null and $winner[2].independent_links[0].site_ties == 2)
    and (row($wr; 2) | competition(0)) == {rank:1,ties:2}
    and (row($wr; 2) | [.candidates[].transport_center]) == [0, 0]),
  one_tied_rival_removed: check("tied rivals";
    ($neutral[5].independent_links[0].site == null and $neutral[5].independent_links[0].site_ties == 2)
    and $winner[5].independent_links[0].site == 2
    and (row($nr; 5) | competition(0)) == {rank:3,ties:1}
    and (row($wr; 5) | competition(0)) == {rank:2,ties:1}),
  untouched_and_unavailable: check("untouched and unavailable";
    ($nr | map(.row)) == [0,1,2,4,5]
    and all([3,4][]; . as $i | $neutral[$i] == $winner[$i] and $neutral[$i] == $control[$i])
    and (row($wr; 4) | [.requested_site,.omitted_site]) == [6,null]
    and (row($cr; 4) | [.requested_site,.omitted_site]) == [null,null]),
  original_signed_candidates: check("candidate preservation";
    ([0,1,2,4,5] | all(.[]; . as $i
      | row($nr; $i) as $before
      | all([row($wr; $i), row($cr; $i)][]; . as $arm
        | $arm.candidates == [$before.candidates[] | select(.site != $arm.omitted_site)]))))
}
