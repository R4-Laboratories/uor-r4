def focus($r; $sign): $r.links[$sign] | [.word, .site, .best];
def tuple($r): $r.tuple | [.sites, .best, .eligible, .ties];
[
  ($baseline[0] | tuple(.) == [[0,10,8],[0,0,0],8,1]),
  ($g[0] | focus(.;0) == [0,4,[8,0]] and tuple(.) == [[4,6,7],[8,1,0],8,1]),
  ($n[0] | focus(.;0) == [1,2,[0,2]] and tuple(.) == [[2,6,7],[0,3,0],3,1]),
  ($gn[0] | focus(.;0) == [0,1,[8,1]] and tuple(.) == [[1,6,7],[8,2,0],3,1]),
  ($wrong_g[0] | focus(.;0) == [3,5,[3,0]] and tuple(.) == [[5,6,7],[3,1,0],8,1]),
  ($wrong_n[0] | focus(.;0) == [1,4,[0,1]] and tuple(.) == [[4,6,7],[0,2,0],3,1]),
  ($gn_wrong_g[0] | focus(.;0) == [3,2,[3,1]] and tuple(.) == [[2,6,7],[3,2,0],3,1]),
  ($gn_wrong_n[0] | focus(.;0) == [0,4,[8,0]] and tuple(.) == [[4,6,7],[8,1,0],3,1]),
  ($gn_wrong_both[0] | focus(.;0) == [3,5,[3,0]] and tuple(.) == [[5,6,7],[3,1,0],3,1]),
  ($gn[0].independent_links[1].site == 10 and $gn[0].links[1].site == 6),
  ($rg[0] | .forced_lexical == [8,2] and .ranks[0][0].governors == [0] and
     .compatible_same_word_assignments == 8 and .best_same_word_assignments == 0),
  ($rgn[0] | .mask.sites == [1,2,3] and .ranks[0][0].eligible == 0 and
     [.ranks[0][1,2,3].eligible] == [1,1,1]),
  ($rn[0] | .forced_lexical == [null,null] and .ranks[0][1].governors == [1]),
  ($baseline[1] | tuple(.) == [null,[0,0,0],9,8] and .tuple.ordinal == null and
     .links[0].word == null and .links[0].site == 0 and
     .links[1].word == null and .links[1].site == 10 and .tuple.rival == [0,0,0]),
  ($rb[1] | .original_tuples == 9 and .compatible_tuples == 9 and
     .compatible_assignments == 81 and .best_tuples == 2 and .best_assignments == 8 and
     .ranks[0][0].governors == [1,4] and .ranks[1][10].governors == [2,5]),
  ($gn[2] | focus(.;1) == [0,3,[8,3]] and tuple(.) == [[6,3,7],[10,4,0],2,1] and
     .collision and .links[0].word == 0),
  ($rgn[2] | .mask.sign == -1 and .forced_lexical == [8,2] and .ranks[1][1].eligible == 0 and
     .compatible_same_word_assignments == 2 and .best_same_word_assignments == 1),
  ($gold[0] | .arm == "correct" and .links[0].site == 1 and .tuple.sites == [1,6,7]),
  ($swap[0] | .arm == "exchanged" and .links[0].site == 2 and .tuple.sites == [2,6,7]),
  ($gold[2] | .links[1].site == 2 and .tuple.sites == [6,2,7]),
  ($swap[2] | .links[1].site == 3 and .tuple.sites == [6,3,7]),
  ($gn[0].links[0] | .governor == {"bytes":[3,5],"tokens":[1,2]} and
    .noun == {"bytes":[39,41],"tokens":[13,14]})
] as $checks
| if all($checks[]; .) then {passed: ($checks | length)}
  else error("finite fixture checks failed at zero-based indices " +
             ([$checks | to_entries[] | select(.value == false) | .key] | tostring))
  end
