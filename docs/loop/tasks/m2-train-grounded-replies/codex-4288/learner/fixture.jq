def require($condition; $description):
  if $condition then $description else error($description) end;

[
  require($trained[0] | .tuple.ordinal == 0 and .tuple.best == [0,0,8] and .tuple.rival == [0,44,8]
      and .links[0].governor == {"bytes":[10,11],"tokens":[20,21]}
      and .links[0].noun == {"bytes":[24,26],"tokens":[35,36]}
      and .links[1].noun == {"bytes":[43,45],"tokens":[53,54]};
    "Directed transport selects the original far noun occurrence"),
  require($gap[0] | .tuple.ordinal == 2 and .tuple.best == [0,3,8]
      and .links[0].noun == {"bytes":[7,9],"tokens":[18,19]};
    "Bare governor gap selects the distinct same-point near occurrence"),
  require($trained[1] | .tuple.ordinal == null and .tuple.ties == 2
      and .links[0].word == 0 and .links[0].site == null and .links[0].site_ties == 2
      and .tuple.best == .tuple.rival;
    "Matched link endpoints at different noun addresses tie and abstain"),
  require($trained[2] | .tuple.ordinal == null and .tuple.ties == 2
      and .links[0].word == null and .links[0].site == 0 and .links[0].word_ties == 2;
    "Same lexical projection on different governor addresses ties and abstains"),
  require($trained[3] | .tuple.ordinal == 2 and .independent_links[1].site == 0
      and .links[1].site == 1 and .links[0].site == 2 and .tuple.best == [0,3,8];
    "Joint legal assignment survives an incompatible pair of independent winners"),
  require($trained[4] | .tuple.ordinal == null and .tuple.ties == 2
      and .links[0].site == 0 and .links[1].site == 1;
    "Heading tie with the same links abstains without concealing their addresses"),
  require($swapped[0] | .links[0].word == 1 and .links[1].word == 0 and .tuple.ordinal == 1;
    "Swapped training signs change the governor assignments"),
  require($endpoints[0] | .links[0].word == 0 and .links[1].word == 1
      and .links[0].site == 1 and .links[1].site == 0 and .tuple.ordinal == 1;
    "Swapped training nouns change attachment with signed governors preserved"),
  require($native[0] | .tuple.ordinal == 1 and .tuple.bank == 0;
    "The frozen native proposer prefers its least bank cost"),
  require($unfitted | all(.tuple.ordinal == null and .tuple.ties == 0 and .links[0].word == null);
    "Unfitted zero centers produce no authorized link"),
  require($correct[0] | .links[0].word == 0 and .links[1].word == 1 and .tuple.ordinal == 0;
    "Correct supplied edges carry both exact signed addresses"),
  require($exchanged[0] | .links[0].word == 0 and .links[1].word == 1
      and .links[0].site == 1 and .links[1].site == 0 and .tuple.ordinal == 1;
    "Exchanging supplied noun sites preserves the exact signed governor addresses"),
  require($wrong_edge[0] | .links[0].word == 0 and .links[0].site == null and .tuple.ordinal == null;
    "Replacing a nonsigned edge governor with an in-range wrong address breaks binding"),
  require($wrong_key[0] | .links[0].word == 3 and .links[0].site == null and .tuple.ordinal == null;
    "Replacing a signed query governor with an unmatched address breaks binding")
]
