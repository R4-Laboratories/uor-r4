def require($test; $label):
  if $test then $label else error("finite check failed: " + $label) end;

def site($r; $s): $r.local_sites[$s];
def span($bytes): $prompt[$bytes[0]:$bytes[1]];

[
  require([$rows[] | [.request,.row,.sign,.word,.requested_site,.rival_site]] ==
    [[0,1,-1,2,3,4],[1,0,1,0,0,1],[2,0,1,0,0,5],[3,0,1,3,0,1],
      [4,1,-1,2,7,4],[5,1,1,0,6,1],[6,0,1,0,2,5],[7,1,1,0,1,0],[8,2,1,0,2,0]];
    "mixed row and sign order including repeated physical rows"),
  require(all($rows[]; . as $r | (.native | del(.tuple.min_sites)) ==
    ($native[$r.row] | del(.row,.arm))); "request-native serializer agrees with standalone native output"),
  require(all($rows[]; . as $r | all([["requested","requested_site"],["rival","rival_site"]][];
    . as [$arm,$site] | ($r[$arm].tuple | [.best,.eligible,.ties]) ==
      ($r.local_sites[$r[$site]].joint | [.best,.eligible,.ties])));
    "all-site joint pass matches exact rank_joint conditioned arms"),
  require(($rows[1] | . as $r |
    site($r;0).cost == [1,0] and site($r;1).cost == [1,1] and
    site($r;0).noun == {bytes:[20,24],tokens:[4,5]} and
    site($r;1).noun == {bytes:[25,29],tokens:[5,6]} and
    span(site($r;0).noun.bytes) == "fern" and span(site($r;1).noun.bytes) == "fern" and
    site($r;0).local_rank == {cheaper:0,equal:1} and
    site($r;1).local_rank == {cheaper:1,equal:1} and
    .requested.tuple.best == [1,6,0] and .rival.tuple.best == [1,2,0] and
    site($r;0).joint.rank == {cheaper:2,equal:1} and
    site($r;1).joint.rank == {cheaper:0,equal:1});
    "same-surface distinct-address local win reverses under free joint completion"),
  require(($rows[1] | . as $r |
    .requested.links[0].word == 0 and .requested.links[0].site == 0 and
    .rival.links[0].word == 0 and .rival.links[0].site == 1 and
    site($r;0).attachment.word == 1 and site($r;0).attachment.best[0] == 0 and
    site($r;0).attachment.rival[0] == 1 and
    .requested.links[1].site == 3 and .rival.links[1].site == 4);
    "exact requested governor forced over cheaper governor, opposite free"),
  require(($rows[1] |
    .requested.tuple.sites == [0,3,5] and .rival.tuple.sites == null and
    .rival.tuple.ties == 2 and .rival.tuple.rival == .rival.tuple.best and
    .rival.tuple.min_sites == [[1],[4],[5,6]] and
    .native_third_sites == [2] and .native.tuple.sites == [2,4,5] and
    .native.links[0].word == 1 and .native.links[1].word == 2);
    "untouched third-site native winner and tied legal headings"),
  require(($rows[0] |
    .requested.links[1].word == 2 and .requested.links[1].site == 3 and
    .requested.links[0].site == 0 and .rival.links[0].site == 2 and
    .requested.tuple.best == [0,9,0] and .rival.tuple.best == [0,1,0] and
    .rival.tuple.rival == [0,5,0]);
    "excluded governor fixed while affirmative and heading remain free"),
  require(($rows[2] | . as $r |
    site($r;5).eligible and site($r;5).joint == {best:[null,null,null],eligible:0,ties:0,rank:null} and
    .rival.tuple.eligible == 0 and .rival.tuple.min_sites == [[],[],[]]);
    "locally eligible site with no legal joint completion"),
  require(($rows[3] | . as $r |
    site($r;0).cost == [16,0] and site($r;0).opposite_lexical == 16 and
    (site($r;0).strict_sign | not) and (site($r;0).eligible | not) and
    .requested.tuple.eligible == 0 and .rival.tuple.eligible == 0);
    "equal signed lexical costs fail strict sign"),
  require(($rows[4] | . as $r |
    site($r;7).role and (site($r;7).nonoverlap | not) and site($r;7).local_rank == null and
    .requested.tuple.eligible == 0 and .rival.tuple.eligible == 3);
    "overlapping word and noun kept as an ineligible original address"),
  require(($rows[5] | . as $r | (site($r;6).role | not) and
    site($r;6).nonoverlap and (site($r;6).eligible | not) and .requested.tuple.best == [null,null,null]);
    "role-ineligible heading site cannot be invented as attachment"),
  require(($rows[6] | . as $r | site($r;2).cost == [1,4] and site($r;2).transport_center == 0 and
    site($r;2).local_rank == {cheaper:2,equal:2} and site($r;5).local_rank == site($r;2).local_rank);
    "site cost ties and first minimizing transport center"),
  require(($rows[7] | .requested.tuple.sites == [1,4,5] and .requested.tuple.ties == 1 and
    .requested.tuple.rival == [1,2,1] and .rival.tuple.best == [1,6,0]);
    "native bank tiebreak appears in constrained runner"),
  require(($rows[8] | .native.tuple.best == [0,1,0] and .native.tuple.rival == [1,5,0] and
    .native.tuple.eligible == 1 and .native.links[0].word == 1 and
    .requested.tuple.best == [1,5,0] and .requested.tuple.rival == [null,null,null] and
    .requested.links[0].word == 0 and .rival.tuple.eligible == 0);
    "native runner can be a strictly higher governor at the same unique tuple")
] | {checks:., total:length}
