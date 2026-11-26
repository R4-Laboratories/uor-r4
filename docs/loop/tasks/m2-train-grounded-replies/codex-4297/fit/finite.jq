def check($ok; $name): if $ok then $name else error($name) end;

def exact($row):
    ($row % 2) as $side | (3 * $side) as $a | (4 + $side) as $e |
    .row == $row and .tuple.ties == 1 and .tuple.sites == [$a,$e,6] and
    .tuple.addresses == [
        {bytes:[4+4*$a,8+4*$a],tokens:[1+$a,2+$a]},
        {bytes:[4+4*$e,8+4*$e],tokens:[1+$e,2+$e]},
        {bytes:[28,32],tokens:[7,8]}
    ] and
    (.links | map([.word,.site,.governor,.noun])) == [
        [0,$a,{bytes:[0,4],tokens:[0,1]},.tuple.addresses[0]],
        [1,$e,{bytes:[32,36],tokens:[8,9]},.tuple.addresses[1]]
    ];

def pairs($reads):
    [range(3) as $pair
     | select(all(range(2*$pair;2*$pair+2); . as $row | $reads[$row] | exact($row)))
     | $pair];

def score($scores; $id): first($scores[] | select(.id == $id));

[
    check(($menu|length)==68 and
          all($menu[]; .id as $id | .group==($id / 4 | floor) and
              .step==([1,2,4,8][$id % 4]) and
              (if .group==16 then .kind=="all" and .sign==null and .slot==null
               else .kind=="center" and .sign==(.group / 8 | floor) and .slot==(.group % 8)
               end));
          "all 68 proposal IDs and their fixed steps"),
    check(($all|length)==69 and score($all;null)==
          {id:null,strongest_loss:10,all_loss:20,strictly_cheaper:4,equal_cost_rivals:2,
           eligible:4,strict_wins:2,strict_losses:2,tied_edges:0,changed:0} and
          score($single;null)==score($all;null) and
          ($base_edges[0]|[.before.rival_site,.before.rivals,.strictly_cheaper,.equal_cost_rivals,
                            .strongest_loss,.all_loss,.candidates])==
          [1,[1,2],2,1,5,10,[[1,1,0],[2,1,0],[3,6,0]]];
          "strongest tied pivot, strict margins, and equal-gold exclusion"),
    check(($single_votes[0].votes[:3]==[0,2,0]) and ($all_votes[0].votes[:3]==[2,2,0]) and
          all($all_votes[1:][]; all(.votes[]; .==0)) and
          score($single;1).strongest_loss==6 and score($single;1).all_loss==8 and
          score($all;1).strongest_loss==0 and score($all;1).all_loss==0 and
          score($all;1).strict_wins==4;
          "strongest versus every strict outranker with both objectives recomputed"),
    check(pairs($base)==[1] and pairs($single_step)==[1] and pairs($all_step)==[0,2] and
          ((pairs($base) | map(select(.!=0)))-(pairs($all_step)|map(select(.!=0))))==[1];
          "same near count with a changed original pair identity is rejected"),
    check(pairs($correct)==[0,1,2] and pairs($exchanged)==[];
          "supplied original address positive and exchanged-address negative"),
    check(($remine_votes[0].votes[:3]==[2,4,0]) and $remine_edges[0].before.rival_site==2 and
          score($remine;1).strict_wins==4 and
          ($zero_votes | all(.[]; all(.votes[]; .==0))) and
          ($zero | all(.[]; .strongest_loss==0 and .all_loss==0 and .changed==0));
          "the next round re-mines a changed pivot and excludes satisfied edges"),
    check(score($all;28).changed==0 and score($all;65).changed==2 and
          score($boundary;32).changed==2 and score($boundary;35).changed==2;
          "inactive proposal, whole-bank proposal and saturation at both i16 bounds")
] | {pass:true, checks:.}
