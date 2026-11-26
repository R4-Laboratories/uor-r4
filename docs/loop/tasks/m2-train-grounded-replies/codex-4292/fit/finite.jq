def require($condition; $detail): if $condition then $detail else error($detail) end;

def near_pairs($reads):
    [range(1;3) as $pair
     | select(all(range(2*$pair;2*$pair+2); . as $row | $reads[$row] |
         .tuple.sites == [0,2,4] and .tuple.ties == 1 and
         (.links | map([.word,.site,.governor.bytes,.governor.tokens,.noun.bytes,.noun.tokens])) ==
         [[0,0,[0,4],[0,1],[4,8],[1,2]],[1,2,[24,28],[6,7],[12,16],[3,4]]]))
     | $pair];

[
    require($summary == [{coordinates:5,centers:1,duplicate_centers:1,proposals:6}] and
            ($menu|map([.id,.kind,.sign,.slot,.axis,.delta[:4]])) ==
            [[0,"coordinate",0,0,0,[-1,0,0,0]],[1,"coordinate",0,0,1,[0,1,0,0]],
             [2,"coordinate",0,0,2,[0,0,1,0]],[3,"coordinate",0,0,3,[0,0,0,-1]],
             [4,"coordinate",1,0,0,[1,0,0,0]],[5,"center",0,0,null,[-1,1,1,-1]]] and
            all($menu[]; (.delta|length)==32 and all(.delta[4:][]; .==0));
            "coordinate order and center order, with singleton center deduplication"),
    require($first[0] == {id:null,hinge:10,strict_wins:0,strict_losses:4,eligible:4,changed:0} and
            $first[1] == {id:0,hinge:6,strict_wins:0,strict_losses:2,eligible:4,changed:1} and
            $first[2] == ($first[1] + {id:1});
            "both moves reduce identical signed far loss, including exact ties"),
    require(near_pairs($parent)==[1] and near_pairs($safe)==[1] and near_pairs($harmful)==[2] and
            all($harmful[2:4][]; .links[0].noun.bytes==[8,12] and .links[0].noun.tokens==[2,3]);
            "the safe move preserves the original near pair and the harmful move replaces it with a different pair at equal count"),
    require($second[3].id==2 and $second[3].changed==0 and
            $second[4].id==3 and $second[4].changed==0 and
            $second[6].id==5 and $second[6].changed==2;
            "repeat the original move menu around the new base and saturate both i16 bounds")
]
