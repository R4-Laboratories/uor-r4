def check($ok; $label): if $ok then $label else error($label) end;
def magnitude: if . < 0 then -. else . end;
def slots: [range(16)];
def ids: ["base", "full"] + [slots[] | "slot-" + (if . < 10 then "0" else "" end) + tostring];
def tail($x; $y; $z): [$x,$y,0,0,0,0,0,0,0,0,0,$z];
def center($group): [4] + [range(19) | 0] + tail(if $group == 1 then 10 else 0 end; 0; 0);
def proposed($score; $group):
    $score.candidate == "full" or $score.candidate == ids[2 + $group];

def check_arm($points; $slots; $scores):
    [range(16) as $id |
        [$points[] | select(.id == $id) | .coordinates[20:]] as $support |
        {id:$id,sign:($id/8|floor),slot:($id%8),active:($id < 3 or $id == 8 or $id == 9),
         support:($support|length),
         medians:(if ($support|length) == 0 then null else
                     [range(12) as $axis | [$support[][$axis]] | sort | .[((length-1)/2|floor)]]
                  end)}
    ] as $replayed |
    $replayed == $slots and ($scores|map(.candidate)) == ids and
    all($scores[]; . as $score |
        [range(16) as $id |
            center($id)[:20] +
            (if proposed($score; $id) and $replayed[$id].support > 0 then $replayed[$id].medians
             else center($id)[20:] end)
        ] as $centers |
        ([$points[] | . as $point | $centers[$point.id] as $center |
            [range(32) as $axis | ($center[$axis] - $point.coordinates[$axis]) | magnitude]
         ]) as $residual |
        $score.assigned_l1 == ($residual|map(add)|add) and
        $score.assigned_tail_l1 == ($residual|map(.[20:]|add)|add) and
        $score.changed == ([range(16) as $id | center($id) as $before |
                            range(32) as $axis | select($before[$axis] != $centers[$id][$axis])]|length));

[
    check(($assignments|map(.id)) == [0,8,0,8,1,8,1,8] and
          ($assignments|map(.endpoint_cost)) == [13,32774,13,13,4,20,10,32785] and
          ($assignments|map(.gold_site)) == [0,1,1,0,0,1,1,0] and
          ($assignments|map([.row,.sign,.slot,.endpoint_cost,.governor,.gold])) ==
          ($archived|map([.row,.sign,.before.gold_center,.before.gold_cost,.governor,.gold]));
          "archived integer gold-center replay, exact addresses, lowest-slot endpoint tie"),
    check(($endpoint_points|map(.id)) == ($assignments|map(.id)) and
          ($ordered_points|map(.id)) == ($assignments|map(.id)) and
          ($ordered_points[0].coordinates[20]) == 24 and ($ordered_points[4].coordinates[20]) == -20 and
          ($endpoint_points[1].coordinates[31]) == -32768 and ($ordered_points[1].coordinates[31]) == 32767;
          "fixed IDs even when ordered points would move to another original slot"),
    check(check_arm($endpoint_points; $endpoint_medians; $endpoint_scores) and
          check_arm($ordered_points; $ordered_medians; $ordered_scores) and
          [$endpoint_medians[0].medians, $endpoint_medians[1].medians, $endpoint_medians[8].medians] ==
          [tail(-3;-7;0), tail(10;1;0), tail(9;0;-1)] and
          [$ordered_medians[0].medians, $ordered_medians[1].medians, $ordered_medians[8].medians] ==
          [tail(24;-5;0), tail(-22;-3;0), tail(-20;0;-4)];
          "lower signed integer medians, all 16 supports, 18 objectives and changed words in each arm"),
    check(($endpoint_scores[:2]|map([.assigned_l1,.assigned_tail_l1,.changed])) == [[65632,65608,0],[65596,65572,5]] and
          ($ordered_scores[:2]|map([.assigned_l1,.assigned_tail_l1,.changed])) == [[65743,65719,0],[65631,65607,6]];
          "both geometries improve assigned L1 and change distinct coordinates")
] | {pass:true, checks:.}
