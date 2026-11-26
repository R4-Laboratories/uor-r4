def trained_arms: ["strongest","all","ordered-all"];

def ordinary_lanes($passing):
    ["native","baseline"] + $passing + ["correct_link","exchanged_link"];

def require_lanes($condition; $why):
    if $condition then . else error($why) end;

def declared_lanes($lanes):
    ($lanes[2:-2]) as $chosen |
    ($lanes == ordinary_lanes($chosen) and ($chosen|length)>0 and
     ($chosen|length)==($chosen|unique|length) and all($chosen[]; IN(trained_arms[])));

def checked_plan($lanes; $pairs):
    . as $plan |
    require_lanes(declared_lanes($lanes) and
        ($plan|length)==$pairs*2*($lanes|length) and
        (map(.arm)|unique)==($lanes|sort) and
        (group_by(.arm)|map(map([.id,.pair_id,.admission])|sort)|unique|length)==1 and
        all(group_by(.arm)[];
            (map(.id)|unique|length)==$pairs*2 and
            (group_by(.pair_id) | length==$pairs and
             all(.[]; length==2 and (map(.admission)|sort)==["defer","take"])));
        "ordinary jobs do not cover the admitted lane and case set");

def checked_rows($plan; $lanes; $pairs):
    ($plan|checked_plan($lanes; $pairs)) as $expected |
    require_lanes(
        (map([.arm,.id,.pair_id,.admission])|sort)==
            ($expected|map([.arm,.id,.pair_id,.admission])|sort) and
        all(group_by(.arm)[];
            group_by(.pair_id) | length==$pairs and
            all(.[]; length==2 and (map(.admission)|sort)==["defer","take"]));
        "ordinary rows differ from admitted cases and paired directions");
