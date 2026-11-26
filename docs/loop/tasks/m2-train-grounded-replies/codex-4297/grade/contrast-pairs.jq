# Both rows have the same governing verbs. Their registered selector changes the requested occurrence.
def need($ok;$why): if $ok then . else error($why) end;
need(length==64 and (map(.case_id)|unique|length)==64 and (map(.prompt_sha256)|unique|length)==64;
    "duplicate or missing matched-interior prompt") |
group_by(.pair_id) |
need(length==32 and all(.[];length==2 and (map(.admission)|unique|length)==1 and
    .[0].distance=="far" and .[1].distance=="far" and .[0].family==.[1].family and
    (map(.selector)|sort)==["first","later"]); "selector pair membership differs") |
map({pair:.[0].pair_id,family:.[0].family,distance:"far",block:.[0].block,
    ids:(sort_by(.selector)|map([.selector,.case_id,.prompt_sha256])),
    exact:all(.[];.arm.exact),full:all(.[];.arm.full),earlier:all(.[];.arm.earlier),
    exact_rows:map(select(.arm.exact))|length,full_rows:map(select(.arm.full))|length,
    both_governors_rows:map(select(.arm.both_governors))|length,neutral_rows:map(select(.arm.neutral))|length})
