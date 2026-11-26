def require($ok; $problem): if $ok then . else error($problem) end;

def raw_bytes: .bytes | (.start -= 1 | .end -= 1);

def new_noun($source): .noun = ($source | raw_bytes) | .native_noun = $source.tokens;

map(
    (if .admission == "take" then [0,"affirmative","affirmed"]
     elif .admission == "defer" then [1,"excluded","excluded"]
     else error("unregistered control polarity") end) as [$sign,$role,$gold_role] |
    .native_scope[$sign] as $target | .neutral_rival as $rival |
    require($target != $rival and (.governors[$role].noun == ($target | raw_bytes));
        "same-spelling control has the wrong original source") |
    .native_scope[$sign] = $rival |
    .neutral_rival = $target |
    .neutral_rival_bytes = ($target | raw_bytes) |
    .gold_binding[$gold_role] = ($rival | raw_bytes) |
    .governors[$role] |= new_noun($rival) |
    .governors.referent |= new_noun($rival)
)
