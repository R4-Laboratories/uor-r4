def require($ok; $problem): if $ok then . else error($problem) end;

def by(f): group_by(f) | map({key:(.[0]|f),value:length}) | from_entries;

def gold_gap:
    .governors.referent as $g |
    if .governor_order == "before" then
        $g.native_noun.start - $g.tokens[1]
    else
        $g.tokens[0] - $g.native_noun.end
    end |
    require(. == $g.gap; "registered gap differs from original BPE addresses");

require(length == 256 and (map(.id) | unique | length) == 256 and
    (map(.structural_prompt_sha256) | unique | length) == 256;
    "wrong number of distinct current-user prompts or native tapes") |
require((map(.structural_history)|unique|length) == 1;
    "assistant history differs across prompts") |
group_by(.family) |
require(length == 4; "expected four whole new wording families") |
map(
    require(length == 64 and (group_by(.pair_id) | length) == 32 and
        (group_by(.fact) | length) == 8 and
        all(group_by(.fact)[]; length == 8 and
            (group_by([.governor_order,.admission,.contrast.selector]) | length) == 8 and
            all(group_by(.governor_order)[]; (map(.requested_role) | unique | length) == 1)) and
        (by(.requested_role) == {home:32,job:32}) and
        (by(.requested_horizon|tostring) == {"3":32,"4":32});
        "a wording family is missing a fact, sign, governor order, selector, role or horizon") |
    {family:.[0].family, rows:length, pairs:(group_by(.pair_id)|length),
     facts:(map(.fact)|unique),
     by_direction:(group_by(.governor_order) | map({direction:.[0].governor_order,
        requested_gaps_bpe:(map(gold_gap)|unique),
        by_sign:by(.admission), by_selector:by(.contrast.selector)}))}
)
