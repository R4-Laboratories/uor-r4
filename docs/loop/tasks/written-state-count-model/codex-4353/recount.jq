# Run with jq -n, --arg pool POOL, and --slurpfile parameters:
# blind = source blind.jsonl, gold = source truth.jsonl,
# emitted = blind-generation.jsonl, ranks = held-ranks.jsonl,
# pairs = held-pairs.jsonl, leftovers = held-unpaired.jsonl, keys = pair keys JSONL.
# Build the last file privately from each truth ID using the exact registered salt:
# jq -r '.id' truth.jsonl | while IFS= read -r id; do
#   key=$(printf '4353-pair-v1%s' "$id" | sha256sum); key=${key%% *}
#   jq -cn --arg id "$id" --arg key "$key" '{id:$id,key:$key}'
# done > pair-keys.jsonl
# Outputs contain counts only. This recount does not read held-grades or held-summary.

def require($ok; $why): if $ok then . else error($why) end;
def arms: ["zero","geometry-435300","ordered-435300","geometry-435301","ordered-435301"];
def lexical_offset: if .gold[0] == 223 then 1 else 0 end;
def lexical_target: .gold[lexical_offset];
def space_stratum: if lexical_offset == 1 then "standalone_space" else "no_standalone_space" end;
def row_key: .arm + "/" + .id;
def pair_key: .arm + "/" + .left_id + "/" + .right_id;
def sum: add // 0;
def count(f): map(select(f)) | length;
def score_integer: type == "number" and floor == . and . > -1125899906842624 and . < 1125899906842624;
def ranking: type == "number" and floor == . and . >= 1 and . <= 4095;

# Review the actual token tape independently. A cap following a correct prefix
# has a lower bound on first wrong position and no observed wrong ID.
def grade($g; $e):
    [$g.gold, $e.emitted] | map(length) | min as $joint
    | ([range(0; $joint) | select($g.gold[.] != $e.emitted[.])] | first // $joint) as $shared
    | ($g | lexical_offset) as $slot
    | {
        stop: $e.stop,
        first_wrong: (if $shared < $joint then $shared + 1 else null end),
        correct_prefix: $shared,
        exact_eos: ($e.stop == "eos" and $e.emitted == $g.gold),
        cap_censored: ($e.stop == "cap" and $shared == ($e.emitted | length)
            and ($g.gold | length) > ($e.emitted | length)),
        early_eos: ($e.stop == "eos" and ($e.emitted | length) < ($g.gold | length)),
        gold_longer_than_cap: (($g.gold | length) > 32),
        unknown_emission: ($e.emitted | index(2) != null),
        lexical_reached: (($e.emitted | length) > $slot and ($slot == 0 or $e.emitted[0] == 223)),
        lexical_match: (($e.emitted | length) > $slot and ($slot == 0 or $e.emitted[0] == 223)
            and $e.emitted[$slot] == $g.gold[$slot])
    };

def assignment_order:
    if length == 0 then {matched: [], leftovers: []}
    else .[0] as $a | .[1:] as $rest
      | ([range(0; $rest | length) | select($rest[.].target != $a.target)] | first) as $match
      | if $match == null then
            ($rest | assignment_order) | .leftovers = ([$a.id] + .leftovers)
        else
            ($rest[0:$match] + $rest[$match+1:] | assignment_order)
            | .matched = ([[$a.id,$rest[$match].id]] + .matched)
        end
    end;

def arm_summary:
    (map(.rank.gold_rank) | sort) as $r
    | length as $n
    | {
        rows: $n,
        first_wrong: count(.first_wrong != null),
        first_wrong_position_histogram: (map(.first_wrong) | map(select(. != null)) | group_by(.)
            | map({key:(.[0] | tostring),value:length}) | from_entries),
        exact_eos: count(.exact_eos),
        early_eos: count(.early_eos),
        eos: count(.stop == "eos"),
        cap: count(.stop == "cap"),
        cap_censored: count(.cap_censored),
        gold_longer_than_cap: count(.gold_longer_than_cap),
        unknown_emission: count(.unknown_emission),
        live_lexical_reached: count(.lexical_reached),
        live_lexical: count(.lexical_match),
        conditional_gold_rank_one: count(.rank.gold_rank == 1),
        prior_gold_rank_one: count(.rank.gold_prior_rank == 1),
        conditional_gold_strictly_beats_rival: count(.rank.gold_score_q48 > .rank.rival_score_q48),
        conditional_gold_vs_space: {
            beats: count(.rank.gold_score_q48 > .rank.space_score_q48),
            loses: count(.rank.gold_score_q48 < .rank.space_score_q48),
            ties: count(.rank.gold_score_q48 == .rank.space_score_q48)
        },
        top_ties: count(.rank.top_ties > 1),
        full_address_alias_blocked: count(.rank.alias_blocked != 0),
        absent_target: count(.target_absent),
        absent_target_conditional_rank_one: count(.target_absent and .rank.gold_rank == 1),
        floor_prior_target_rank_one: count(.rank.gold_prior_q48 == 1 and .rank.gold_rank == 1),
        mean_gold_rank: [($r|sum),$n],
        median_gold_rank: (if $n == 0 then null else [($r[(($n-1)/2|floor)] + $r[($n/2|floor)]),2] end)
    };

def paired_summary:
    {rows:length, strict_two_way:count(.strict), positive_assignment:count(.advantage > 0),
     tied_assignment:count(.advantage == 0), negative_assignment:count(.advantage < 0)};

def compare($a; $b; $pair_a; $pair_b):
    [range(0; $a|length) | {trial:$a[.], control:$b[.]}] as $rows
    | [range(0; $pair_a|length) | {trial:$pair_a[.], control:$pair_b[.]}] as $pair_rows
    | {
        rows: ($rows | length),
        first_wrong_progress: ($rows | {
            improved: count(.trial.correct_prefix > .control.correct_prefix),
            unchanged: count(.trial.correct_prefix == .control.correct_prefix),
            worse: count(.trial.correct_prefix < .control.correct_prefix)
        }),
        live_lexical: ($rows | {
            gains: count(.trial.lexical_match and (.control.lexical_match|not)),
            losses: count(.control.lexical_match and (.trial.lexical_match|not)),
            both: count(.trial.lexical_match and .control.lexical_match)
        }),
        conditional_lexical: ($rows | {
            gains: count(.trial.rank.gold_rank == 1 and .control.rank.gold_rank != 1),
            losses: count(.control.rank.gold_rank == 1 and .trial.rank.gold_rank != 1)
        }),
        exact_eos: ($rows | {
            gains: count(.trial.exact_eos and (.control.exact_eos|not)),
            regressions: count(.control.exact_eos and (.trial.exact_eos|not)),
            preserved: count(.control.exact_eos and .trial.exact_eos)
        }),
        strict_two_way: ($pair_rows | {
            gains: count(.trial.strict and (.control.strict|not)),
            losses: count(.control.strict and (.trial.strict|not)),
            both: count(.control.strict and .trial.strict)
        })
    };

$gold | length as $n
| INDEX($gold[]; .id) as $g
| INDEX($blind[]; .id) as $b
| INDEX($keys[]; .id) as $k
| INDEX($emitted[]; row_key) as $e
| INDEX($ranks[]; row_key) as $r
| INDEX($pairs[]; pair_key) as $p
| (arms | length) as $arm_n
| require($n > 0 and ($g|length) == $n and ($b|length) == $n and ($blind|length) == $n
    and ($keys|length) == $n and ($k|length) == $n and ($e|length) == ($n*$arm_n)
    and ($emitted|length) == ($e|length) and ($r|length) == ($e|length)
    and ($ranks|length) == ($r|length); "held row matrix or key cardinality differs")
| [($g|keys[]) as $id | $g[$id] as $t | $b[$id] as $prompt | $k[$id].key as $key
    | require(($id | test("^[0-9a-f]{24}$")) and ($key | strings | test("^[0-9a-f]{64}$"))
        and $prompt.id == $id and ($prompt.prefix|length)>0
        and ($t.gold|length)>=2 and $t.gold[-1]==1
        and all($t.gold[0:-1][]; .>2 and .<4096)
        and ($t|lexical_target)>2 and ($t|lexical_target)!=223;
        "malformed held source or missing original lexical target")
    | {id:$id, key:$key, target:($t|lexical_target), stratum:($t|space_stratum)}] as $sources
| [$sources | group_by(.stratum)[] | sort_by(.key,.id) | assignment_order] as $assign
| ($assign|map(.matched)|flatten(1)) as $matched
| ($assign|map(.leftovers)|flatten|sort) as $unpaired
| require(($pairs|length) == ($matched|length)*$arm_n and ($p|length) == ($pairs|length)
    and ($leftovers|map(.id)|sort) == $unpaired; "wrong registered assignment or unpaired matrix")
| [arms[] as $arm
    | {arm:$arm, grades:[
        $sources[] | .id as $id | $g[$id] as $t | $e[$arm+"/"+$id] as $out | $r[$arm+"/"+$id] as $rank
        | require($out != null and $rank != null and $out.emitted == ($out.steps|map(.id))
            and ($out.emitted|length)>=1 and ($out.emitted|length)<=32
            and all($out.emitted[]; .>0 and .<4096)
            and all($out.emitted[0:-1][]; .!=1)
            and (($out.stop=="eos" and $out.emitted[-1]==1)
                or ($out.stop=="cap" and ($out.emitted|length)==32 and $out.emitted[-1]!=1));
            "malformed persisted actual emissions")
        | require($rank.gold_lexical_id == ($t|lexical_target)
            and $rank.reference_lexical_offset == ($t|lexical_offset)
            and ($rank.score.gold_rank|ranking) and ($rank.score.gold_prior_rank|ranking)
            and $rank.score.rival != $rank.gold_lexical_id
            and (($rank.score.gold_rank==1) == ($rank.score.token==$rank.gold_lexical_id))
            and (($rank.score.gold_prior_rank==1) == ($rank.score.frozen_token==$rank.gold_lexical_id));
            "GPU reference rank does not name the authentic lexical target")
        | grade($t;$out) + {rank:$rank.score,
            target_absent: (($b[$id].prefix | index($rank.gold_lexical_id)) == null),
            stratum:($t|space_stratum)}
        | require((.lexical_reached|not)
            or ($out.emitted[$rank.reference_lexical_offset] == $rank.score.token);
            "reference score equivocates on the actual self-fed lexical boundary")
    ], assignments:[
        $matched[] | .[0] as $a | .[1] as $b
        | $p[$arm+"/"+$a+"/"+$b] as $row
        | require($row != null and $row.space_stratum == ($g[$a]|space_stratum)
            and $row.left_gold_id == ($g[$a]|lexical_target) and $row.right_gold_id == ($g[$b]|lexical_target)
            and $row.left_own == $r[$arm+"/"+$a].score and $row.right_own == $r[$arm+"/"+$b].score
            and all([$row.left_own.gold_score_q48, $row.right_own.gold_score_q48,
              $row.left_swapped.gold_score_q48, $row.right_swapped.gold_score_q48][]; score_integer);
            "pair result differs from registered identity or original GPU scores")
        | ($row.left_own.gold_score_q48 + $row.right_own.gold_score_q48
            - $row.left_swapped.gold_score_q48 - $row.right_swapped.gold_score_q48) as $advantage
        | ($row.left_own.gold_score_q48 > $row.left_swapped.gold_score_q48
            and $row.right_own.gold_score_q48 > $row.right_swapped.gold_score_q48) as $strict
        | require($row.strict_two_way == $strict and $row.assignment_advantage_q48 == $advantage;
            "reported pair reduction differs from raw Q48 mass")
        | {strict:$strict, advantage:$advantage, stratum:$row.space_stratum}
    ]}] | INDEX(.[];.arm) as $all
| {
    schema:"uor-r4/4353-held-independent-recount/1", pool:$pool, rows:$n,
    matched_pairs:($matched|length), unpaired:($unpaired|length),
    first_wrong_progress_unit:"number of verified initial reference IDs, with no fabricated mismatch at cap",
    rank_rational_columns:["numerator","denominator"],
    arms:($all | map_values({
        all:(.grades|arm_summary),
        strata:(.grades|group_by(.stratum)|map({key:.[0].stratum,value:arm_summary})|from_entries),
        assignments:(.assignments|paired_summary),
        assignment_strata:(.assignments|group_by(.stratum)|map({key:.[0].stratum,value:paired_summary})|from_entries)
    })),
    comparisons:([
        ["435300","435301"][] as $seed | ("geometry-"+$seed) as $geo
        | ["zero", ("ordered-"+$seed)][] as $control
        | {key:($geo+"_vs_"+$control),
            value:compare($all[$geo].grades;$all[$control].grades;$all[$geo].assignments;$all[$control].assignments)}
    ]|from_entries)
}
