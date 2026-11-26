def require($ok; $reason): if $ok then . else error($reason) end;

def histogram:
    group_by(.) | map({key:(.[0]|tostring),value:length}) | from_entries;

def pairs:
    group_by(.pair_id) |
    map(sort_by(.row) |
        require(length == 2 and
            .[0].selector == "first" and .[1].selector == "later" and
            .[0].family == .[1].family and .[0].affirmative_verb == .[1].affirmative_verb and
            .[0].sign == .[1].sign and .[0].order == .[1].order and
            .[0].case_id != .[1].case_id;
            "pair does not contain the anchored first and later rows") |
        {id:.[0].pair_id,family:.[0].family,sign:.[0].sign,order:.[0].order,
         affirmative_verb:.[0].affirmative_verb,contested_verb:.[0].contested_verb,
         source:all(.[];.source),full:all(.[];.full),
         admission:all(.[];.lexical.affirmative.status=="own"),
         focus_admission:all(.[];.local_eligible),
         completed:all(.[];.completion_eligible),
         local:all(.[];.local_third.status=="win"),
         pinned:all(.[];.pinned_third.status=="win"),
         free_third:all(.[];.pinned_target_free_third.status=="win"),
         free_both:all(.[];.both_free_third.status=="win")});

def relations:
    {status:(map(.status)|histogram),
     deciding_key:(map(. | select(.first_key != null) | [.status,.first_key]|join("/"))|histogram)};

def measure:
    . as $rows | pairs as $p |
    {rows:length,pairs:($p|length),
     exact_signed_rows:(map(select(.source))|length),exact_signed_pairs:($p|map(select(.source))|length),
     exact_full_rows:(map(select(.full))|length),exact_full_pairs:($p|map(select(.full))|length),
     affirmative_gate:(map(.lexical.affirmative.status)|histogram),
     excluded_gate:(map(.lexical.excluded.status)|histogram),
     contested_gate:(map(.lexical.contested.status)|histogram),
     lexical_membership_changes:([.[].lexical_membership_changes[]] |
        group_by([.role,.before,.after]) |
        map({role:.[0].role,before:.[0].before,after:.[0].after,count:length,
             surfaces:(map(.word.word)|histogram)})),
     eligible_local_rows:(map(select(.local_eligible))|length),
     eligible_completed_rows:(map(select(.completion_eligible))|length),
     route_attribution:(map(.status)|histogram),
     pinned_third_pairs:($p|map(select(.pinned))|length),
     pinned_target_free_third_pairs:($p|map(select(.free_third))|length),
     both_free_third_pairs:($p|map(select(.free_both))|length),
     same_spelling:(map(.same_spelling)|relations),
     local_third:(map(.local_third)|relations),
     pinned_third:(map(.pinned_third)|relations),
     pinned_target_free_third:(map(.pinned_target_free_third)|relations),
     both_free_third:(map(.both_free_third)|relations),
     strongest_pinned_site_count:(map(.strongest_fixed.sites|length)|histogram),
     strongest_free_site_count:(map(.strongest_free.sites|length)|histogram)};

def row_examples($rows):
    {exact_signed:([$rows[] | select(.source)][0]),
     wrong_same_spelling:([$rows[] | select(.status | contains("same-spelling"))][0]),
     wrong_governor_same_target:([$rows[] | select(.status == "wrong-governor")][0]),
     strongest_pinned_third:([$rows[] | select(.pinned_third.status == "loss" and
           .pinned_third.first_key == "transport")][0]),
     free_governor_lexical_block:([$rows[] | select(.pinned_target_free_third.status == "loss" and
           .pinned_target_free_third.first_key == "lexical")][0]),
     missing_retain:([$rows[] | select(.contested_verb=="retain" and .pinned_third.status=="missing-target")][0]),
     admitted_retain:([$rows[] | select(.contested_verb=="retain" and .local_eligible)][0])};

def exact_ids($rows;$field):
    [$rows[] | select(.[$field]) | .id];

def effects($left;$right):
    ($left | pairs) as $a | ($right | pairs) as $b |
    require(($a|map(.id)) == ($b|map(.id)); "factorial has unmatched original pairs") |
    ["source","full","admission","focus_admission","completed","local","pinned","free_third","free_both"] |
    map(. as $id |
        exact_ids($a;$id) as $initial |
        exact_ids($b;$id) as $updated |
        {key:$id,value:{before:($initial|length),after:($updated|length),
         kept:($updated - ($updated - $initial)|length), gains:($updated - $initial),losses:($initial - $updated)}}) |
    from_entries;

def effect_on_admissions($without;$with):
    [range($without|length) as $i | $without[$i] as $a | $with[$i] as $b |
        select($a.lexical.affirmative.status != "own" and $b.lexical.affirmative.status == "own") |
        {row:$b.row,case_id:$b.case_id,pair_id:$b.pair_id,verb:$b.affirmative_verb,sign:$b.sign,
         eligible:$b.completion_eligible,pinned_win:($b.pinned_third.status == "win"),
         exact:$b.source,full:$b.full}] |
    group_by([.verb,.sign]) |
    map({verb:.[0].verb,sign:.[0].sign,rows:length,
         eligible_completions:(map(select(.eligible))|length),pinned_third_wins:(map(select(.pinned_win))|length),
         exact_signed_rows:(map(select(.exact))|length),exact_full_rows:(map(select(.full))|length),
         pairs:(group_by(.pair_id)|map(select(length == 2))|length),
         exact_signed_pairs:(group_by(.pair_id)|map(select(length == 2 and all(.[];.exact)))|length),
         exact_full_pairs:(group_by(.pair_id)|map(select(length == 2 and all(.[];.full)))|length)});

def stratified:
    . as $rows |
    {overall:measure,
     by_family:(group_by(.family) | map({family:.[0].family} + measure)),
     by_verb_sign:(group_by([.affirmative_verb,.sign]) |
        map({affirmative_verb:.[0].affirmative_verb,sign:.[0].sign} + measure)),
     by_family_verb_sign:(group_by([.family,.affirmative_verb,.sign]) |
        map({family:.[0].family,affirmative_verb:.[0].affirmative_verb,sign:.[0].sign} + measure)),
     by_governor_order:(group_by(.order) | map({order:.[0].order} + measure)),
     examples:row_examples($rows)};

def compare_lanes($a;$b):
    {paired:effects($a;$b),
     by_verb_sign:
        ($a | group_by([.affirmative_verb,.sign]) |
            map(.[0] as $key |
                map(.) as $a_subset |
                ($b | map(select(.affirmative_verb == $key.affirmative_verb and .sign == $key.sign))) as $b_subset |
                {affirmative_verb:$key.affirmative_verb,sign:$key.sign,paired:effects($a_subset;$b_subset)})),
     newly_admitted_affirmative:effect_on_admissions($a;$b)};

def pair_interaction:
    ($baseline|pairs) as $b | ($margin|pairs) as $m |
    ($eligibility|pairs) as $e | ($combined|pairs) as $c |
    ["source","full","admission","focus_admission","completed","local","pinned","free_third","free_both"] |
    map(. as $field |
        [range($b|length) as $i |
            [$b[$i][$field],$m[$i][$field],$e[$i][$field],$c[$i][$field]] as $pattern |
            {id:$b[$i].id,pattern:$pattern}] as $pairs |
        {key:$field,value:{interaction:($pairs |
          map(.pattern | map(if . then 1 else 0 end) | .[3] - .[2] - .[1] + .[0]) | add),
          combined_only:[$pairs[] | select(.pattern == [false,false,false,true]) | .id],
          stable_all_factors:[$pairs[] | select(.pattern == [true,true,true,true]) | .id]}}) |
    from_entries;

require(($controls|length) == 1 and $controls[0].gold_exact_original_pairs == 128 and
    $controls[0].swapped_exact_wrong_original_pairs == 128 and
    all([$baseline,$margin,$eligibility,$combined,$wrong][];
       length == 256 and (pairs|length) == 128 and
       (map([.row,.pair_id,.case_id]) == ($baseline|map([.row,.pair_id,.case_id]))));
    "factorial lanes or held supplied controls have missing identities") |
{
    controls:$controls[0],
    arms:{baseline:($baseline|stratified),margin:($margin|stratified),
          eligibility:($eligibility|stratified),combined:($combined|stratified),wrong:($wrong|stratified)},
    effects:{margin:compare_lanes($baseline;$margin),
        eligibility:compare_lanes($baseline;$eligibility),
        margin_after_eligibility:compare_lanes($eligibility;$combined),
        eligibility_after_margin:compare_lanes($margin;$combined),
        full_factorial:compare_lanes($baseline;$combined),
        wrong_sign:compare_lanes($baseline;$wrong)},
    paired_interaction:pair_interaction
}
