def need($ok; $message): if $ok then . else error($message) end;

def count(p): map(select(p)) | length;
def tally: group_by(.) | map({key:.[0],value:length}) | from_entries;

def deciding_keys($rows):
    [$rows[].deciding_keys | to_entries[]] | group_by(.key) |
    map({key:.[0].key,value:(map(.value.key)|tally)}) | from_entries;

def pair_summary:
    group_by(.pair_id) | map(
        sort_by(.row) | . as [$first,$later] |
        need(length == 2 and $first.selector == "first" and $later.selector == "later" and
            $first.sign == $later.sign and $first.order == $later.order and $first.gap == $later.gap and
            $first.requested == $later.rival and $first.rival == $later.requested;
            "pair boundaries or original same-spelling addresses differ") |
        {id:$first.pair_id,family:$first.family,sign:$first.sign,order:$first.order,
         gap:$first.gap,row:$first.row,case_ids:map(.case_id),sides:.,
         source:all(.[];.source),full:all(.[];.full),local:all(.[];.local.relation=="win"),
         joint:all(.[];.joint.relation=="win"),
         fixed_third:all(.[];.gold_vs_fixed_third=="win"),free_third:all(.[];.gold_vs_third=="win"),
         native_third:any(.[];.native.choice=="third site")}
    ) | sort_by(.row);

def summary:
    . as $p | [.[].sides[]] as $r |
    {pairs:($p|length),rows:($r|length),source_pairs:count(.source),full_pairs:count(.full),
     local_pairs:count(.local),joint_pairs:count(.joint),
     fixed_third_pairs:count(.fixed_third),free_third_pairs:count(.free_third),
     source_rows:($r | count(.source)),full_rows:($r | count(.full)),
     first_source_rows:($r | count(.selector=="first" and .source)),
     later_source_rows:($r | count(.selector=="later" and .source)),
     native_chose_correct_site_wrong_governor:($r | count(.native.choice=="requested" and (.source|not))),
     native_no_tuple_rows:($r | count(.native.ties==0)),native_tied_rows:($r | count(.native.ties>1)),
     same_spelling_local:([$r[].local.relation]|tally),same_spelling_conditioned:([$r[].joint.relation]|tally),
     local_third:([$r[].local_third_relation]|tally),fixed_third:([$r[].gold_vs_fixed_third]|tally),
     fixed_target_vs_free_third:([$r[].gold_vs_third]|tally),free_target_vs_free_third:([$r[].free_target_vs_third]|tally),
     native_choices:([$r[].native.choice]|tally),
     deciding_keys:deciding_keys($r),
     third_masked_local_rows:($r | count(.local.relation=="win" and .native.choice=="third site")),
     third_masked_joint_rows:($r | count(.joint.relation=="win" and .native.choice=="third site")),
     third_masked_local_and_joint_rows:($r | count(.local.relation=="win" and .joint.relation=="win" and .native.choice=="third site")),
     free_third_with_tied_minimum_rows:($r | count(([.third_free.sites[].ties] | add//0)>1)),
     free_third_with_alternative_governor_rows:($r | count(.governor.i as $word | any(.third_free.sites[].governors[]; .i != $word)))};

def changes($key):
    {gain_ids:[.[] | select(.from[$key]==false and .to[$key]) | .to.id],
     loss_ids:[.[] | select(.from[$key] and .to[$key]==false) | .to.id],
     both:count(.from[$key] and .to[$key]),neither:count(.from[$key]==false and .to[$key]==false)} |
    . + {gain:(.gain_ids|length),loss:(.loss_ids|length),net:((.gain_ids|length)-(.loss_ids|length))};

def movements:
    {cases:length,source:changes("source"),full:changes("full"),local:changes("local"),
     joint:changes("joint"),fixed_third:changes("fixed_third"),free_third:changes("free_third")};

def sample(p): [first(.[] | select(p))][0];

def samples:
    [.[] | . as $pair | range(2) as $i |
        {from:$pair.from.sides[$i],to:$pair.to.sides[$i]}] as $sides |
    {first_paired_source_gain:sample(.from.source==false and .to.source),
     first_paired_source_loss:sample(.from.source and .to.source==false),
     first_local_and_joint_gain_masked_by_third:sample((.from.local==false or .from.joint==false) and
         .to.local and .to.joint and .to.native_third and .to.source==false),
     first_failure_through_third:sample(.to.native_third and .to.source==false),
     first_native_tie:sample(any(.to.sides[]; .native.ties>1)),
     first_source_row_gain:($sides|sample(.from.source==false and .to.source)),
     first_source_row_loss:($sides|sample(.from.source and .to.source==false)),
     first_strict_local_reversal:($sides|sample(.from.local.relation=="loss" and .to.local.relation=="win")),
     first_strict_local_reversal_masked_by_third:($sides|sample(.from.local.relation=="loss" and
         .to.local.relation=="win" and .to.native.choice=="third site")),
     first_fixed_win_preempted_by_free_third:($sides|sample(.to.gold_vs_fixed_third=="win" and
         .to.gold_vs_third=="loss")),
     first_later_strict_local_reversal:($sides|sample(.to.selector=="later" and
         .from.local.relation=="loss" and .to.local.relation=="win"))};

def compare($from;$to):
    need(($from.pairs|map([.id,.case_ids])) == ($to.pairs|map([.id,.case_ids]));
        "the two physical readers saw different original pair identities") |
    [range($to.pairs|length) as $i | {from:$from.pairs[$i],to:$to.pairs[$i]}] as $joined |
    {from:$from.id,to:$to.id,all:($joined|movements),
     by_family:($joined|group_by(.to.family)|map({family:.[0].to.family}+movements)),
     by_sign:($joined|group_by(.to.sign)|map({sign:.[0].to.sign}+movements)),
     historic_gaps:($joined|map(select(.to.gap==[25,18]))|movements),
     examples:($joined|samples),historic_gap_examples:($joined|map(select(.to.gap==[25,18]))|samples)};

need(length >= 2 and (map(.id)|unique|length) == length;
    "fresh read lanes are missing or repeated") |
([$election[0].selected,$election[0].best_initial,$election[0].best_learned,"original","median"] |
    map(select(.!=null))) as $needed |
. as $observations |
need(all($needed[]; . as $id | any($observations[]; .id==$id));
    "a prespecified comparator or elected state is missing") |
($observations|map({id,pairs:(.rows|pair_summary)})) as $states |
($states | map(select(.id==$election[0].selected)) | .[0]) as $selected |
{
    selection:$election[0].selection, selected:$selected.id,
    comparison_order:($states|map(.id)),
    election_details:[$election[0].ranking[]? | .id as $id | select(any($states[];.id==$id))],
    states:($states|map({id,all:(.pairs|summary),
        by_family:(.pairs|group_by(.family)|map({family:.[0].family}+summary)),
        by_sign:(.pairs|group_by(.sign)|map({sign:.[0].sign}+summary)),
        by_governor_order:(.pairs|group_by(.order)|map({order:.[0].order}+summary)),
        historic_gaps:(.pairs|map(select(.gap==[25,18]))|summary)})),
    comparisons:[$states[]|select(.id != $selected.id)|compare(.;$selected)],
    initial_vs_learned:(if $election[0].best_initial!=null and $election[0].best_learned!=null then
        compare(($states[]|select(.id==$election[0].best_initial));
            ($states[]|select(.id==$election[0].best_learned))) else null end)
}
