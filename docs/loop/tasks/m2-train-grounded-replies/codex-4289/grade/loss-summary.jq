def count(f):map(select(f))|length;
def counts:group_by(.)|map({key:.[0],value:length})|from_entries;
def grid:
  . as $rows | ["loss","tie","win"]|map(. as $b | {key:.,value:
    (["loss","tie","win"]|map(. as $a|{key:.,value:($rows|count(.before==$b and .after==$a))})|from_entries)})|from_entries;
def summary:
 . as $all | map(select(.eligible)) |
 {events:($all|length),eligible:length,missing_gold:($all|count(.gold_in_slot|not)),unadmitted:($all|count(.governor_admitted|not)),
  no_wrong_occurrence:($all|count(.rivals_in_pool==0)),eligible_losses:count(.before=="loss"),loss_to_win:count(.before=="loss" and .after=="win"),
  before_hinge:(map(.old_hinge)|add//0),after_hinge:(map(.new_hinge)|add//0),
  before_outcomes:(map(.before)|counts),after_outcomes:(map(.after)|counts),transitions:grid,
  strict_losses_that_beat_all_original:count(.before=="loss" and .beats_all_old_rivals),
  beat_old_but_still_wrong:count(.before=="loss" and .beats_all_old_rivals and .after=="loss"),
  beat_old_but_now_tied:count(.before=="loss" and .beats_all_old_rivals and .after=="tie"),
  original_pivot_changed:count(.current_pivot_equals_old|not),initial_minimum_ties:count(.old_tie_count>1),
  original_loss_categories:(map(select(.before=="loss"))|group_by(.old_kind)|map({key:.[0].old_kind,value:(map(.after)|counts)})|from_entries)};
if length!=512 or (map([.row,.sign])|unique|length)!=512 or (map(.chosen_step)|unique|length)!=1
 or (group_by(.pair)|length)!=128 or any(group_by(.pair)[];length!=4 or (map(.id)|unique|length)!=2)
then error("signed edge census incomplete") else . end |
summary + {step:.[0].chosen_step,
 by_sign:(group_by(.sign)|map({key:(if .[0].sign==0 then "affirmed" else "excluded" end),value:summary})|from_entries),
 by_family:(group_by(.family)|map({key:.[0].family,value:summary})|from_entries),
 all_four_signed_edges_won_before:(group_by(.pair)|count(all(.[];.eligible and .before=="win"))),
 all_four_signed_edges_won_after:(group_by(.pair)|count(all(.[];.eligible and .after=="win"))),
 examples:(group_by([.transition,.old_kind])|map(sort_by(.id,.sign)|first) |
  map({id,family,admission,sign,governor,gold,old_kind,old_rival,new_rival,old_margin,old_pivot_after,new_margin,transition}))}
