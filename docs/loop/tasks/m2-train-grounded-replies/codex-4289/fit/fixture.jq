def check($name; $condition):
  if $condition then $name else error($name) end;

[
  check("each original signed event traced"; [$base, $contrast, $positive] | all(length == 10)),
  check("only original disjoint sign-slot rivals, including repeated projection and shared BPE";
    $base[0] |
      .gold_in_slot and .governor_admitted and .eligible_for_fit and .eligible_rivals == 4 and
      [.candidates[].site] == [1, 2, 7, 8] and
      .candidates[-1].address.tokens == .governor.tokens and
      .candidates[-1].address.bytes != .governor.bytes and
      .candidates[0].address.bytes != .gold.bytes),
  check("genuine wrong, not tied, both minimum wrong occurrences retained, lowest pivot";
    $base[0].before |
      .gold_cost == 4 and .rival_cost == 1 and .margin == -3 and .loss == 4 and
      .rivals == [1, 7] and .rival_ties == 2 and .rival_site == 1 and .outcome == "loss"),
  check("gold tie is not a loss and opposite sign slots are independent";
    $base[3] |
      .sign == 1 and .eligible_rivals == 1 and [.candidates[].site] == [4] and
      .before.margin == 0 and .before.loss == 1 and .before.outcome == "tie"),
  check("center tie keeps first center and strict win contributes no update";
    $contrast[2] |
      .before.gold_center == 0 and .before.margin == 10 and .before.loss == 0 and
      .after.gold_cost == 9 and .after.margin == 12),
  check("lexical tie makes raw loss ineligible";
    $base[4] | .gold_in_slot and (.governor_admitted | not) and (.eligible_for_fit | not) and
      .before.outcome == "loss"),
  check("absent gold slot and absent wrong rival leave margin undefined";
    ($base[6] | (.gold_in_slot | not) and .governor_admitted and .before.outcome == "no_gold" and
      .before.margin == null and .before.loss == null) and
    ($base[8] | .gold_in_slot and .governor_admitted and .before.outcome == "no_rival" and
      .eligible_rivals == 0 and .before.margin == null and .before.loss == null)),
  check("contrastive remine displaces tied old rivals while another rival still outranks gold";
    $contrast[0] |
      .chosen_step == 1 and .final_step == 8 and (.history | length) == 9 and
      [.history[].rival_site] == [1,2,2,2,2,2,2,2,2] and
      .after.margin == -2 and .after.rivals == [2] and .flip == "loss_to_loss" and
      [.initial_rivals_after[] | [.site,.after.center,.after.cost,.after.margin]] ==
      [[1,1,2,-1],[7,1,2,-1]]),
  check("positive-only trajectory mines a three-way tie and then flips the strict loss";
    $positive[0] |
      .chosen_step == 4 and .final_step == 8 and (.history | length) == 9 and
      .history[1].rivals == [1,2,7] and .history[1].rival_site == 1 and
      [.history[].margin] == [-3,-2,-2,0,1,1,1,1,1] and
      .after.outcome == "win" and .flip == "loss_to_win"),
  check("pocket breaks equal hinge by strict wins and then picks the earliest matching sweep";
    $pocket[0] |
      .chosen_step == 1 and .final_step == 8 and
      [.history[].margin] == [0,2,0,2,0,2,0,2,0] and .flip == "tie_to_win" and
      .final.outcome == "tie"),
  check("unmodified blind ranker loads each elected state and keeps original source addresses";
    [$base_reader[0].tuple.sites, $contrast_reader[0].tuple.sites, $positive_reader[0].tuple.sites] ==
    [[1,3,5],[2,3,5],[0,3,5]] and $positive_reader[0].links[0].noun == $base[0].gold)
]
