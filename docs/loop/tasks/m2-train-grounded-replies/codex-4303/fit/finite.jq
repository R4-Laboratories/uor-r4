def require($condition; $message):
  if $condition then true else error($message) end;
def rows($states; $name):
  [$states[] | select(.candidate == $name)];
def positive:
  select(.row < 4);
def skip_and_tie:
  .[4:6] as $tied
  | .[6:8] as $no_third
  | .[8:10] as $bad_target
  | all($tied[];
        .opportunity == "eligible" and .margin == 0 and (.strict_win | not)
        and .joint[1].min_sites == [2,3] and .joint[1].eligible == 3 and .joint[1].ties == 3)
    and all($no_third[];
        .opportunity == "no_third" and .margin == null and .debt == 0
        and .joint[0].eligible == 1 and .joint[1].eligible == 0)
    and all($bad_target[];
        .opportunity == "target_non_strict_sign" and .margin == null and .debt == 0
        and all(.joint[]; .eligible == 0));

rows($fit; "start") as $start
| rows($fit; $picked) as $learned
| rows($known; "start") as $edited
| {
    coverage: require(
      all([$start, $learned, $edited][]; length == 10 and (map(.row) == [range(10)]))
      and ($start[0:4] | map([.sign,.selector])) == [[1,"first"],[1,"later"],[-1,"first"],[-1,"later"]];
      "fixture does not cover both signs and selectors"
    ),
    starting_loss: require(
      all($start[] | positive;
        .margin == -2 and .joint[1].min_sites == [2] and .joint[1].tuple == 2 and (.strict_win | not))
      and all($oracle_start[] | select(.row < 4);
          .local_sites[.rival_site].joint.best < .local_sites[2].joint.best
          and .local_sites[7].cost[1] < .local_sites[2].cost[1]
          and .local_sites[7].joint.best > .local_sites[2].joint.best
        );
      "baseline does not falsify rival inclusion or local-only third selection"
    ),
    bilateral_known_update: require(
      all($edited[] | positive;
        .margin == 2 and .strict_win and .joint[1].min_sites == [3] and .joint[1].tuple == 3
        and .joint[1].ties == 2 and .joint[1].eligible == 4);
      "known bilateral update did not improve or rerank both signs and selectors"
    ),
    actual_fit: require(
      all($learned[] | positive; .strict_win and .margin > 0 and .joint[1].min_sites == [3])
      and any($schedule[]; .phase == "fit" and .path == "balanced" and .accepted and .candidate == $picked);
      "accepted TRAIN update did not win both signs and selectors"
    ),
    tie_and_skips: require(
      all([$start, $learned, $edited][]; skip_and_tie)
      and any($schedule[];
        .phase == "start" and .score.opportunities.eligible == [4,2]
        and .score.opportunities.no_third == [0,2] and .score.opportunities.target_non_strict_sign == [2,0]
        and .score.comparable_pairs == [2,1]);
      "ties or missing opportunities changed TRAIN counts"
    ),
    budget: require(
      all($complete[];
        .proposal_evaluations <= 1377 and .exported <= 65 and .base_replays <= 16
        and .snapshot_replays == .exported and .state_bytes == 13336);
      "exceeded the frozen fit envelope"
    )
  }
