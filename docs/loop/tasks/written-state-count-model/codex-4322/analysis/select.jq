# Input is an object keyed by configuration and then split (grade.jq outputs).
def values_at($split): .[$split].contrasts["0"] | {directed,exact,pairs};
. as $all |
if (keys != ["balanced","baseline","both","weak"]) or
   any(.[]; [.train.contrasts["0"].pairs,.train.contrasts["1"].pairs,
             .dev.contrasts["0"].pairs,.dev.contrasts["1"].pairs,
             .protect.overall.pairs,.protect.contrasts["1"].pairs]
            !=[192,192,200,100,50,50]) or
   $all.baseline.protect.overall.exact!=50
then error("incomplete phase 1 or baseline protected-pair reproduction failed")
else
  ($all.baseline|values_at("train")) as $train |
  ($all.baseline|values_at("dev")) as $dev |
  [to_entries[] | select(.key!="baseline") |
    {config:.key,train:(.value|values_at("train")),dev:(.value|values_at("dev")),
     protected_pairs:.value.protect.overall.pairs,protected_exact:.value.protect.overall.exact}] |
  map(. + {checks:{train_directed:(.train.directed>$train.directed),
                   dev_directed:(.dev.directed>$dev.directed),
                   dev_exact:(.dev.exact>$dev.exact),
                   protect:(.protected_pairs==50 and .protected_exact==50)}}) |
  map(. + {eligible:all(.checks[];.)}) |
  sort_by(.config as $rank | [-.dev.exact,-.dev.directed,-.train.directed,
           (if .config=="both" then 2 else 1 end),
           (["balanced","weak","both"]|index($rank))]) |
  {baseline:{train:$train,dev:$dev,protected_exact:$all.baseline.protect.overall.exact},
   eligible:(map({key:.config,value:.eligible})|from_entries), candidates:.,winner:(map(select(.eligible))[0].config)}
end
