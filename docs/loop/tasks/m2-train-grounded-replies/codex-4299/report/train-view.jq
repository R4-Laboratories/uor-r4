def released($elections; $arm):
    if $elections[$arm].qualified then $elections[$arm].final else $elections.endpoint.base end;
def counts:
    {candidate,changed,assigned_tail_l1,assigned_l1,protected_near:(24-(.lost_protected|length)),
     near:(.near_full|length),far:(.far_full|length),far_gained,far_lost,lost_protected};

. as $elections |
if $view=="train" then
    with_entries(.key as $arm | .value |=
        ({qualified,chosen,proposals,feasible_proposals,objective_before,objective_after,
          measurements:([.base,.full,released($elections;$arm)]|map(counts))}
        | if .qualified then . else
              . + {chosen:null,diagnostic_candidate:"base",selected_operator:"endpoint",
                   selected_state:"original_protected",diagnostic_objective_after:.objective_after,
                   objective_after:null}
          end))
elif $view=="protected" then
    (["protected_near_pair","endpoint_elected","ordered_elected","endpoint_full","ordered_full"]|@tsv),
    ($elections.endpoint.base.near_full[] as $id |
     [$id,(released($elections;"endpoint").near_full|index($id)!=null),
          (released($elections;"ordered").near_full|index($id)!=null),
          ($elections.endpoint.full.near_full|index($id)!=null),
          ($elections.ordered.full.near_full|index($id)!=null)]|@tsv)
else error("unknown TRAIN report view") end
