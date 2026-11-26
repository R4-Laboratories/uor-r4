def require($ok; $message): if $ok then $message else error($message) end;

[
    require(($mixed | length) == 8 * 64 and ($mixed | map(.step) | unique) == [range(8)];
            "every active sign, slot and axis on the eight pre-update states"),
    require(($mixed | map(select(.step == 0 and .sign == 0 and .slot == 0 and .axis < 8) | [.near,.far]))
            == [[2,-4],[2,-2],[2,2],[1,0],[0,1],[0,0],[0,1],[0,-1]];
            "opposition with a residual, exact cancellation, agreement, one-sided votes, zero hinge and both i16 limits"),
    require(($mixed | all(if .step == 0 and (.sign == 1 or .axis >= 8) then .near == 0 and .far == 0 else true end));
            "inactive vote dimensions and strictly winning sign produce zero"),
    require(($summary | length) == 8 and $summary[0] ==
            {step:0,opposed:2,cancelled:1,agree:1,near_only:1,far_only:3,zero:57};
            "opposition counts by sign, slot and axis with cancellation a subset"),
    require(($quality | length) == 9 and $quality[0] ==
            {step:0,hinge:15,strict_wins:5,strict_losses:3,far:{hinge:9,strict_wins:2,strict_losses:2,eligible:4},
            near:{hinge:6,strict_wins:3,strict_losses:1,eligible:4}};
            "all nine states report hinge and strict wins and losses with exact parent strata"),
    require(($events | map(select(.row == 3 and .sign == 0)) | length) == 1 and
            ($events | any(.row == 3 and .sign == 0 and .before.outcome == "win" and .before.loss == 0));
            "near axis 5 operands belong to an already won edge and emit no vote"),
    require(($all | all(.near == 0)) and
            ($all | map([.step,.sign,.slot,.axis,.far])) == ($mixed | map([.step,.sign,.slot,.axis,.near + .far]));
            "changing distance labels changes the decomposition while preserving each exact net sum")
]
