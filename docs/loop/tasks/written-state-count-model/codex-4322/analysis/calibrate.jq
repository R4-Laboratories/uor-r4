# Push both signed margin boundaries through the sealed finite prompts.
# Oracle token events are reused here only to construct an exact grader fixture.
def score($side): if $side==0 then -8 else 9 end;
.common.own_score=score(.side) |
.common.peer_score=score(1-.side) |
.common.own_choice=(if .side==0 then "eos" else "copy" end) |
.common.peer_choice=(if .side==0 then "copy" else "eos" end) |
.arms.learned=.arms.correct |
.arms.swapped=.arms.opposite |
.arms.learned.steps[.contrast+2].action_score=.common.own_score |
.arms.swapped.steps[.contrast+2].action_score=.common.peer_score
