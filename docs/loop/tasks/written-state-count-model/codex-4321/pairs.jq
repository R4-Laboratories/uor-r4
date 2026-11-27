def n(f):map(select(f))|length;
def ids($arm): .arms[$arm].steps|map(.committed_id);
def model($arm): {ids:ids($arm),exact:(.arms[$arm].terminal and .arms[$arm].first_mismatch==null),first_wrong:.arms[$arm].first_mismatch};
def analyze:
 group_by(.pair_id)|map(sort_by(.side)|.[0] as $a|.[1] as $b|
 {learned:all(.[];.learned.exact),swapped:all(.[];.swapped.exact),native:all(.[];.native.exact),
   reaches:all(.[];.reached_stage),condition_ok:($a.score<0 and $b.score>0),condition_reverse:($a.score>0 and $b.score<0),
   real_both_correct:all(.[];.shared_live_action_correct),
   before:map(select(.learned.first_wrong!=null and .learned.first_wrong<.contrast+3))|length,
   failed_release:map(select(.learned.first_wrong==.contrast+3))|length,
   later:map(select(.learned.first_wrong!=null and .learned.first_wrong>.contrast+3))|length}) |
 {pairs:length,reached_both:n(.reaches),oriented_and_reached:n(.condition_ok and .reaches),oriented_but_early_failure:n(.condition_ok and (.reaches|not)),
 actual_both_actions_correct:n(.real_both_correct),
 oriented_lost_after_shared_action:n(.real_both_correct and (.learned|not)),
 gained_vs_native_pairs:n(.learned and (.native|not)),lost_vs_native_pairs:n(.native and (.learned|not)),
 gained_vs_swapped_pairs:n(.learned and (.swapped|not)),lost_vs_swapped_pairs:n(.swapped and (.learned|not)),
 wrong_before_common_token:(map(.before)|add),wrong_at_common_token:(map(.failed_release)|add),wrong_after_common_token:(map(.later)|add)};
map(. as $o|.contrast as $stage | .arms.learned.steps[$stage+2] as $step |
{pair_id,contrast,family,side,score:.common.own_score,learned:model("learned"),swapped:model("swapped"),native:model("native"),
 reached_stage:($step.stage==$stage and (ids("learned")[:$stage+2])==.expected[:$stage+2]),
 shared_live_action_correct:($step.stage==$stage and $step.selected_source==(if .side==0 then "selected_eos" else "verified_copy" end) and $step.committed_id==.expected[$stage+2])}) |
group_by(.contrast)|map({contrast:.[0].contrast,overall:analyze,families:(group_by(.family)|map({family:.[0].family,counts:analyze}))})
