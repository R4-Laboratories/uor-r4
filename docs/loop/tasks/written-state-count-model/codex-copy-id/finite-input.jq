# Prior sealed prompts provide instrument controls before fresh scoring.
def control:
  . + {stratum:"pre_fit_finite",expected_second:1,expected_terminal_after_first:false,
       expected_terminal_after_second:true,wrong_same_id_pos:null};
if .pair_id == "probe-261" or .pair_id == "probe-262" or .pair_id == "finite-empty"
   or .pair_id == "finite-unscoped" then
  control | .cohort="finite_instrument"
elif .group_id=="fcf0e2827db09a8f2089f57e" and .cohort=="controlled" then
  control | .cohort="finite_collision" |
  ., (select(.assignment==0 and .side==0) | .pair_id="instrument-same-token-other-address" |
      .wrong_same_id_pos=.context_start | .prefix[.wrong_same_id_pos]=.target)
else empty end
