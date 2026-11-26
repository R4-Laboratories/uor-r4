def sig($read): [$read.local,$read.joint,$read.native.choice,$read.native.governor_correct];
def transition: [sig(.before),sig(.after)];
def flips_local_and_joint:
    [.before.local,.after.local,.before.joint,.after.joint]==["win","loss","win","loss"];
def nrows($r): [$r[].row_index]|unique|length;
def state_view:
    {requested_local:.requested_local.cost,rival_local:.rival_local.cost,
     requested_complete:.requested_tuple.cost,rival_complete:.rival_tuple.cost,
     requested_center:.requested_local.center,rival_center:.rival_local.center,
     native_site:.native.signed_site,native_governors:.native.signed_governors};
def witness:
    {row_index,request,pair_id,selector,sign,family,governor,requested,rival,
     before:(.before|state_view),after:(.after|state_view)};
([$historical[]|select(.historical_lost and .selector=="later")|transition]|unique) as $later |
([$historical[]|select(.historical_lost and .selector=="first")|transition]|unique) as $first |
if ($later|length)!=1 or ($first|length)!=1 then error("heterogeneous measured historical signatures")
else
  ($reads|map(select(flips_local_and_joint))) as $flips |
  ($reads|map(select(transition==$later[0]))) as $full_later |
  ($reads|map(select(transition==$first[0]))) as $full_first |
  ($reads|map(select(.selector!=null))|sort_by(.pair_id,.selector)|group_by(.pair_id)|map(
      if length!=2 or .[0].selector!="first" or .[1].selector!="later" or
          .[0].requested!=.[1].rival or .[0].rival!=.[1].requested or .[0].governor!=.[1].governor
      then error("incomplete fixed-geometry pair")
      else {id:.[0].pair_id,sign:.[0].sign,before_both:all(.[];.before.native.choice=="requested"),
            after_both:all(.[];.after.native.choice=="requested"),
            before_both_with_governor:all(.[];.before.native.choice=="requested" and .before.native.governor_correct==true),
            after_both_with_governor:all(.[];.after.native.choice=="requested" and .after.native.governor_correct==true),
            identical:(.[0]|transition)==$first[0] and (.[1]|transition)==$later[0]}
      end)) as $pairs |
  {requests:($reads|length),prompts:nrows($reads),pairs:($pairs|length),
   both_correct_before:([$pairs[]|select(.before_both)]|length),
   both_correct_after:([$pairs[]|select(.after_both)]|length),
   lost_previously_correct_pairs:([$pairs[]|select(.before_both and (.after_both|not))]|length),
   complete_paired_reproductions:[$pairs[]|select(.identical)],
   stable_requested_row_matches:($full_first|length),
   exact_failing_row_matches:($full_later|length),
   exact_failing_row_matches_on_later:([$full_later[]|select(.selector=="later")]|length),
   local_and_conditioned_win_to_loss:{requests:($flips|length),prompts:nrows($flips),
       on_first:([$flips[]|select(.selector=="first")]|length),on_later:([$flips[]|select(.selector=="later")]|length),
       third_site_on_both_passes:([$flips[]|select(.before.native.choice=="third site" and .after.native.choice=="third site")]|length),
       examples_by_family:($flips|group_by(.family)|map(.[0]|witness))},
   patterns:($reads|group_by([.selector,transition])|map({selector:.[0].selector,signature:(.[0]|transition),n:length})),
   paired_cases:$pairs}
end
