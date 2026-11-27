# Reconstruct the single shared table along the observed optimization trajectory.
def assert_that(v;msg): if v then . else error(msg) end;
def n(v): map(select(v))|length;
$words[0] as $w | assert_that(($schedule|length)==($audit|length) and ($w|length)==4096;"invalid audit shape") |
(reduce range(0;($schedule|length)) as $i ({};
 $schedule[$i] as $s | $audit[$i] as $a | $s.joint_stage as $stage |
 (if $s.joint_copy then 1 else -1 end) as $sign | $a.changed_coordinates as $cs |
 assert_that($a.slot==$i and $a.after-$a.before==$sign*($cs|length) and ($cs|unique|length)==($cs|length);"slot invalid") |
 reduce $cs[] as $c (.; assert_that(($c>=0 and $c<4096 and ($stage==0 or $stage==1));"coordinate or stage invalid") |
   .[($c|tostring)][$stage] = ((.[($c|tostring)][$stage]//0)+$sign))
 )) as $sums |
[range(0;4096) as $i | ($sums[$i|tostring]//[]) as $c |
 {period:($c[0]//0),b:($c[1]//0),touched_both:($c[0]!=null and $c[1]!=null),stored:$w[$i]}] |
assert_that(all(.[];.period+.b==.stored);"reconstructed word differs from D11 weight") |
{checked_words:length,touched_by_both:n(.touched_both),nonzero_net_both:n(.period!=0 and .b!=0),
 opposing_net:n(.period*.b<0),reinforcing_net:n(.period*.b>0),exactly_cancelled:n(.period*.b<0 and .stored==0),
 nonzero_stored:n(.stored!=0),largest_absolute_partial:([.[]|.period,.b]|map(if .<0 then (-.) else . end)|max)}
