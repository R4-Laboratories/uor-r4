# Inspect trajectories by original committed IDs using the independently graded pairs.
{contrasts:(.contrasts|with_entries(.value |= {pairs,directed,inverted,actual_both_directed,exact,short_exact,long_exact,margin_pass_both,open_cap,
 errors_short:(.wrong_short|group_by(.value.position)|map({token:.[0].value.position,n:(map(.n)|add)})),
 errors_long:(.wrong_long|group_by(.value.position)|map({token:.[0].value.position,n:(map(.n)|add)}))})),
 wording:(.pairs | group_by(.wording|map(rtrimstr(".")))|map({family:.[0].family,wording:(.[0].wording|map(rtrimstr("."))),pairs:length,
 directed:(map(select(.directed))|length),exact:(map(select(.exact))|length)}))}
