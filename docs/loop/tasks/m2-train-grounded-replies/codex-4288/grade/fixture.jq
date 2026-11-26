def mk($ok): {exact:$ok,full:$ok,both_governors:$ok,earlier:$ok,swapped:false,neutral:false,
  ties:0,eligible:1,first_source_error:(if $ok then "none" else "affirmed_noun" end)};
[range(128) as $i | ["take","defer"][] as $sign |
 {case_id:(($i|tostring)+$sign),pair_id:($i|tostring),family:("f"+(($i%4)|tostring)),
  distance:(if $i<64 then "near" else "far" end),admission:$sign,
  arms:{native:mk(false),unfitted:mk(false),correct_link:mk(true),exchanged_link:mk(false),
        governor_gap:mk(false),learned:(mk($mode!="halves" or $sign=="take") |
          if $mode=="no_far_links" and $i>=64 then .full=false else . end)}}]
