def require($test;$msg): if $test then . else error($msg) end;
require(length==256 and (map(.case_id)|unique|length)==256 and (map(.prompt_sha256)|unique|length)==256;
        "source rows missing or repeated") |
group_by(.pair_id) | require(length==128 and all(.[]; length==2 and (map(.admission)|sort)==["defer","take"] and
 (map(.family)|unique|length)==1 and (map(.distance)|unique|length)==1 and (map(.block)|unique|length)==1);
 "unpaired or mismatched source cohort") |
map({pair:.[0].pair_id,family:.[0].family,distance:.[0].distance,block:.[0].block,
 ids:(sort_by(.admission)|map([.admission,.case_id,.prompt_sha256])),
 exact:all(.[];.arm.exact),full:all(.[];.arm.full),earlier:all(.[];.arm.earlier),
 exact_rows:map(select(.arm.exact))|length,full_rows:map(select(.arm.full))|length,
 both_governors_rows:map(select(.arm.both_governors))|length,neutral_rows:map(select(.arm.neutral))|length})
