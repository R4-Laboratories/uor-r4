if ($grade|length)!=1 then error("missing TRAIN trace grade") else . end |
($keys|INDEX(.[];[.pair_id,.side]|@json)) as $meta |
[$grade[0].pairs[] | .pair_id as $id | ($meta[[$id,0]|@json].pair_axis) as $axis |
 if $axis!=.axis or $axis!=$meta[[$id,1]|@json].pair_axis then error("axis mismatch") else . end |
 select(.axis=="clause-order-swap" or .axis=="selector-swap")] as $scoped |
if ($scoped|length)!=224 or ($scoped|map(.pair_id)|unique|length)!=224 or ($grade[0].pairs|length)!=256 then error("unexpected TRAIN census") else . end |
{scoped_train_pairs:($scoped|length),identical_slice:([$scoped[]|select(.identical_slice)]|length),
 shared:([$scoped[]|select(.class=="identical_support")]|length)} |
. + {mode: (if .identical_slice==.scoped_train_pairs then "stop_at_slice" elif .shared>0 then "link" else "difference" end)}
