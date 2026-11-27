def norm: map(rtrimstr("."));
def n(f): map(select(f))|length;
def summarize:
 {pairs:length,directed:n(.directed),exact:n(.exact),exact_short:n(.short_exact),exact_long:n(.long_exact),
  bare_types:(map(.wording|norm)|unique|length),exact_types:([.[]|select(.exact)|.wording|norm]|unique|length),
  errors_short:(map(.first_wrong[0].position)|group_by(.)|map({token:.[0],n:length})),
  errors_long:(map(.first_wrong[1].position)|group_by(.)|map({token:.[0],n:length}))};
($keys|INDEX(.[];[.pair_id,.side]|@json)) as $m |
[.pairs[]|. as $p | ($m[[.pair_id,0]|@json]) as $a | ($m[[.pair_id,1]|@json]) as $b |
 if $a.pair_axis!=$b.pair_axis or [($a.directive),($b.directive)]!=.wording then error("key identity") else . end |
 .+{axis:$a.pair_axis}] as $p |
if ($p|map(.axis)|group_by(.)|map([.[0],length]))!=[["clause-order-swap",112],["direct-count-control",32],["selector-swap",112]] then error("axis census") else . end |
{scoped:([$p[]|select(.axis!="direct-count-control")]|summarize),axes:($p|group_by(.axis)|map({key:.[0].axis,value:summarize})|from_entries),all:($p|summarize)}
