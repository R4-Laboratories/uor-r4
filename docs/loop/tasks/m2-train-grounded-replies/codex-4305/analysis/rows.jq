def require($ok; $msg): if $ok then . else error($msg) end;

def address($row; $axis; $i):
  if $i == null then null
  else $row.coordinate_map[$axis][$i] as $a |
    require($i >= 0 and $a != null; "address outside original map") |
    {i:$i,bytes:[$a.bytes.start,$a.bytes.end],tokens:[$a.tokens.start,$a.tokens.end],
     text:$row.text[$a.bytes.start:$a.bytes.end]}
  end;
def span: if . == null then null else {bytes,tokens} end;
def match_label($row; $axis; $label):
  address($row;$axis;$label.index) as $a |
  require($a.text==$label.text and $a.bytes==[$label.bytes.start,$label.bytes.end] and
    $a.tokens==[$label.tokens.start,$label.tokens.end]; "label does not identify its original bytes and BPE");
def checked_link($row; $l):
  address($row;"words";$l.word) as $g | address($row;"sites";$l.site) as $n |
  require(($g|span)==$l.governor and ($n|span)==$l.noun;"selected native address disagrees with original map") |
  {g:$g,n:$n,cost:$l.best,ties:$l.link_ties};
def check_tuple($row;$t):
  if $t.ordinal==null then require($t.ties!=1;"unique tuple without an original ordinal")
  else require($t.ordinal>=0 and $row.coordinate_map.tuples[$t.ordinal]==$t.sites and
      [$t.sites[]|address($row;"sites";.)|span]==$t.addresses; "native tuple does not exist at original addresses")
  end;
def expected($x):
  if $x.sign==1 then {governors:[$x.governor.index,$x.opposite_governor.index],sites:[$x.requested.index,$x.opposite.index,$x.heading.index]}
  else {governors:[$x.opposite_governor.index,$x.governor.index],sites:[$x.opposite.index,$x.requested.index,$x.heading.index]} end;

require(($source|length)==($reads|length) and ($source|length)==($masks|length);"incomplete dense device capture") |
(INDEX($near[];.row_index)) as $enrolled | ($meta|group_by(.row_index)|map({key:(.[0].row_index|tostring),value:.})|from_entries) as $metadata |
range($source|length) as $i | ($i|tostring) as $key |
select($part!="near" or $enrolled[$key]!=null) |
$source[$i] as $row | $reads[$i] as $observed | $masks[$i] as $mask |
[$metadata[$key][]|select(.sign==$mask.sign and .governor.index==$mask.g)] as $matches |
require(($matches|length)==1 and $row.row_index==$i and $observed.row==$i and $mask.row==$i and
  ($row.text|utf8bytelength)==($row.text|length); "case, mask, or byte-address alignment mismatch") |
$matches[0] as $x | ($x.sign|if .==1 then 0 elif .== -1 then 1 else error("bad sign") end) as $side |
require($x.pair_id==$row.pair_id and $x.case_id==$row.case_id and $x.opposite_governor.index==$mask.wg;
  "source identity mismatch") |
reduce [$x.requested,$x.rival,$x.opposite,$x.heading][] as $label (.;match_label($row;"sites";$label)) |
reduce [$x.governor,$x.opposite_governor][] as $label (.;match_label($row;"words";$label)) |
expected($x) as $gold |
if $part=="near" then require($enrolled[$key].pair_id==$row.pair_id and
  $enrolled[$key].sites==$gold.sites and $enrolled[$key].governors==$gold.governors;"archived guard address changed") else . end |
check_tuple($row;$observed.tuple) |
[$observed.links[]|checked_link($row;.)] as $ls |
($ls[$side]) as $pick | ($observed.tuple.ties==1) as $unique |
($ls|map(.g.i)) as $govs | ($pick.g.i==$x.governor.index) as $gov | ($pick.n.i==$x.requested.index) as $noun |
($govs[0]!=null and $govs[1]!=null and $govs[0]!=$govs[1]) as $distinct |
(if $observed.tuple.ties==0 then "absent"
 elif $observed.tuple.ties>1 then "tie"
 elif $noun then "requested" elif $pick.n.i==$x.rival.index then "same-spelling rival"
 elif $pick.n.i==$x.opposite.index then "opposite argument" elif $pick.n.i==$x.heading.index then "heading"
 elif ($mask.good|index($pick.n.i))!=null then "other argument" else "outside argument set" end) as $kind |
{row:$i,pair:$row.pair_id,case:$row.case_id,selector:$x.selector,family:$x.family,fact:$x.fact,order:$x.governor_order,sign:$x.sign,
  expected_governor:address($row;"words";$x.governor.index),expected_noun:address($row;"sites";$x.requested.index),
  duplicate:address($row;"sites";$x.rival.index),valid:$mask.good,displaced:$mask.wrong,
  native:$pick,other_link:$ls[1-$side],tuple:$observed.tuple.sites,cost:$observed.tuple.best,
  eligible_tuples:$observed.tuple.eligible,ties:$observed.tuple.ties,kind:$kind,
  unique:$unique,governor:($unique and $gov),noun:($unique and $noun),
  signed:($unique and $gov and $noun),distinct_signed:($unique and $gov and $noun and $distinct),
  repeated_governor:($unique and ($distinct|not)),opposite:($unique and $govs[1-$side]==$gold.governors[1-$side] and $ls[1-$side].n.i==$gold.sites[1-$side]),
  heading:($unique and $observed.tuple.sites[2]==$gold.sites[2]),
  full:($unique and $govs==$gold.governors and $observed.tuple.sites==$gold.sites)}
