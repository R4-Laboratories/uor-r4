def require($test; $message): if $test then . else error($message) end;
def valid_cost($n): type=="array" and length==$n and all(.[]; type=="number" and floor==. and .>=0);
def cmp_cost($a;$b):
    if $a==null and $b==null then "neither"
    elif $a==null then "requested-ineligible"
    elif $b==null then "rival-ineligible"
    elif $a<$b then "win" elif $a>$b then "loss" else "tie" end;
def as_cost($x;$n):
    if $x==null or ($x|type)=="array" and all($x[]; .==null) then null
    elif $x|valid_cost($n) then $x else error("invalid integer score") end;
def compact_addr: if .==null then null else {index,text,bytes,tokens} end;
def span_text($t;$span):
    if ($t|utf8bytelength)==($t|length) then $t[$span.start:$span.end]
    else (reduce ($t|explode[]) as $c ({p:0,out:[]};
        (.p) as $start | .p+=([$c]|implode|utf8bytelength) |
        if $start>=$span.start and .p<=$span.end then .out+=[$c] else . end) | .out|implode)
    end;
def addr($row;$kind;$i):
    if $i==null then null
    else $row.coordinate_map[$kind][$i] as $a |
        require($a!=null and $i>=0; "index outside coordinate map") |
        {index:$i,bytes:$a.bytes,tokens:$a.tokens,text:span_text($row.text;$a.bytes)}
    end;
def same_or_none($s;$x): if $s==null then null else $s==$x end;
def where_site($i;$m):
    if $i==null then "unresolved"
    elif $i==$m.requested_site_index then "requested"
    elif $i==$m.rival_site_index then "same-word rival"
    else "third site" end;
def tuple_info($x;$row;$idx;$m):
    $x.tuple as $t | (if $t.sites==null then [null,null,null] else $t.sites end) as $sites |
    ($t.min_sites//$x.native_min_sites//($sites|map(if .==null then [] else [.] end))) as $sets |
    ([$sets[]|if length==1 then addr($row;"sites";.[0]) else null end]) as $consensus |
    {cost:as_cost($t.best;3), runner_up:as_cost($t.rival;3), ties:$t.ties,eligible:$t.eligible,
     ordinal:$t.ordinal,sites:[$sites[]|addr($row;"sites";.)],
     signed_governors:[$x.links[]|addr($row;"words";.word)], signed_site:addr($row;"sites";$sites[$idx]),
     opposite_site:addr($row;"sites";$sites[1-$idx]),heading:addr($row;"sites";$sites[2]),
     opposite_correct:same_or_none($sites[1-$idx];$m.opposite.index),heading_correct:same_or_none($sites[2];$m.heading.index),
     consensus_signed_site:$consensus[$idx],consensus_opposite_site:$consensus[1-$idx],consensus_heading:$consensus[2],
     min_sites:$sets};
def local_info($z;$site):
    ($z.local_sites|map(select(.site==$site))) as $got |
    require(($got|length)==1;"missing or duplicated original site") |
    $got[0] as $x | as_cost($x.cost;2) as $raw |
    (if $x.eligible then $raw else null end) as $cost |
    require((($x.eligible==true or $x.eligible==false) and ($x.eligible==false or $cost!=null));"local eligibility has no finite score") |
    {cost:$cost,eligible:$x.eligible,role:$x.role,nonoverlap:$x.nonoverlap,strict_sign:$x.strict_sign,
     best_complete_cost:as_cost($x.joint.best;3),complete_rank:$x.joint.rank,
     center:$x.transport_center,cheaper:(if $cost!=null then [$z.local_sites[]|select(.eligible and .cost<$cost)]|length else null end),
     equal:(if $cost!=null then [$z.local_sites[]|select(.eligible and .cost==$cost)]|length else null end)};
def state($z;$row;$m):
    (if $m.sign==1 then 0 elif $m.sign== -1 then 1 else error("bad sign") end) as $i |
    require([$z.row,$z.sign,$z.word,$z.requested_site,$z.rival_site]==
        [$m.row_index,$m.sign,$m.governor_word_index,$m.requested_site_index,$m.rival_site_index];"request identity drift") |
    local_info($z;$m.requested_site_index) as $g | local_info($z;$m.rival_site_index) as $r |
    tuple_info($z.requested;$row;$i;$m) as $jg | tuple_info($z.rival;$row;$i;$m) as $jr |
    tuple_info($z.native;$row;$i;$m) as $n |
    cmp_cost($g.cost;$r.cost) as $local | cmp_cost($jg.cost;$jr.cost) as $joint |
    where_site($n.signed_site.index;$m) as $which |
    where_site($n.consensus_signed_site.index;$m) as $consensus |
    ([$z.local_sites[]|select(.joint.eligible>0 and (.joint.best|valid_cost(3)) and .joint.rank.cheaper==0)|
        {source:addr($row;"sites";.site),cost:.joint.best,ties:.joint.ties}]) as $best_fixed |
    require($g.best_complete_cost==$jg.cost and $r.best_complete_cost==$jr.cost;"conditional and per-site best disagree") |
    require(all([[$jg,$m.requested_site_index],[$jr,$m.rival_site_index]][];
        . as [$t,$site]|$t.ties!=1 or ($t.signed_site.index==$site and $t.signed_governors[$i].index==$m.governor_word_index));
        "unique conditioned tuple did not contain specified link") |
    {local:$local,joint:$joint,requested_local:$g,rival_local:$r,requested_tuple:$jg,rival_tuple:$jr,
     best_at_fixed_governor:$best_fixed,
     native:($n+{choice:$which,consensus_choice:$consensus,
       governor_correct:(if $n.ties==0 then null else same_or_none($n.signed_governors[$i].index;$m.governor_word_index) end),
       local_at_chosen:(if $n.consensus_signed_site!=null then local_info($z;$n.consensus_signed_site.index) else null end)}),
     stage:(if $local=="loss" then "local loss"
        elif $local=="win" and $joint=="loss" then "joint-only reversal"
        elif $local=="tie" then "local tie"
        elif $local!="win" or ($joint!="win" and $joint!="tie" and $joint!="loss") then "eligibility"
        elif $joint=="tie" then "joint tie"
        elif $which=="third site" and $n.signed_governors[$i].index==$m.governor_word_index then "third site at correct governor"
        elif $which=="third site" then "third site with different governor"
        elif $which=="unresolved" and $consensus!="unresolved" then "tuple tie with resolved signed site"
        elif $which=="unresolved" then "unresolved native read"
        elif $which!="requested" or $n.signed_governors[$i].index!=$m.governor_word_index then "different governor"
        else "requested wins" end)};
require(($meta|length)==($before|length) and ($meta|length)==($after|length);"incomplete request capture") |
range($meta|length) as $id | $meta[$id] as $m | $rows[$m.row_index] as $row |
require($row!=null and $m.requested.text==$m.rival.text and $m.requested_site_index!=$m.rival_site_index and
    ($m.requested|compact_addr)==addr($row;"sites";$m.requested_site_index) and
    ($m.rival|compact_addr)==addr($row;"sites";$m.rival_site_index) and
    ($m.governor|compact_addr)==addr($row;"words";$m.governor_word_index) and
    $before[$id].request==$id and $after[$id].request==$id;"original governor/noun/spelling enrollment mismatch") |
({request:$id,cohort:$cohort} + ($m|{row_index,pair_id,case_id,family,fact,distance,role,sign,selector,historical_lost,
    governor:(.governor|compact_addr),requested:(.requested|compact_addr),rival:(.rival|compact_addr),
    opposite_governor:(.opposite_governor|compact_addr),opposite:(.opposite|compact_addr),heading:(.heading|compact_addr)}) +
 {before:state($before[$id];$row;$m),after:state($after[$id];$row;$m)})
