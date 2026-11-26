def at($row;$sign): if $row.arm.tokens==null then null else [$row.arm.selection[$sign],$row.arm.tokens[$sign]] end;
def target($row;$sign): [$row.gold[$sign],$row.gold_tokens[$sign]];
group_by(.pair_id) | map(sort_by(.selector) as $p |
    if ($p|length)!=2 then error("incomplete interior contrast") else
        $p[0] as $a | $p[1] as $b |
        [range(2) as $i|select(target($a;$i)!=target($b;$i))|$i] as $moved |
        {pair:$a.pair_id,family:$a.family,changed_roles:$moved,complete:all($p[];.arm.full),
         correctly_followed:(($moved|length)>0 and all($moved[];. as $i|
            at($a;$i)==target($a;$i) and at($b;$i)==target($b;$i))),
         chose_distinct:(($moved|length)>0 and all($moved[];. as $i|
            at($a;$i)!=null and at($b;$i)!=null and at($a;$i)!=at($b;$i))),
         abstained:any($p[];.arm.selection==null)}
    end) |
if length==32 and all(.[];.changed_roles|length>0) then
    {pairs:length,full:(map(select(.complete))|length),correct_source_flips:(map(select(.correctly_followed))|length),
     distinct_reads:(map(select(.chose_distinct))|length),pairs_with_abstention:(map(select(.abstained))|length),records:.}
else error("contrast gold source did not change on 32 enrolled pairs") end
