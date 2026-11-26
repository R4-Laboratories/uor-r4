def job($i):
    [.arm,"fresh",.fold,.id,$model,($out+"/"+.arm+"/fresh/wire/"+.id+".json"),"-",
     if $i<4 then 1 else 0 end] | @tsv;

([.[] | select(.arm=="native")] | to_entries[] | .key as $i | .value |
    ["stock","fresh",.fold,.id,$stock_model,"-","-",if $i<4 then 1 else 0 end] | @tsv),
(foreach .[] as $row ({}; .[$row.arm] += 1; .[$row.arm]-1 as $i | $row | job($i)))
