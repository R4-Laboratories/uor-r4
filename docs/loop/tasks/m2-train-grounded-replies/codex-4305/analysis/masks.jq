[inputs|select(length>0)|[scan("[^[:space:]]+")|tonumber]|
 . as $a|$a[4] as $n|$a[5+$n] as $k|
 if ($n<1 or $k!=$n or ($a|length)!=6+$n+$k) then error("bad admission row") else
 {row:$a[0],sign:$a[1],g:$a[2],wg:$a[3],good:$a[5:5+$n],wrong:$a[6+$n:]} end |
 if ((.good|unique|length)!=$n or (.wrong|unique|length)!=$k or
   ((.good+.wrong)|unique|length)!=$n+$k) then error("duplicate or overlapping site lists") else . end] | .[]
