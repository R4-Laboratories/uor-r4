# Synthetic bookkeeping inputs for the result reducer. The two "echo" addresses are distinct.
def addr($a;$b;$id): {bytes:{start:$a,end:$b},tokens:{start:$id,end:($id+1)}};
def packed: {bytes:[.bytes.start,.bytes.end],tokens:[.tokens.start,.tokens.end]};
[addr(3;7;31),addr(8;12;32),addr(13;17;33),addr(18;21;34),addr(22;26;35),addr(27;30;36)] as $sites |
addr(0;2;30) as $gov |
def e($i;$arm):
  ($i%2) as $sg | (($i/2|floor)%3) as $kind |
  (if $i<2 then -1 else 1 end) as $control |
  (if $arm=="winner" then 0 elif $arm=="control" and $control>=0 then $control else null end) as $mask |
  [3,12,9,([12,5,9][$kind])] as $cost |
  [range(4) as $k | select($k!=$mask) | {site:$k,noun:($sites[$k]|packed),cost:[0,$cost[$k]],transport_center:2}] as $candidates |
  ($candidates|map(.cost[1])|min) as $min | [$candidates[]|select(.cost[1]==$min)|.site] as $wins |
  (if ($wins|length)==1 then $wins[0] else null end) as $w |
  (if $w==null then null elif $sg==0 then [$w,4,5] else [4,$w,5] end) as $tuple |
  {panel:{row:$i,case_id:("id"+($i|tostring)),pair_id:("pair"+($i/2|floor|tostring)),failed_sign:$sg,
    links:[{governor_word:0,gold_site:(if $sg==0 then 2 else 4 end)},
           {governor_word:0,gold_site:(if $sg==1 then 2 else 4 end)}],winner_site:0,control_site:$control},
   tsv:([$sg,0,$control]|@tsv),coord:{id:("id"+($i|tostring)),ordinal:$i,prompt_sha256:"fixture",sites:$sites,words:[$gov]},
   ref:{id:("id"+($i|tostring)),pair_id:("pair"+($i/2|floor|tostring)),structural_prompt_sha256:"fixture",
     turns:["","go echo echo test hat mark end"],family:"finite",admission:(if $sg==0 then "defer" else "take" end),
     native_scope:[null,null,{bytes:{start:28,end:31},tokens:$sites[5].tokens}]},
   rank:{row:$i,sign:$sg,word:0,governor:($gov|packed),requested_site:$mask,omitted_site:$mask,candidates:$candidates},
   raw:{row:$i,tuple:{sites:$tuple,addresses:(if $tuple==null then null else [$tuple[]|$sites[.]|packed] end),ties:($wins|length)},
     links:[range(2) as $sign|{word:0,governor:($gov|packed),site:(if $sign==$sg then $w else 4 end)}]},
   joined:{case_id:("id"+($i|tostring)),arm:{both_governors:true,exact:($w==2),full:($w==2),
     first_source_error:(if $w==2 then "none" elif $w==null then "joint_tie" elif $sg==0 then "affirmed_noun" else "excluded_noun" end)}}};
{neutral:[range(256)|e(.;"neutral")],winner:[range(256)|e(.;"winner")],control:[range(256)|e(.;"control")]}
