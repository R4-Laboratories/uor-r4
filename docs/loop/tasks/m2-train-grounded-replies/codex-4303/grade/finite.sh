#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
out=$1; mkdir "$out"
jq -nc '
 def g($p;$s):{index:$p,text:$s};
 range(4) as $r | {row_index:$r,pair_id:(if $r<2 then "one" else "two" end),
 selector:(if $r%2==0 then "first" else "later" end),family:"synthetic",sign:1,
 governor:g(0;"retain"),opposite_governor:g(1;"reject")}' > "$out/meta"
jq -nc 'range(4) as $i|{row:$i,words:[[2,2,0,0],[1,0,0,0],[1,0,0,0]]}' > "$out/base"
jq -nc 'range(4) as $i|{row:$i,words:[[0,2,1,0],[1,0,0,0],[0,1,1,0]]}' > "$out/probe"
jq -nc '
 range(4) as $i|{row:$i,pair_id:(if $i<2 then "one" else "two" end),
 selector:(if $i%2==0 then "first" else "later" end),source:($i==0),full:false,local:{relation:"win",requested:[0,2]},
 native:{choice:"third site"},joint:{requested:(if $i==2 then null else [0,2,3] end)},
 third_fixed:{cost:(if $i==3 then null else [0,4,3] end)},third_free:{cost:[0,1,3]},requested_free:[0,2,3]}' > "$out/rows"
jq -enc --slurpfile base "$out/base" --slurpfile obs "$out/probe" --slurpfile raw "$out/rows" --slurpfile meta "$out/meta" \
    -f "$here/metrics.jq" | jq -e 'select(.source==0 and .fixed_pairs==1 and .fixed_rows==2 and .retain_admit==4 and
    .fixed_relations=={"win":2,"missing-target":1,"no-third":1} and .other_word_gains==4 and .free_pairs==0)' > "$out/metrics.json"
jq -nc '
 def rec($id;$m;$e;$n;$w;$r;$a):
 {id:$id,m:$m,e:$e,mid:(if $m==0 then "b" else "m"+($m|tostring) end),eid:(if $e==0 then "b" else "e1" end),
 sha:$id,reg:$r,near:24,t:{fixed_pairs:$w,fixed_rows:$w,blocked:0,worst:0,debt:0,source:$n,retain_admit:$a,other_losses:0}};
 [rec("b";0;0;0;0;4;0),rec("e1";0;1;0;0;4;3),rec("m1";1;0;1;2;4;0),rec("m1-e1";1;1;1;2;4;3),
  rec("m2";2;0;8;5;4;0),rec("m2-e1";2;1;8;5;3;3)]' > "$out/grid"
jq -e -f "$here/elect.jq" "$out/grid" | jq -e 'select(.choice.combined=="m1-e1" and .protected_both_gains==1)' > "$out/winner.json"
jq 'map(if .id=="m1-e1" then .reg=3 else . end)' "$out/grid" | jq -e -f "$here/elect.jq" |
    jq -e 'select(.protected==0 and .choice.lost_guard_reads==1)' > "$out/diagnostic.json"
