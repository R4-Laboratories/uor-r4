# Three-token diagnostics with hand-assigned roots, signed terms, and scores.
def make($id;$a;$b;$u;$v;$s;$t):
  [range(2) as $i |
    (if $i==0 then $a else $b end) as $addr |
    (if $i==0 then $u else $v end) as $weights |
    (if $i==0 then $s else $t end) as $score |
    {pair_id:$id,side:$i,contrast:0,family:"finite",instruction_start:1,bias:0,
     baseline_score:$score,scores:[$score,0,0],peer_scores:[(if $i==0 then $t else $s end),0,0],residual_score:0,
     visits:[range($addr|length) as $j | {id:(10+$j+10*$i),root_bucket:$addr[$j],root_term:$weights[$j],adjacency_bucket:null,adjacency_term:null}]}];
(make("same";[1,2];[2,1];[3,4];[4,3];7;7)
 +make("zero";[1,2];[1,3];[1,0];[1,0];1;1)
 +make("cancel";[1,2];[3,4];[2,4];[3,3];6;6)
 +make("flipped";[1,2];[3,4];[-2,-4];[3,3];-6;6)) as $rows |
{rows:$rows,panel:[$rows[]|{pair_id,side,contrast,family,prefix:([27]+[.visits[].id]+[201,35,560,652,714,28])}],keys:[$rows[]|{pair_id,side,directive:(.pair_id+(.side|tostring))}]}
