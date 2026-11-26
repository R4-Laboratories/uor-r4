# Compare only the physical readings at the same original TRAIN governors and gold sites.
def describe($values):
    ($values|sort) as $v | ($v|length) as $n |
    {min:$v[0],median:(($v[(($n-1)/2|floor)]+$v[($n/2|floor)])/2),max:$v[-1]};
if ($endpoint|length)!=512 or ($ordered|length)!=512 then error("incomplete signed TRAIN capture") else
    [range(512) as $i|$endpoint[$i] as $e|$ordered[$i] as $o|select($e.row<128)|
     if [$e.row,$e.sign,$e.gold_site,$e.governor_word]!=[$o.row,$o.sign,$o.gold_site,$o.governor_word]
     then error("original gold site or governor changed")
     else {e:$e.before,o:$o.before} end] as $v |
    if ($v|length)!=256 then error("far edge enrollment") else
     {far_edges:256,endpoint_gold:describe([$v[].e.gold_cost]),ordered_gold:describe([$v[].o.gold_cost]),
      endpoint_rival:describe([$v[].e.rival_cost]),ordered_rival:describe([$v[].o.rival_cost]),
      gold_cost_increases:([$v[]|select(.o.gold_cost>.e.gold_cost)]|length),
      rival_cost_increases:([$v[]|select(.o.rival_cost>.e.rival_cost)]|length),
      changed_gold_center:([$v[]|select(.o.gold_center!=.e.gold_center)]|length),
      changed_rival_site:([$v[]|select(.o.rival_site!=.e.rival_site)]|length)}
    end
end
