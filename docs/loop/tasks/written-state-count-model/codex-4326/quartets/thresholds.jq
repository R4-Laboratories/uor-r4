# Count simultaneous strict STOP/COPY switches possible under one hypothetical
# integer threshold on the already observed scores. Ties never count here.
def bounds:
  map(select(.[1] - .[0] >= 2) | [.[0] + 1, .[1] - 1]) as $intervals |
  ([$intervals[][0]] | unique | map(. as $t | [$intervals[] | select(.[0] <= $t and $t <= .[1])] | length) | max // 0) as $best |
  {pairs:length, at_zero:([.[] | select(.[0] < 0 and .[1] > 0)]|length),
   individually_separable:($intervals|length), best_common_threshold:$best};
def summarize($stage):
  . as $p | reduce ["zero","period_only","post_b_only","full"][] as $m ({};
   .[$m]=([$p[].models[$m][$stage] | if .target == ["stop","copy"] then .scores else error("unopposed control") end] | bounds));
{
 selector_at_period:(map(select(.axis=="selector-swap"))|summarize("period")),
 clause_at_period:(map(select(.axis=="clause-order-swap"))|summarize("period")),
 direct_after_b:(map(select(.axis=="direct-count-control" and .contrast==1))|summarize("after_b"))
}
