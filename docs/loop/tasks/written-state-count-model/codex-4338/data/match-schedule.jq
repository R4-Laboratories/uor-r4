def select_groups($kind;$n):
  [.[] | select(.kind==$kind)] | sort_by(.rank) |
  if length < $n then error("insufficient " + $kind) else .[:$n] | map(.ids) end;

(select_groups("mixed";60)) as $mixed |
(if $mix == "natural" then select_groups("both";72)
 else select_groups("both";24) + (select_groups("single";96) | flatten as $ids |
   [range(0;96;2) as $i|$ids[$i:$i+2]]) end) as $positive |
# A fixed permutation of 132 blocks; expose the same polarity at every step
# under both mixtures. The unpaired historical rows keep their null mate IDs.
[range(132) | ((. * 53 + 17) % 132) as $block |
  if $block < 72 then $positive[$block]
  elif $block % 2 == 0 then $mixed[$block-72]
  else $mixed[$block-72]|reverse end] | flatten |
if length != 264 or (unique|length) != 264 then error("duplicate or missing case") else .[] end
