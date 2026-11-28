($tok[0].model.vocab | to_entries | map({key:(.value|tostring),value:.key}) | from_entries) as $v
| reduce inputs as $r ({};
  if $r.is_impossible then .published_impossible += 1 else
   $r.gold_source as $s | ($s.source_ids|length) as $n | $s.first_position as $p |
   ($v[$r.prefix[$p+$n]|tostring] // "" | ltrimstr("Ġ")) as $next |
   (if $s.response_lexical_offset == 0 then "immediate" elif $s.response_lexical_offset == 1 and $s.response_ids[0] == 223 then "after_space" else error("unexpected lead-in") end) as $phase |
   (if $next|test("^[A-Za-z0-9]") then "ascii_alphanumeric" elif $next|test("^[[:punct:]]") then "ascii_punctuation" else "other" end) as $class |
   if (($s.whole_span_ids|not) or $r.prefix[$p:$p+$n]!=$s.source_ids or $s.response_ids[$s.response_lexical_offset:]!=$s.source_ids or $p<$r.user_bounds[0] or $p+$n>=$r.user_bounds[1] or ($r.prefix|length)+($s.response_ids|length)+1>192) then error("source mismatch") else
     .supported += 1 | if $n<=8 then .reachable += 1 | .entry_phases[$phase] += 1 | .next_source[$class] += 1 | if $n==8 then .at_cap += 1 else . end else .over_cap += 1 end
   end end)
