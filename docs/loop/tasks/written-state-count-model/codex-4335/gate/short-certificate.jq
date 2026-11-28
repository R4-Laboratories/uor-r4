# Exhaust the zero vectors and oppositely directed pairs. The coefficients
# are differences of signed unit coordinates, so the integer products are exact.
def gcd($a; $b): if $b == 0 then $a else gcd($b; $a % $b) end;
def axis_gcd: reduce .[] as $x (0; gcd(.; ($x | if . < 0 then -. else . end)));
def term($inequality; $multiplier):
    {inequality: $inequality, multiplier: ($multiplier | tostring)};
def proof($terms):
    {status: "impossible", box_maximum: "0",
     weighted_minimum: ($terms | map((.multiplier | tonumber) * .inequality.minimum) | add | tostring),
     terms: $terms};

($keys | INDEX(.case_id)) as $index
| [$rows[] as $row
   | $index[$row.case_id].target as $site
   | select($site != null)
   | ($row.candidates | map(select(.pos == $site.pos and .id == $site.id))) as $winners
   | if ($winners | length) != 1 then error("gold address missing or repeated") else . end
   | $winners[0] as $winner
   | $row.candidates[] | select(.pos != $winner.pos)
   | . as $loser
   | {case_id: $row.case_id, winner: $winner.pos, loser: $loser.pos, minimum: 1,
      coefficients: [range(32) as $i | $winner.real_signed[$i] - $loser.real_signed[$i]]}
  ] as $inequalities
| ($inequalities | map({expr: ., divisor: (.coefficients | axis_gcd)})) as $rows
| [$rows[] | select(.divisor == 0)] as $aliases
| if ($aliases | length) > 0 then
    {comparisons: ($rows | length), alias_comparisons: ($aliases | length),
     minimum_terms: 1, proof: proof([term($aliases[0].expr; 1)])}
  else
    ($rows | map(. + {primitive: (.divisor as $d | .expr.coefficients | map(. / $d))})) as $normal
    | ($normal | INDEX(.primitive | tojson)) as $lookup
    | first($normal[] as $left
            | $lookup[$left.primitive | map(if . == 0 then 0 else -. end) | tojson] as $right
            | select($right != null) | [$left, $right]) // null
    | if . == null then
        {comparisons: ($rows | length), alias_comparisons: 0, minimum_terms: null,
         excluded_through_terms: 2, proof: null}
      else
        {comparisons: ($rows | length), alias_comparisons: 0,
         minimum_terms: 2, proof: proof([term(.[0].expr; .[1].divisor), term(.[1].expr; .[0].divisor)])}
      end
  end
