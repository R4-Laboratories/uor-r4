def identities($rows; $with_rivals):
  [$rows[] | select(.kind=="opposing") | [.first[] | if $with_rivals then [.ordinal,.rival_id] else [.ordinal] end] | sort] | unique;
def overlap($old; $new):
  ($old + $new | unique | length) as $union | ($old|length) + ($new|length) - $union;
def describe($arm):
  {pairs: ($arm|length),
   shared_row_pairs: overlap(identities($snapped;false);identities($arm;false)),
   shared_row_and_rival_pairs: overlap(identities($snapped;true);identities($arm;true))};
{snapped:describe($snapped), transported:describe($transported), raw:describe($raw), swapped:describe($swapped)}
