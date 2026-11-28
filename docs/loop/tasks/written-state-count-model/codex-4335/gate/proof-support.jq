# The prime is small enough that all products and differences are exact.
# Rank modulo the prime is a lower bound on rank over the rationals.
def residue: ((. % 257) + 257) % 257;
def eliminate($basis):
    reduce $basis[] as $b
        (.;
         .[$b.pivot] as $factor
         | if $factor == 0 then .
           else [$b.axis[$b.pivot] as $pivot | . as $axis
                 | range(32) as $i | ($pivot * $axis[$i] - $factor * $b.axis[$i] | residue)]
           end);
def basis:
    reduce .[] as $axis
        ([]; . as $known | ($axis | map(residue) | eliminate($known)) as $remainder
         | first(range(32) | select($remainder[.] != 0)) // null
         | if . == null then $known else $known + [{pivot: ., axis: $remainder}] end);

select(.level=="panel_positive_only" and .scope=="supported_gold_vs_sites"
       and .arm=="actual" and .offsets=="zero" and .margin==1)
| if .verdict.status != "impossible" or .verdict.box_maximum != "0" then
    {terms: null, rank_mod_257: null, support_is_irreducible: null}
  else
    .verdict.terms as $terms
    | ($terms | map(.inequality.coefficients) | basis | length) as $rank
    | {terms: ($terms | length), rank_mod_257: $rank,
       support_is_irreducible: ($rank + 1 == ($terms | length))}
  end
