($base[0].scoped) as $b | ($candidate[0].scoped) as $c |
if ($b.pairs!=224 or $c.pairs!=224 or $base[0].axes["direct-count-control"].pairs!=32 or $candidate[0].axes["direct-count-control"].pairs!=32 or $protect_base[0].original_exact_pairs!=50 or $protect_base[0].retained_exact_pairs!=50 or $protect_candidate[0].original_exact_pairs!=50)
then error("invalid scoped or protection census") else . end |
{baseline:{pairs:$b.pairs,directed:$b.directed,exact:$b.exact,wordings:$b.exact_types,protect:$protect_base[0].retained_exact_pairs},
 candidate:{pairs:$c.pairs,directed:$c.directed,exact:$c.exact,wordings:$c.exact_types,protect:$protect_candidate[0].retained_exact_pairs}} |
. + {checks:{directed:(.candidate.directed>.baseline.directed),exact:(.candidate.exact>.baseline.exact),wordings:(.candidate.wordings>.baseline.wordings),protected:(.candidate.protect==50)}} |
. + {eligible:all(.checks[];.)}
