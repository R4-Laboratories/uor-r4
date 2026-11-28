# Finite test of representability before fitting

The result of interest is one signed integer table shared across both questions and all eligible original sites. A separate table per prompt, or a table selected using the evaluation answers, supplies only a capacity ceiling. No solver witness is a prediction or trained state.

At position p write b(p) for the frozen native site score and x(p,q) for its normalized 32 signed coordinates expressed in question q's canonical frame. NoRead has the frozen threshold and a zero feature vector. For each required comparison (a,b) impose

    sum over occupied axes of w[i] * (x(a,q)[i] - x(b,q)[i])
        >= margin + frozen_score(b) - frozen_score(a)

The weights are integers within ±4096. A tie on a lexical read can produce an abstention in the ordinary client. The test requires positive evidence for NoRead, i.e., a strict margin against wrong sites. Test margins 1 and 8. With offsets removed, set *both* frozen source and NoRead scores to zero. All active, fixed, swapped and question-removed arms keep source positions and offsets from the current prompt. In the zero-weight baseline each compared score is precisely the chosen offset.

Check the old nine and new pairs separately. For each prompt, compare the required target against its named counterpart, then every eligible wrong site and NoRead. On an unsupported prompt compare NoRead against its named counterpart, then against every eligible site. For each pair combine constraints across its two members, and for each panel combine all member constraints. A failure of any subset certifies failure of the complete shared system. Do not infer pooled feasibility by adding individually successful pairs.

For a constructive pass, check every raw inequality and the bounds using the returned integer weights. An impossibility certificate can use nonnegative integer multiples of valid rows: if the resulting lower bound exceeds 4096 times the sum of the absolute combined coefficients, the constraints conflict. Record cases and original sites and verify the contradiction in integer arithmetic. If neither a verified witness nor a verified contradiction is available, preserve the outcome as unknown. A solver timeout is not a negative. A signed permutation preserves the word box, so active and fixed frames must have identical feasibility when each individual prompt gets its own table.

Use a GPU fit only after an actual conditioned table is feasible across the independently graded fresh panel with all rivals and the same-byte fixed-query control establishes an advantage. Keep outcomes for the inherited checkpoint and the zero-offset diagnostic distinct. The longer-context count baseline uses its own measured artifact budget under #4063; the 64-byte comparison here applies only to the proposed added reader state.
