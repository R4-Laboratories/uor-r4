($capture | INDEX(.case_id)) as $cases
| [.[] | select(.level == "panel_positive_only") |
    {
        arm,
        offsets,
        margin,
        scope,
        requirement_count,
        actual_witness_pass,
        status: .verdict.status,
        proof: (
            if .verdict.status == "impossible" then
                {
                    terms: (.verdict.terms | length),
                    cases: ([.verdict.terms[].inequality.case_id] | unique | length),
                    passage_rivals: ([.verdict.terms[].inequality
                        | select(.loser != null and .loser < $cases[.case_id].question_start)] | length),
                    question_rivals: ([.verdict.terms[].inequality
                        | select(.loser != null and .loser >= $cases[.case_id].question_start)] | length),
                    noread_rivals: ([.verdict.terms[].inequality | select(.loser == null)] | length),
                    weighted_minimum: .verdict.weighted_minimum,
                    box_maximum: .verdict.box_maximum
                }
            else null end
        )
    }
]
