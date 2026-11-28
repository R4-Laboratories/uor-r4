def rotate:
    if length == 32 then .[8:] + .[:8] else error("expected four eight-axis banks") end;

if $kind == "features" then
    .candidates[] |= (.real_signed |= rotate | .fixed_signed |= rotate
        | .swapped_signed |= rotate | .no_question_signed |= rotate)
elif $kind == "verdicts" then
    if .verdict.status == "feasible" then .verdict.words |= rotate
    elif .verdict.status == "impossible" then .verdict.terms[].inequality.coefficients |= rotate
    else error("tail32 verdict must carry a witness or certificate")
    end
else error("unknown kind")
end
