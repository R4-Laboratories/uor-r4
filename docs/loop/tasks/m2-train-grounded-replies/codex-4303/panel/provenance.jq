def corpus_signatures:
    {
        families:(map(.family)|unique),
        facts:(map(.fact)|unique),
        fact_values:([.[].fact_spans[].value]|unique),
        scope_clauses:([.[].scope_clauses[]]|unique),
        noun_surfaces:([.[] | .turns[1] as $text | .gold_binding |
            (.affirmed,.excluded,.referenced) | $text[.start:.end]]|unique),
        reply_clauses:([.[] | .turns[1] as $text |
            (.role_phrase,.pending_phrase) | $text[.start:.end]]|unique),
        prompts:(map(.id)|unique),
        tapes:(map(.structural_prompt_sha256)|unique)
    };

. as $references |
($prior|add) as $spent |
($spent|corpus_signatures) as $recorded |
($references|corpus_signatures) as $new |
($new | to_entries | map(.key as $key |
    .value |= map(. as $value | select($recorded[$key] | index($value)))) | from_entries) as $overlaps |
if ($spent|length) != $expected or any($overlaps[]; length > 0) then
    error("the source corpus is incomplete or has reused identities: \($overlaps)")
else
    {
        comparison_rows:($spent|length),
        sealed_rows:($references|length),
        families:$new.families,
        facts:$new.facts,
        fact_values:$new.fact_values,
        noun_surfaces:$new.noun_surfaces,
        overlaps:$overlaps
    }
end
