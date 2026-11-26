def require($ok; $problem): if $ok then . else error($problem) end;

def sources:
    {families:(map(.family)|unique), facts:(map(.fact)|unique),
     fact_values:([.[].fact_spans[].value]|unique),
     scope_clauses:([.[].scope_clauses[]]|unique),
     noun_surfaces:([.[] | .turns[1] as $text | .gold_binding |
        (.affirmed,.excluded,.referenced) | $text[.start:.end]]|unique),
     reply_clauses:([.[] | .turns[1] as $text |
        (.role_phrase,.pending_phrase) | $text[.start:.end]]|unique),
     prompts:(map(.id)|unique), tapes:(map(.structural_prompt_sha256)|unique)};

def overlap($a; $b): [$a[] | . as $value | select($b|index($value)!=null)];

sources as $fresh_sources | ($prior|add|sources) as $spent |
{
    fresh_families:$fresh_sources.families,
    described_facts:$fresh_sources.facts,
    fact_values:$fresh_sources.fact_values,
    noun_surfaces:$fresh_sources.noun_surfaces,
    comparison_rows:($prior|add|length),
    overlaps:($fresh_sources | to_entries | map(.key as $k |
        .value |= overlap(.; $spent[$k])) | from_entries)
} |
require(.comparison_rows == 1032 and all(.overlaps[]; length == 0);
    "fresh wording, facts, nouns, clause or tape overlaps, or a prior cohort is missing")
