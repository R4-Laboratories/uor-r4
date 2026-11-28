def count(f): map(select(f)) | length;
def sumfield($field): map(.[$field]) | add // 0;
def assess($census; $target; $mode):
    [$census.traces[] | select($mode=="relaxed" or .preceding_delimiter)] as $traces
    | [$traces[].candidates[] | select(.eligible) | .range[0]] | unique as $starts
    | [$traces[].selection.source | select(.!=null)] | unique as $chosen
    | [$traces[].lexical_spans[][0]] | unique as $lexical
    | {
        assertions: ($traces|length),
        fenced_assertions: ($traces|count(.fence!=null)),
        candidate_sites: ($starts|length),
        selected_sites: ($chosen|length),
        gold_in_fenced_interval: ($target!=null and any($traces[];
            .fence != null and .fact[1] <= $target and $target < .fence)),
        gold_in_lexical_sites: ($target!=null and ($lexical|index($target)!=null)),
        gold_candidate: ($target!=null and ($starts|index($target)!=null)),
        gold_selected: ($target!=null and ($chosen|index($target)!=null)),
        wrong_candidates: ($starts|count(.!=$target)),
        wrong_selections: ($chosen|count(.!=$target))
    };
def report($rows; $mode):
    ($rows | map(select(.supported))) as $supported
    | ($rows | map(select(.supported|not))) as $unsupported
    | {
        asserted_rows: ($rows|count(.observed.assertions>0)),
        fenced_assertions: ($rows|map(.observed)|sumfield("fenced_assertions")),
        rows_with_candidates: ($rows|count(.observed.candidate_sites>0)),
        rows_with_selections: ($rows|count(.observed.selected_sites>0)),
        supported_gold_in_fenced_interval: ($supported|count(.observed.gold_in_fenced_interval)),
        supported_gold_in_lexical_sites: ($supported|count(.observed.gold_in_lexical_sites)),
        supported_gold_candidate: ($supported|count(.observed.gold_candidate)),
        supported_gold_selected: ($supported|count(.observed.gold_selected)),
        supported_with_wrong_candidates: ($supported|count(.observed.wrong_candidates>0)),
        supported_wrong_candidate_sites: ($supported|map(.observed)|sumfield("wrong_candidates")),
        supported_with_wrong_selections: ($supported|count(.observed.wrong_selections>0)),
        supported_wrong_selected_sites: ($supported|map(.observed)|sumfield("wrong_selections")),
        unsupported_asserted_rows: ($unsupported|count(.observed.assertions>0)),
        unsupported_with_candidates: ($unsupported|count(.observed.candidate_sites>0)),
        unsupported_candidate_sites: ($unsupported|map(.observed)|sumfield("candidate_sites")),
        unsupported_with_selections: ($unsupported|count(.observed.selected_sites>0)),
        unsupported_selected_sites: ($unsupported|map(.observed)|sumfield("selected_sites"))
    };

if (map(.case_id)|sort) != ($truth|map(.case_id)|sort)
   or (map(.case_id)|unique|length)!=($truth|length)
then error("capture and grade case sets differ") else . end
| . as $rows
| ($truth|INDEX(.case_id)) as $keys
| {
    rows: length,
    supported: ($truth|count(.target != null)),
    unsupported: ($truth|count(.target == null)),
    cuts: (["separator","tail32"] | map(. as $cut
        | ($rows|map(.[$cut])) as $census
        | {
            cut: $cut,
            exact_id_slices: ($census|sumfield("exact_id_slices")),
            start_losses: ($census|sumfield("start_losses")),
            word_bound_losses: ($census|sumfield("word_bound_losses")),
            binder_gap_losses: ($census|sumfield("binder_gap_losses")),
            genuine_word_pairs: ($census|sumfield("native_assertions") + sumfield("delimiter_only_assertions")),
            native_assertions: ($census|sumfield("native_assertions")),
            delimiter_only_assertions: ($census|sumfield("delimiter_only_assertions")),
            modes: (["native","relaxed"] | map(. as $mode
                | [$rows[] as $row | $keys[$row.case_id].target as $target
                   | {supported: ($target!=null),
                      observed: assess($row[$cut]; $target.pos; $mode)}]
                | report(.; $mode) + {mode:$mode}))
        }))
}
