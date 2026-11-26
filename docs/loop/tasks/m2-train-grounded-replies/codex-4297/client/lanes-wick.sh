# Shared selection, routing and grading for capture and the no-generation fixture.
read_gate() { jq -ce -L "$here" -f "$here/gate.jq" "$1"; }
lanes_from() { jq -ce '.lanes' "$1"; }
pairs_from() { jq -er '.pairs' "$1"; }

write_jobs() {
    local capture=$1 selected=$2 pairs=$3 artifact=$4 stock=$5
    jq -es -L "$here" --argjson lanes "$selected" --argjson pairs "$pairs" \
        'include "lanes"; checked_plan($lanes; $pairs) | true' "$capture/plan.jsonl" >/dev/null || return
    jq -rs --arg model "$artifact" --arg stock_model "$stock" --arg out "$capture" \
        -f "$here/jobs.jq" "$capture/plan.jsonl"
}

plan_capture() {
    local capture=$1 reference=$2 endpoints=$3 artifact=$4 stock=$5 selected pairs arm id wire
    selected=$(lanes_from "$capture/selection.json")
    pairs=$(pairs_from "$capture/selection.json")
    jq -ce -L "$here" --argjson selected "$selected" --slurpfile device "$endpoints" \
        -f "$here/plan.jq" "$reference" > "$capture/plan.jsonl"
    write_jobs "$capture" "$selected" "$pairs" "$artifact" "$stock" > "$capture/jobs.tsv"
    mkdir -p "$capture/stock/fresh/"{plain,trace,wire}
    while IFS= read -r arm; do mkdir -p "$capture/$arm/fresh/"{plain,trace,wire}; done < <(jq -r '.[]' <<< "$selected")
    while IFS=$'\t' read -r arm id wire; do
        printf '%s\n' "$wire" > "$capture/$arm/fresh/wire/$id.json"
    done < <(jq -r '[.arm,.id,(.wire|tojson)] | @tsv' "$capture/plan.jsonl")
}

collect_capture() {
    local capture=$1 reference=$2 endpoints=$3 artifact=$4 stock_artifact=$5 model_hash=$6 stock_hash=$7
    local selected pairs arm cohort _fold id _artifact _wire _field _plain lane raw code
    selected=$(lanes_from "$capture/selection.json")
    pairs=$(pairs_from "$capture/selection.json")
    cmp "$capture/jobs.tsv" <(write_jobs "$capture" "$selected" "$pairs" "$artifact" "$stock_artifact")
    while IFS=$'\t' read -r arm cohort _fold id _artifact _wire _field _plain; do
        lane="$capture/$arm/$cohort"
        raw="$lane/trace/$id"
        code=$(cat "$raw.exit")
        if [[ $code == 0 ]]; then
            jq -ce --arg id "$id" '{id:$id,exit:0,record:.}' "$raw.json"
        else
            jq -nc --arg id "$id" --argjson code "$code" '{id:$id,exit:$code,record:null}'
        fi >> "$lane/rows.jsonl"
    done < "$capture/jobs.tsv"
    mkdir "$capture/grade"
    while IFS= read -r arm; do
        jq -nce -L "$here" -L "$prior" --arg arm "$arm" --arg model "$model_hash" \
            --arg stock_model "$stock_hash" --arg codec_hash "$codec_sha" \
            --slurpfile reference "$reference" --slurpfile device "$endpoints" --slurpfile codec "$codec" \
            --slurpfile stock "$capture/stock/fresh/rows.jsonl" -f "$here/grade.jq" \
            "$capture/$arm/fresh/rows.jsonl" > "$capture/grade/$arm.jsonl"
    done < <(jq -r '.[]' <<< "$selected")
    jq -es -L "$here" --argjson selected "$selected" --argjson pair_count "$pairs" \
        --slurpfile plan "$capture/plan.jsonl" -f "$here/summary.jq" "$capture/grade/"*.jsonl > "$capture/grade/summary.json"
}

grade_tokens() {
    local capture=$1 output=$2 selected pairs
    selected=$(lanes_from "$capture/selection.json")
    pairs=$(pairs_from "$capture/selection.json")
    jq -ce -f "$here/audit-tokens.jq" "$capture/grade/"*.jsonl > "$output/tokens.jsonl"
    jq -es -L "$here" --argjson selected "$selected" --argjson pair_count "$pairs" \
        --slurpfile plan "$capture/plan.jsonl" -f "$here/tokens-summary.jq" "$output/tokens.jsonl" > "$output/summary.json"
}

check_differential() {
    local capture=$1 result=$2 selected pairs arm fold lane
    selected=$(lanes_from "$capture/selection.json") || return
    pairs=$(pairs_from "$capture/selection.json") || return
    cmp <(jq -r '.[]+"-fresh"' <<< "$selected" | sort) \
        <(find "$result" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) || return
    while IFS= read -r arm; do
        lane="$result/$arm-fresh"
        cmp "$lane/cases.tsv" <(awk -F '\t' -v arm="$arm" '$1==arm{print $3 FS $4 FS $5}' "$capture/jobs.tsv") || return
        cmp <(awk -F '\t' '{print "fold-"$1}' "$lane/cases.tsv" | sort -u) \
            <(find "$lane" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) || return
        while IFS= read -r fold; do
            jq -e --argjson count "$(awk -F '\t' -v fold="$fold" '$1==fold{n++}END{print n+0}' "$lane/cases.tsv")" \
                '.cases==$count and .turns+.unissued==2*.cases' "$lane/fold-$fold/receipt.json" >/dev/null || return
        done < <(cut -f1 "$lane/cases.tsv" | sort -u)
    done < <(jq -r '.[]' <<< "$selected")
    jq -e --argjson count "$((pairs * 2 * $(jq length <<< "$selected")))" '.cases==$count' "$result/summary.json" >/dev/null
}
