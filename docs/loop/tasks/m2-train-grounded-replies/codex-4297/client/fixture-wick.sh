#!/usr/bin/env bash
# Use sealed historical traces to exercise the production planner and grader.
set -euo pipefail
umask 077
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
source "$here/lanes-wick.sh"
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-all-outranker-interval"
client="$study/client"
donor="$runs/20261125-codex-bound-instruction-state"
prior="$donor/ordinary-recipe-final-v1"
codec="$donor/custody-v1/signed/tokenizer.json"
old="$runs/20261125-codex-governing-source-pairs/held/primary"
legacy="$client/fixture-v1"
historical_native="$old/capture-baseline-v1/baseline/fresh/rows.jsonl"
historical_exchanged="$old/capture-controls-v1/scope-swap/fresh/rows.jsonl"
model=06a7d1f197870a193699b75dd58ce6269455da3047da87f032de1d45600ccfa2
stock=2e16a8ea27b316cce01dac5744c6b02da38e0ee6c0abc3fcb610e381b1c85302
codec_sha=a9fb13f70c69ffe5afed40c8251a8e702adf0b00732d35ccff9457a3309ee377
gate_template="$study/grade-v2/summary.json"
out="$client/${1:?new fixture version}"
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$client") == zfs &&
    $here == "$client"/methods-v* && $out =~ ^$client/fixture-v[1-9][0-9]*$ ]]
verify() { (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS); }
sha() { sha256sum "$1" | cut -d' ' -f1; }
verify "$here"; verify "$legacy"; verify "${gate_template%/*}"
grep -Fxq "$(sha "$historical_native")  ./baseline/fresh/rows.jsonl" "$old/capture-baseline-v1/SHA256SUMS"
grep -Fxq "$(sha "$historical_exchanged")  ./scope-swap/fresh/rows.jsonl" "$old/capture-controls-v1/SHA256SUMS"
mkdir "$out"
sha256sum "$here/SHA256SUMS" "$legacy/SHA256SUMS" "${gate_template%/*}/SHA256SUMS" \
    "$historical_native" "$historical_exchanged" "$codec" "$prior/"{raw,score}.jq > "$out/inputs.sha256"
reference="$legacy/reference.json"
jq -nc --slurpfile refs "$reference" '
    INDEX($refs[0][];.id) as $wanted | inputs | select($wanted[.id]!=null)' "$historical_native" > "$out/native.jsonl"
jq -nc --slurpfile refs "$reference" '
    INDEX($refs[0][];.id) as $wanted | inputs | select($wanted[.id]!=null)' "$historical_exchanged" > "$out/exchanged.jsonl"
jq -nce --slurpfile native "$out/native.jsonl" --slurpfile exchanged "$out/exchanged.jsonl" '
    INDEX($native[];.id) as $records |
    INDEX($exchanged[];.id) as $swapped |
    inputs | ($records[.case_id].record.turns[1].role_action_trace.bound_instruction.selections[0]) as $read |
    ($swapped[.case_id].record.turns[1].role_action_trace.bound_instruction.binding) as $other |
    .arms.native=(if $read.status=="chosen" then {
         selection:($read.witnesses|map(.bytes|map_values(.-1))),
         tokens:($read.witnesses|map(.tokens)),bank_cost:$read.distance
      } else {selection:null,tokens:null} end) |
    .arms.baseline=.arms.correct_link |
    .arms.exchanged_link=([$other.affirmed,$other.excluded,$other.referenced]|
        {selection:map(.bytes|map_values(.-1)),tokens:map(.tokens)})' "$legacy/device.jsonl" > "$out/device.jsonl"
endpoints="$out/device.jsonl"
cat "$legacy/"{take,defer}.jsonl > "$out/supplied.jsonl"

make_gate() {
    local passing=$1 dest=$2
    jq -ce --argjson passing "$passing" --arg ref "$(sha "$reference")" --arg endpoints "$(sha "$endpoints")" '
       .pairs=1 | .reference_sha256=$ref | .endpoints_sha256=$endpoints |
       .passing=$passing | .pass=($passing|length>0) | .gate.at_least_one_trained_arm=.pass |
       .arms|=with_entries(.key as $arm |
          (.value|={arm:$arm,pairs:1,reference_sha256:$ref,endpoints_sha256:$endpoints,
                    pass:($arm|IN($passing[])),gate:(.gate|map_values($arm|IN($passing[])))}))' \
        "$gate_template" > "$dest"
}

mock_differential() {
    local capture=$1 result=$2 arm count lane
    mkdir "$result"
    while IFS= read -r arm; do
        lane="$result/$arm-fresh"; mkdir "$lane" "$lane/fold-8"
        awk -F '\t' -v arm="$arm" '$1==arm{print $3 FS $4 FS $5}' "$capture/jobs.tsv" > "$lane/cases.tsv"
        jq -n --argjson cases "$(wc -l < "$lane/cases.tsv")" '{cases:$cases,turns:($cases*2),unissued:0}' > "$lane/fold-8/receipt.json"
    done < <(jq -r '.lanes[]' "$capture/selection.json")
    count=$(awk -F '\t' '$1!="stock"{n++}END{print n+0}' "$capture/jobs.tsv")
    jq -n --argjson cases "$count" '{cases:$cases}' > "$result/summary.json"
}

exercise() {
    local name=$1 allowed=$2 capture="$out/$1" arm id observed raw _cohort _fold _rest
    mkdir "$capture" "$capture/tokens"
    make_gate "$allowed" "$capture/gate.json"
    read_gate "$capture/gate.json" > "$capture/selection.json"
    jq -e '.pass and .pairs==1' "$capture/selection.json" >/dev/null
    plan_capture "$capture" "$reference" "$endpoints" fixture-study fixture-stock
    while IFS=$'\t' read -r arm _cohort _fold id _rest; do
        case "$arm" in
            stock) observed="$legacy/stock.jsonl" ;;
            native) observed="$out/native.jsonl" ;;
            exchanged_link) observed="$out/exchanged.jsonl" ;;
            *) observed="$out/supplied.jsonl" ;;
        esac
        raw="$capture/$arm/fresh/trace/$id"
        jq -ce --arg id "$id" 'select(.id==$id)|.record' "$observed" > "$raw.json"
        printf '0\n' > "$raw.exit"
    done < "$capture/jobs.tsv"
    collect_capture "$capture" "$reference" "$endpoints" fixture-study fixture-stock "$model" "$stock"
    grade_tokens "$capture" "$capture/tokens"
    mock_differential "$capture" "$capture/writer-ledger"
    check_differential "$capture" "$capture/writer-ledger"

    jq -e --slurpfile plan "$capture/selection.json" '
        .source_identity_disagreements==[] and (.arms|map(.arm))==$plan[0].lanes and
        all(.arms[];.all.endpoints==2 and
            (if .arm=="correct_link" then .paired_take_eos_and_defer_stock==1
             elif .arm=="exchanged_link" then .all.gold_scope_exact==0 else true end))' "$capture/grade/summary.json" >/dev/null

    awk -F '\t' '$2!="fresh"||$8!=1{exit 1} $1=="stock"{n++} END{if(n!=2)exit 1}' "$capture/jobs.tsv"
    jq -nce --arg name "$name" --slurpfile selection "$capture/selection.json" \
        --slurpfile plan "$capture/plan.jsonl" --slurpfile observed "$capture/tokens/tokens.jsonl" '
        {scenario:$name,passing:$selection[0].passing,lanes:($selection[0].lanes|length),stock_cases:2,
         planned_cases:($plan|length),graded_token_cases:($observed|length)}' >> "$out/positive-cases.jsonl"
}
exercise singleton '["ordered-all"]'
exercise mixed '["strongest","all"]'
exercise all '["strongest","all","ordered-all"]'

# Six distinct synthetic case IDs per lane exercise the first-four boundary.
jq -sce '[range(3) as $batch | .[] | .id=(($batch|tostring)+.id[1:])|.pair_id=(.pair_id+($batch|tostring))][]' \
    "$out/singleton/plan.jsonl" > "$out/high-count-plan.jsonl"
mkdir "$out/high-count"; cp "$out/high-count-plan.jsonl" "$out/high-count/plan.jsonl"
write_jobs "$out/high-count" "$(lanes_from "$out/singleton/selection.json")" 3 fixture-study fixture-stock > "$out/high-count/jobs.tsv"
awk -F '\t' '{++count[$1]; if($8!=(count[$1]<=4)) exit 1}
    END{if(length(count)!=6)exit 1; for(arm in count)if(count[arm]!=6)exit 1}' "$out/high-count/jobs.tsv"

make_gate '[]' "$out/refused.json"
read_gate "$out/refused.json" > "$out/refused-selection.json"
jq -e '.pass==false and .passing==[]' "$out/refused-selection.json" >/dev/null
base="$out/singleton/gate.json"
for bad in duplicate unknown forged missing incomplete wrong_hash extra_arm wrong_gate_flag; do
    case "$bad" in
        duplicate) edit='.passing+=["ordered-all"]' ;;
        unknown) edit='.passing=["foreign"]' ;;
        forged) edit='.arms.strongest.pass=true|.passing+=["strongest"]' ;;
        missing) edit='.passing=[]' ;;
        incomplete) edit='del(.arms.all.gate.train_reduced_loss)' ;;
        wrong_hash) edit='.arms.all.reference_sha256=("b"*64)' ;;
        extra_arm) edit='.arms.foreign=.arms.all' ;;
        wrong_gate_flag) edit='.gate.at_least_one_trained_arm=false' ;;
    esac
    jq -c "$edit" "$base" > "$out/$bad.json"
    if read_gate "$out/$bad.json" > "$out/$bad-selection.json" 2> "$out/$bad.log"; then
        echo "forged gate was accepted: $bad" >&2; exit 1
    fi
done
cap="$out/singleton"
selected=$(lanes_from "$cap/selection.json")
for bad in unexpected_grade missing_case duplicate_case swapped_identity; do
    case "$bad" in
        unexpected_grade) edit='.[0].arm="strongest"' ;;
        missing_case) edit='.[1:]' ;;
        duplicate_case) edit='.[0]=.[1]' ;;
        swapped_identity) edit='.[0].admission=(if .[0].admission=="take" then "defer" else "take" end)' ;;
    esac
    jq -sc "$edit|.[]" "$cap/tokens/tokens.jsonl" > "$out/$bad.jsonl"
    if jq -es -L "$here" --argjson selected "$selected" --argjson pair_count 1 --slurpfile plan "$cap/plan.jsonl" \
        -f "$here/tokens-summary.jq" "$out/$bad.jsonl" > "$out/$bad-summary.json" 2> "$out/$bad.log"; then
        echo "incorrect token ledger accepted: $bad" >&2; exit 1
    fi
done
cp -a "$cap/writer-ledger" "$out/unexpected-differential"
mkdir "$out/unexpected-differential/stock-fresh"
if (check_differential "$cap" "$out/unexpected-differential") >/dev/null 2>&1; then
    echo "unexpected stock lane admitted to writer differential" >&2; exit 1
fi
jq -n --slurpfile cases "$out/positive-cases.jsonl" '
    {schema:"uor-r4.codex-4297.client-selection-fixture/1",pass:true,cases:$cases,
     refused_empty:true,forged_gate_checks:8,invalid_token_ledgers_rejected:4,
     per_lane_first_four_checked:true,differential_extra_lane_rejected:true,
     source:"sealed historical ordinary records with correct supplied traces reused for synthetic trained and baseline lanes",
     generated_replies:0,gpu_runs:0}' > "$out/receipt.json"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
cat "$out/receipt.json"
