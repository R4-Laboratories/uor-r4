#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/archive-lib.sh"
[[ $# == 3 ]]
root=$(realpath -e "${1:?extracted archive}")
layout=$(realpath -e "${2:?archived layout}")
out=${3:?new replay output}
live=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median
[[ $layout == "$root"/storage/archive-v*/layout.json ]]
jq -e '.schema=="uor-r4.codex-4299.archive-layout/1"' "$layout" >/dev/null

from_layout() {
    local name
    name=$(jq -er --arg key "$1" --arg re "$2" '.[$key] | select(type=="string" and test($re))' "$layout")
    safe_member "$name"
    printf '%s/%s\n' "$root" "$name"
}
bound() {
    local manifest=$1 expected=$2
    grep -Fxq "$(digest "$expected")  $live/${expected#"$root"/}" "$manifest"
}
cmp_file() {
    cmp "$1" "$2" || { printf 'Cold replay disagrees with %s\n' "${2#"$root"/}" >&2; return 1; }
    printf '%s\n' "${2#"$root"/}" >> "$out/compared.txt"
}

methods=$(from_layout methods '^methods-v[1-9][0-9]*$')
panel=$(from_layout panel '^panel/panel-v[1-9][0-9]*$')
train=$(from_layout train '^train-v[1-9][0-9]*$')
held=$(from_layout held '^held-v[1-9][0-9]*$')
grade=$(from_layout grade '^grade-v[1-9][0-9]*$')
release=$(from_layout release '^release-v[1-9][0-9]*$')
permit=$(from_layout permit '^permit-v[1-9][0-9]*$')
train_audit=$(from_layout train_audit '^audit/train-v[1-9][0-9]*$')
held_audit=$(from_layout held_audit '^audit/held-v[1-9][0-9]*$')
train_build=$(from_layout train_build '^audit/build-v[1-9][0-9]*/SHA256SUMS$')
held_build=$(from_layout held_build '^audit/build-v[1-9][0-9]*/SHA256SUMS$')
adapter=$(from_layout adapter '^client/methods-v[1-9][0-9]*$')
fixture=$(from_layout client_fixture '^client/fixture-v[1-9][0-9]*$')
cv=$(jq -er '.writer_version' "$layout"); version "$cv"
historical="${layout%/*}/ancestry/historical"
for dir in "$train" "$held" "$grade" "$panel" "$methods" "$release" "$permit" \
    "$train_audit" "$held_audit" "${train_build%/*}" "${held_build%/*}" "$adapter" "$fixture"; do verify "$dir"; done
(cd "$panel" && sha256sum --strict --quiet -c blind.sha256 labels.sha256)
for file in "$panel/SHA256SUMS" "$methods/SHA256SUMS" "$adapter/SHA256SUMS"; do bound "$release/inputs.sha256" "$file"; done
bound "$train/inputs.sha256" "$release/inputs.sha256"
for file in "$train/selection.json" "$train/selection.sha256" "$train/SHA256SUMS" "$methods/SHA256SUMS" \
    "$permit/verified.json" "$train_audit/SHA256SUMS"; do
    bound "$permit/inputs.sha256" "$file"
done
jq -e '.pass==true' "$permit/verified.json" >/dev/null
for file in "$permit/inputs.sha256" "$train/SHA256SUMS" "$panel/SHA256SUMS" "$methods/SHA256SUMS"; do
    bound "$held/inputs.sha256" "$file"
done
for file in "$train/SHA256SUMS" "$held/SHA256SUMS" "$panel/SHA256SUMS" "$methods/SHA256SUMS"; do
    bound "$grade/inputs.sha256" "$file"
done
bound "$train_audit/inputs.sha256" "$train_build"
bound "$held_audit/inputs.sha256" "$held_build"
for file in "$train/SHA256SUMS" "$held/SHA256SUMS" "$grade/SHA256SUMS" "$panel/SHA256SUMS" "$train_audit/SHA256SUMS"; do
    bound "$held_audit/inputs.sha256" "$file"
done

source "$methods/common-wick.sh"
inputs="$methods/inputs"
mkdir "$out" "$out/joined" "$out/states"
cmp_file "$inputs/premise.json" "$historical/premise.json"

bash "$methods/reduce-train-wick.sh" "$train" "$panel" "$out/train"
cmp_file "$out/train/selection.json" "$train/selection.json"
for arm in endpoint ordered; do
    for id in $(states); do
        cmp_file "$out/train/$arm/$id.pairs.json" "$train/$arm/$id.pairs.json"
    done
    for member in candidates.jsonl selected.json; do
        cmp_file "$out/train/$arm/$member" "$train/$arm/$member"
    done
    decision="$out/train/$arm/selected.json"
    selected=$(jq -er '.chosen | select(type=="string")' "$decision")
    elected_mode=$(jq -er --arg arm "$arm" '
        if .arm==$arm and .qualified==true then $arm
        elif .arm==$arm and .qualified==false and .chosen=="base" then "endpoint"
        else error("invalid TRAIN deployment decision") end' "$decision")
    for kind in selected full; do
        name=$arm; mode=$elected_mode; expected="$train/device/$arm/$selected.state"
        if [[ $kind == full ]]; then
            name=${arm}_full; mode=$arm; expected="$train/device/$arm/full.state"
        elif jq -e '.qualified==false' "$decision" >/dev/null; then
            expected="$inputs/protected.state"
        fi
        cmp_file "$expected" "$train/states/$name.state"
        "$inputs/bundle" unpack "$train/states/$name.bridge" "$out/states/$name.state" > "$out/states/$name.mode"
        if ! cmp -s <(printf '%s\n' "$mode") "$out/states/$name.mode"; then
            printf 'Cold replay expected %s geometry for %s\n' "$mode" "$name" >&2
            exit 1
        fi
        cmp_file "$out/states/$name.mode" "$held/unpacked/$name.mode"
        cmp_file "$out/states/$name.state" "$held/unpacked/$name.state"
        cmp_file "$out/states/$name.state" "$train/states/$name.state"
    done
done
"${train_build%/*}/median-audit" train "$inputs/primary.words" "$inputs/protected.state" "$panel" \
    "$historical" "$train" "$out/train-audit.json"
cmp_file "$out/train-audit.json" "$train_audit/readback.json"

arms=(native unfitted gap ordered_gap baseline ordered_base far_only endpoint ordered endpoint_full ordered_full correct_link exchanged_link)
for domain in held contrast; do
    p="$panel/$domain"; d="$held/$domain"; j="$out/joined/$domain"
    mkdir "$j"
    for arm in "${arms[@]}"; do
        kind=learned
        case $arm in
            native|unfitted|correct_link|exchanged_link) kind=$arm ;;
            gap|ordered_gap) kind=governor_gap ;;
        esac
        join_rows "$domain" "$p" "$kind" "$d/$arm.raw.jsonl" > "$j/$arm.rows.jsonl"
        cmp_file "$j/$arm.rows.jsonl" "$d/$arm.rows.jsonl"
        pair_rows "$domain" < "$j/$arm.rows.jsonl" > "$j/$arm.pairs.json"
        cmp_file "$j/$arm.pairs.json" "$d/$arm.pairs.json"
        case $arm in
            endpoint|endpoint_full|ordered|ordered_full)
                case $(cat "$out/states/$arm.mode") in
                    endpoint) tape="$p/blind/text.bin"; gap=gap ;;
                    ordered) tape="$p/ordered/text.bin"; gap=ordered_gap ;;
                    *) printf 'Invalid geometry tag for %s\n' "$arm" >&2; exit 1 ;;
                esac
                "$inputs/textlinks-trace-audit" "$tape" "$inputs/primary.words" "$out/states/$arm.state" \
                    "$d/$arm.raw.jsonl" "$d/$gap.raw.jsonl" > "$j/$arm.replayed.json"
                cmp_file "$j/$arm.replayed.json" "$d/$arm.replayed.json"
                ;;
        esac
    done
done
seal "$out/joined"
bash "$methods/reduce-held-wick.sh" "$train" "$out/joined" "$panel" "$out/grade"
grade_members() { (cd "$1" && find . -type f ! -name SHA256SUMS ! -name inputs.sha256 -printf '%P\n' | sort); }
cmp <(grade_members "$out/grade") <(grade_members "$grade")
while IFS= read -r member; do cmp_file "$out/grade/$member" "$grade/$member"; done < <(grade_members "$grade")
"${held_build%/*}/median-audit" held "$inputs/primary.words" "$inputs/protected.state" "$inputs/far_only.state" \
    "$panel" "$train" "$out/train-audit.json" "$held" "$out/grade" "$out/held-audit.json"
cmp_file "$out/held-audit.json" "$held_audit/readback.json"

here="$adapter"
source "$here/lanes-wick.sh"
read_gate "$out/grade/summary.json" > "$out/admission.json"
jq -e '.pass and .generated==0 and .false_gate_refused and .forged_gate_rejected' "$fixture/receipt.json" >/dev/null
for arm in endpoint ordered both; do
    mkdir "$out/fixture-$arm"
    read_gate "$fixture/$arm/gate.json" > "$out/fixture-$arm/selection.json"
    cmp_file "$out/fixture-$arm/selection.json" "$fixture/$arm/selection.json"
    grade_tokens "$fixture/$arm" "$out/fixture-$arm"
    cmp_file "$out/fixture-$arm/tokens.jsonl" "$fixture/$arm/tokens/tokens.jsonl"
    cmp_file "$out/fixture-$arm/summary.json" "$fixture/$arm/tokens/summary.json"
done
case $(jq -er '.writer_status' "$layout") in
NOT_RUN)
    refusal="$root/client/refusal-$cv"; verify "$refusal"
    bound "$refusal/inputs.sha256" "$grade/summary.json"
    jq -e 'select(.pass==false and .passing==[]) | {status:"NOT_RUN",reason:"fresh source gate",passing,
        arms:(.arms|map_values([.gate|to_entries[]|select(.value==false)|.key]))}' \
        "$out/grade/summary.json" > "$out/writer.json"
    cmp_file "$out/writer.json" "$refusal/receipt.json"
    ;;
CAPTURED)
    capture="$root/client/capture-$cv"; tokens="$root/client/tokens-$cv"; writer="$root/client/writer-$cv"
    for lane in "$capture" "$tokens" "$writer"; do verify "$lane"; done
    bound "$tokens/inputs.sha256" "$grade/summary.json"
    cmp_file "$out/admission.json" "$capture/selection.json"
    prior="${layout%/*}/ancestry/client"; codec="$prior/tokenizer.json"
    codec_sha=$(digest "$codec")
    model_sha=$(awk 'NR==1 {print $1}' "$capture/artifacts.sha256")
    stock_sha=$(awk 'NR==2 {print $1}' "$capture/artifacts.sha256")
    [[ $model_sha =~ ^[0-9a-f]{64}$ && $stock_sha =~ ^[0-9a-f]{64}$ ]]
    new="$out/client"
    mkdir "$new" "$out/tokens"
    cp "$out/admission.json" "$new/selection.json"
    reference="$panel/held/reference.json"; endpoints="$out/grade/endpoints.jsonl"
    plan_capture "$new" "$reference" "$endpoints" study-model stock-model
    cmp_file "$new/plan.jsonl" "$capture/plan.jsonl"
    while IFS=$'\t' read -r arm cohort _fold id _; do
        safe_member "$arm/$cohort/trace/$id.exit"
        trace="$arm/$cohort/trace/$id"
        cp "$capture/$trace.exit" "$new/$trace.exit"
        if [[ $(cat "$new/$trace.exit") == 0 ]]; then cp "$capture/$trace.json" "$new/$trace.json"; fi
    done < "$new/jobs.tsv"
    collect_capture "$new" "$reference" "$endpoints" study-model stock-model "$model_sha" "$stock_sha"
    jq -e '.source_identity_disagreements==[]' "$new/grade/summary.json" >/dev/null
    for path in "$new/grade/"*.json*; do cmp_file "$path" "$capture/grade/${path##*/}"; done
    grade_tokens "$new" "$out/tokens"
    for member in tokens.jsonl summary.json; do cmp_file "$out/tokens/$member" "$tokens/$member"; done
    check_differential "$capture" "$writer"
    jq -n --arg summary "$(digest "$out/tokens/summary.json")" --arg differential "$(digest "$writer/summary.json")" \
        '{status:"CAPTURED",tokens_sha256:$summary,writer_differential_sha256:$differential}' > "$out/writer.json"
    ;;
*) exit 65 ;;
esac

jq -n --arg train "$(digest "$train/SHA256SUMS")" --arg held "$(digest "$held/SHA256SUMS")" \
    --arg panel "$(digest "$panel/SHA256SUMS")" --arg grade "$(digest "$grade/SHA256SUMS")" \
    --arg release "$(digest "$release/inputs.sha256")" --arg permit "$(digest "$permit/inputs.sha256")" \
    --arg selection "$(digest "$out/train/selection.json")" --arg gate "$(digest "$out/grade/summary.json")" \
    --argjson compared "$(wc -l < "$out/compared.txt")" --slurpfile writer "$out/writer.json" '
    {schema:"uor-r4.codex-4299.cold-restore/1",pass:true,disagreements:0,
     train_sha256:$train,held_sha256:$held,panel_sha256:$panel,grade_sha256:$grade,
     release_sha256:$release,permit_sha256:$permit,selection_sha256:$selection,gate_sha256:$gate,
     independent_train:true,independent_held:true,train_elections:2,train_candidates:36,
     held_sets:2,held_lanes:26,typed_replayed_lanes:8,compared_files:$compared,writer:$writer[0]}' > "$out/receipt.json"
