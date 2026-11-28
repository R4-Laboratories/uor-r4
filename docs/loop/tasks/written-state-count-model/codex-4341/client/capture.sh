#!/usr/bin/env bash
# Execute after the registered FIT and dispatch approval, on the registered hardware.
set -euo pipefail
umask 077
[[ $# == 11 ]] || {
    echo 'usage: capture.sh STAGE STAGE_SHA EVIDENCE EVIDENCE_SHA PACK PACK_SHA ARM CLIENT ARTIFACT TOKENIZER FRESH_OUT' >&2
    exit 2
}
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/common.sh"
stage=$(realpath "$1")
evidence=$(realpath "$3")
pack=$(realpath "$5")
resolve_arm "$7"
client=$(realpath "$8")
artifact=$(realpath "$9")
tokenizer=$(realpath "${10}")
out=${11}
sealed "$stage" STAGE.sha256 "$2" bindings.json blind.jsonl positive-blind.jsonl gold.jsonl truth.jsonl
pin "$4" "$evidence"
pin "$6" "$pack"
check_model_binding "$evidence" "$pack" "$stage"
pin "$(jq -er .ordinary_client_sha256 "$evidence")" "$client"
pin "$(jq -er .artifact_sha256 "$pack")" "$artifact"
pin "$(jq -er .tokenizer_sha256 "$evidence")" "$tokenizer"
pin be96b48b1e27a01098ff2f2899e063f1237c44061da5cc503784fc2c0f8d9fe8 "$prior/client/capture.sh"
pin 8e3465b51be867839a092ff2ba73a862d4e777d1c23a239926b485032cd186e5 "$prior/client/intervene.sh"
mkdir "$out"
out=$(realpath "$out")
jq -n --arg stage_sha256 "$2" --arg scores_sha256 "$4" --arg pack_sha256 "$6" --arg arm "$7" \
    '{$stage_sha256,$scores_sha256,$pack_sha256,$arm}' > "$out/bindings.json"

bash "$prior/client/capture.sh" "$client" "$artifact" "$tokenizer" "$stage/blind.jsonl" \
    "$out/ordinary" --native-occurrence-trace
members=(bindings.json ordinary/INPUTS.sha256 ordinary/records.jsonl)
if [[ $7 == natural.geometry.actual ]]; then
    jq -c --slurpfile gold "$stage/gold.jsonl" '
        ($gold|INDEX(.case_id)) as $g | select($g[.case_id]!=null)
    ' "$out/ordinary/records.jsonl" > "$out/positive-records.jsonl"
    for mode in gold-native gold-stop; do
        bash "$prior/client/intervene.sh" "$client" "$artifact" "$tokenizer" \
            "$stage/positive-blind.jsonl" "$stage/gold.jsonl" "$stage/truth.jsonl" \
            "$out/positive-records.jsonl" "$mode" "$out/$mode"
        members+=("$mode/actions.jsonl" "$mode/records.jsonl")
    done
fi
(cd "$out" && sha256 "${members[@]}" > CAPTURE.sha256)
