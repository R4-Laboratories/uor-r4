#!/usr/bin/env bash
# Verify and project completed ordinary-client captures without running a model.
set -euo pipefail
umask 077
[[ $# == 10 ]] || {
    echo 'usage: verify.sh STAGE STAGE_SHA EVIDENCE EVIDENCE_SHA PACK PACK_SHA ARM CAPTURE CAPTURE_SHA FRESH_OUT' >&2
    exit 2
}
here=$(cd "$(dirname "$0")" && pwd -P)
source "$here/common.sh"
stage=$(realpath "$1")
evidence=$(realpath "$3")
pack=$(realpath "$5")
resolve_arm "$7"
capture=$(realpath "$8")
out=${10}
sealed "$stage" STAGE.sha256 "$2" bindings.json cases.json
pin "$4" "$evidence"
pin "$6" "$pack"
check_model_binding "$evidence" "$pack" "$stage"
members=(bindings.json ordinary/INPUTS.sha256 ordinary/records.jsonl)
if [[ $7 == natural.geometry.actual ]]; then
    members+=(gold-native/actions.jsonl gold-native/records.jsonl gold-stop/actions.jsonl gold-stop/records.jsonl)
fi
sealed "$capture" CAPTURE.sha256 "$9" "${members[@]}"
jq -e --arg stage "$2" --arg score "$4" --arg pack "$6" --arg arm "$7" '
    .=={stage_sha256:$stage,scores_sha256:$score,pack_sha256:$pack,arm:$arm}
' "$capture/bindings.json" >/dev/null
pin "$grader" "$prior/data/client-grade.jq"
mkdir "$out"
# Use the original validation definitions, without their historical panel entry point.
# The digest checked above binds all original code, including status_by_text.
sed '/^grades(\.) as \$rows/,$d' "$prior/data/client-grade.jq" > "$out/verifier.jq"
cat "$here/verify.jq" >> "$out/verifier.jq"

verify_mode() {
    local mode=$1 row hash
    while IFS= read -r row || [[ -n $row ]]; do
        hash=$(printf '%s\n' "$row" | sha256)
        jq -ce --arg hash "${hash%% *}" '
            if keys==["case_id","client"] then .+{capture_sha256:$hash}
            else error("invalid raw capture wrapper") end
        ' <<< "$row"
    done < "$capture/$mode/records.jsonl" |
        jq -ces --arg mode "$mode" --arg key "$key" --arg tokenizer "$(jq -er .tokenizer_sha256 "$evidence")" \
           --arg arm unused --arg gold_sha unused --arg truth_sha unused \
           --argjson gold '[]' --argjson truth '[]' --argjson visible '[]' --argjson decisions '[]' \
           --slurpfile cases "$stage/cases.json" --slurpfile model <(model_reference "$evidence") \
           --slurpfile pack "$pack" -f "$out/verifier.jq" > "$out/$mode.json"
}

verify_mode ordinary
receipt=(ordinary.json verifier.jq)
if [[ $7 == natural.geometry.actual ]]; then
    verify_mode gold-native
    verify_mode gold-stop
    jq -en --slurpfile actions "$capture/gold-native/actions.jsonl" \
        --slurpfile forced "$capture/gold-stop/actions.jsonl" \
        --slurpfile cases "$stage/cases.json" '
        $actions==$forced and
        $actions==[$cases[0][]|select(.target.kind=="source") |
          {case_id,text,peer:null,control:(.target.span+{ids:.target.source_ids,prompt_ids:.prompt_ids})}]
    ' >/dev/null
    receipt+=(gold-native.json gold-stop.json)
fi

jq -n --arg stage_sha256 "$2" --arg scores_sha256 "$4" --arg pack_sha256 "$6" \
    --arg capture_sha256 "$9" --arg grader_sha256 "$grader" --arg arm "$7" \
    --arg roster_sha256 "$(jq -er .roster_sha256 "$stage/bindings.json")" \
    --arg artifact_sha256 "$(jq -er .artifact_sha256 "$pack")" \
    '{$stage_sha256,$scores_sha256,$pack_sha256,$capture_sha256,$grader_sha256,$arm,$roster_sha256,$artifact_sha256}' \
    > "$out/trace.json"
(cd "$out" && sha256 trace.json "${receipt[@]}" > TRACE.sha256)
