#!/usr/bin/env bash
# Work in a private, checksum-verified run root. The external lease manager owns GPU 0.
set -euo pipefail
umask 077
plan=$1
go=$2
go_sha=$3
key=$4
out=$5
test ! -e "$out"
mkdir "$out"
on_exit() {
    code=$?
    printf '%s\n' "$code" > "$out/exit.tmp"
    mv "$out/exit.tmp" "$out/exit"
}
trap on_exit EXIT
export CUDA_DISABLE_PTX_JIT=1
test "$(sha256sum "$go" | cut -d' ' -f1)" = "$go_sha"
jq -e '.authorized and .fit_verdict=="TEST FITNESS: FIT"' "$go" >/dev/null
test "$(sha256sum approved-files.sha256 | cut -d' ' -f1)" = "$(jq -er .approved_files_sha256 "$go")"
sha256sum -c approved-files.sha256
test "$(sha256sum "$key" | cut -d' ' -f1)" = "$(jq -er .blind_mapping_sha256 "$go")"
date -u +%FT%TZ > "$out/started"
bin/visible-answer-train fit "$plan" "$go" "$go_sha" "$out/train" > "$out/fit.log" 2>&1
jq -e '
  .matched_budget and (.arms|length==2)
  and all(.arms[]; .accepted_updates>0 and .committed_moves>0 and
    .all_19_sections_native_round_trip and (.native_section_sha256|length)==19 and
    ([.learned_geometry_changed_words[]]|add)>0 and .full_targets==13983847 and
    .geometry_targets==259604 and .batches==1015)
  and (.different_serialized_native_sections|type)=="array"
' "$out/train/training-pair.json" >/dev/null
date -u +%FT%TZ > "$out/trained"
for arm in reasoning-first answer-first; do
    code=$(jq -er --arg arm "$arm" '.[$arm]' "$key")
    artifact="$out/train/$arm/artifact.d11"
    want=$(jq -er .artifact_d11_sha256 "$out/train/$arm/training-receipt.json")
    test "$(sha256sum "$artifact" | cut -d' ' -f1)" = "$want"
    bin/visible-answer-eval-pod generate "$artifact" "$code" panel/prompts.jsonl "$out/generate-$code"
done
for arm in reasoning-first answer-first; do
    code=$(jq -er --arg arm "$arm" '.[$arm]' "$key")
    bin/visible-answer-eval-pod teacher "$out/train/$arm/artifact.d11" "$code" panel/teacher.jsonl "$out/teacher-$code"
done
date -u +%FT%TZ > "$out/completed"
