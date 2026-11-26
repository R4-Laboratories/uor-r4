#!/usr/bin/env bash
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-fixed-assignment-median
m=$(realpath "${1:?train}"); h=$(realpath "${2:?held}"); g=$(realpath "${3:?grade}"); c=$(realpath "${4:?client receipt}"); out=${5:?new report}
[[ $(hostname -s) == wick && $m == "$s"/train-* && $h == "$s"/held-* && $g == "$s"/grade-* && $c == "$s"/client/* && $out == "$s"/report-v* ]]
for dir in "$m" "$h" "$g" "$c" "$m-lease" "$h-lease"; do (cd "$dir" && sha256sum --strict --quiet -c SHA256SUMS); done
mkdir "$out"
report_dir=$(cd "$(dirname "$0")" && pwd -P)
jq --arg view train -f "$report_dir/train-view.jq" "$m/selection.json" > "$out/train.json"
jq -r --arg view protected -f "$report_dir/train-view.jq" "$m/selection.json" > "$out/protected-near.tsv"
for arm in endpoint ordered; do
    cp "$m/$arm/candidates.jsonl" "$out/$arm.train-candidates.jsonl"
    cp "$m/device/$arm/medians.jsonl" "$out/$arm.medians.jsonl"
done
for domain in held contrast; do
    cp "$g/$domain/counts.json" "$out/$domain.json"; cp "$g/$domain/movements.json" "$out/$domain-movements.json"
done
for lane in baseline ordered_base endpoint ordered endpoint_full ordered_full far_only correct_link exchanged_link; do
    for dist in near far; do
        jq -nc --arg lane "$lane" --arg dist "$dist" --slurpfile data "$g/held/$lane.first-$dist.json" \
            '{lane:$lane,distance:$dist,first:$data[0]}' >> "$out/first-sources.jsonl"
    done
    jq -c --arg arm "$lane" '{arm:$arm,pairs,full,correct_source_flips,distinct_reads,pairs_with_abstention}' \
        "$g/contrast/$lane.flips.json" >> "$out/first-later.jsonl"
done
cp "$g/summary.json" "$out/source-gate.json"
if jq -e '.pass' "$g/summary.json" >/dev/null; then
    [[ $c == "$s"/client/tokens-* ]]; cp "$c/summary.json" "$out/writer.json"
else
    [[ $c == "$s"/client/refusal-* ]]; cp "$c/receipt.json" "$out/writer.json"
fi
jq -n --arg fit "$(cat "$m/fitter.time")" --arg train "$(cat "$m-lease/resources.txt")" --arg held "$(cat "$h-lease/resources.txt")" \
    --arg gpu "$(cat "$m-lease/device.txt")" --argjson train_bytes "$(du -sb "$m"|cut -f1)" --argjson held_bytes "$(du -sb "$h"|cut -f1)" \
    --argjson allocated "$(du -sB1 "$s"|cut -f1)" '{fitter:$fit,train:$train,held:$held,gpu:$gpu,train_bytes:$train_bytes,held_bytes:$held_bytes,
       allocated_before_archive:$allocated,training_rows:256,fixed_assignments:512,state_bytes:13336,tagged_bytes:13352,
       scored_train_states_per_arm:18,held_arms_per_panel:13,primary_prompts:256,contrast_prompts:64}' > "$out/resources.json"
sha256sum "$m/SHA256SUMS" "$h/SHA256SUMS" "$g/SHA256SUMS" "$c/SHA256SUMS" "$m-lease/SHA256SUMS" "$h-lease/SHA256SUMS" "$0" "$report_dir/train-view.jq" > "$out/inputs.sha256"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

