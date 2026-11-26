#!/usr/bin/env bash
# Reconstruct the training margins and the held decision from the downloaded raw reads.
set -euo pipefail
umask 077
s=/mnt/plexpool/uor-r4-corpus/runs/20261126-codex-far-transport-margin
root=$(realpath "${1:?restored archive root}")
meas=${2:?measurement version}
grade=${3:?grade version}
out=${4:?new verification output under the study}
[[ $(hostname -s) == wick && $root == "$s"/restore-v*/extracted && $out == "$s"/restored-grades-v* ]]
[[ $meas =~ ^v[1-9][0-9]*$ && $grade =~ ^v[1-9][0-9]*$ ]]
mkdir "$out"
method="$root/methods-v1/grade"
p="$root/panel/panel-v1"
g="$root/grade-$grade"
m="$root/measurement-$meas"
for arm in contrastive positive; do
    jq -nce --slurpfile reference "$p/train/reference.json" --slurpfile coords "$p/train/blind/coordinates.jsonl" \
       -f "$method/loss-cases.jq" "$m/train/$arm.first.trace" > "$out/$arm.edges.jsonl"
    jq -es -f "$method/loss-summary.jq" "$out/$arm.edges.jsonl" > "$out/$arm.loss.json"
    cmp "$out/$arm.edges.jsonl" "$g/$arm.edges.jsonl"
    cmp "$out/$arm.loss.json" "$g/$arm.loss.json"
done
for split in train held; do
    d="$m/$split"
    jq -nce --slurpfile reference "$p/$split/reference.json" --slurpfile coords "$p/$split/blind/coordinates.jsonl" \
       --slurpfile native "$d/native.jsonl" --slurpfile unfitted "$d/unfitted.jsonl" --slurpfile learned "$d/learned.jsonl" \
       --slurpfile gap "$d/gap.jsonl" --slurpfile correct "$d/correct.jsonl" --slurpfile exchange "$d/exchanged.jsonl" \
       --slurpfile base "$d/baseline.jsonl" --slurpfile pos "$d/positive.jsonl" -f "$method/rows.jq" > "$out/$split.jsonl"
    jq -es --arg split "$split" -f "$method/source-scores.jq" "$out/$split.jsonl" > "$out/$split.scores.json"
    cmp "$out/$split.jsonl" "$g/$split.jsonl"
    cmp "$out/$split.scores.json" "$g/$split.scores.json"
done
jq -c '.arms |= with_entries(select(.key|IN("native","baseline","learned","correct_link","exchanged_link")))' \
   "$out/held.jsonl" > "$out/endpoints.jsonl"
sha(){ sha256sum "$1" | cut -d' ' -f1; }
jq -e --slurpfile training "$out/contrastive.loss.json" --arg reference_sha "$(sha "$p/held/reference.json")" \
   --arg writer_sha "$(sha "$out/endpoints.jsonl")" --arg source_sha "$(sha "$out/held.jsonl")" \
   -f "$method/release.jq" "$out/held.scores.json" > "$out/summary.json"
jq -es --slurpfile reference "$p/held/reference.json" -f "$method/trace.jq" "$out/held.jsonl" > "$out/held-errors.json"
for f in endpoints.jsonl summary.json held-errors.json; do cmp "$g/$f" "$out/$f"; done
(cd "$out" && sha256sum *.jsonl *.json > SHA256SUMS)
