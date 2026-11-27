#!/usr/bin/env bash
set -euo pipefail
fit=${1:?measured GPU fit directory}
prior=${2:?original 4325 slot schedule}
out=${3:?new receipt directory}
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$out"
jq -n --slurpfile prior "$prior" --slurpfile groups "$fit/schedule.jsonl" -f "$here/matched-schedule.jq" > "$out/exposures.json"
for arm in rowwise paired; do
  jq -n --slurpfile scores "$fit/$arm-train-scores.jsonl" --slurpfile trace "$fit/$arm-slots.jsonl" --slurpfile schedule "$fit/schedule.jsonl" -f "$here/last-pair-update.jq" > "$out/$arm-last.json"
done
