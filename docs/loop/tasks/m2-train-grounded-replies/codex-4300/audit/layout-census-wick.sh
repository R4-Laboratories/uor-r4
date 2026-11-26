#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
study="$runs/20261126-codex-switch-source-competition"
audit="$study/audit"
here=$(cd "$(dirname "$0")" && pwd)
layout=$(realpath -e "${1:?sealed diagnostic layout}")
out=$(realpath -m "${2:?new independent layout census receipt}")
primary="$study/panel/panel-v1"
registration="$study/layout-registration-v1"
grade="$study/grade-v1"
reader="$audit/build-v4"
origin="$audit/replay-v1"
codec="$runs/20261125-codex-bound-instruction-state/custody-v1/signed/tokenizer.json"
[[ $# == 2 && $(hostname -s) == wick && $(findmnt -n -o FSTYPE --target "$study") == zfs &&
   $here == "$audit"/recipe-v* && $layout == "$study"/matched-train/panel-v* &&
   $out == "$audit"/layout-census-v* ]]
for directory in "$layout" "$primary" "$registration" "$grade" "$reader" "$origin"; do
    (cd "$directory" && sha256sum --strict --quiet -c SHA256SUMS)
done
mkdir "$out"
"$reader/switch-audit" census "$layout" "$codec" "$out/packet.json"
"$reader/switch-audit" geometry "$layout" "$codec" "$layout/endpoint/text.bin" "$out/geometry.json"
jq -ne --slurpfile original "$primary/train-original/reference.json" --slurpfile historical "$primary/historical/reference.json" \
   --slurpfile derived "$primary/train-derived/reference.json" --slurpfile refs "$layout/reference.json" \
   --slurpfile historic_requests "$primary/historical/requests.jsonl" --slurpfile req "$layout/requests.jsonl" \
   --slurpfile primary "$grade/summary.json" --slurpfile primary_before "$origin/train-derived-before.json" \
   -f "$here/layout-preflight.jq" > "$out/readback.json"
sha256sum "$0" "$here/layout-preflight.jq" "$codec" "$layout/SHA256SUMS" "$primary/SHA256SUMS" \
   "$registration/SHA256SUMS" "$grade/SHA256SUMS" "$reader/SHA256SUMS" "$origin/SHA256SUMS" > "$out/inputs.sha256"
(
    cd "$out"
    find . -type f ! -name .seal.pending -print0 | sort -z | xargs -0 -r sha256sum > .seal.pending
    mv .seal.pending SHA256SUMS
)
