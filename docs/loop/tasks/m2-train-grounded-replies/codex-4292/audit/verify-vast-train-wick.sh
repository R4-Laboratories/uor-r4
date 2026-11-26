#!/usr/bin/env bash
# Compare a finished provider batch to the separately sealed production grade.
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-protected-transport"
previous="$runs/20261126-codex-mixed-transport"
original="$previous/measurement-v1/train/mixed"
here=$(cd "$(dirname "$0")" && pwd)
capture=$(realpath "${1:?restored provider root}")
job=$(realpath "${2:?restored provider job log}")
grade=$(realpath "${3:?sealed provider grade}")
audit=$(realpath "${4:?sealed independent raw batch}")
version=${5:?new supplemental audit version}
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
[[ $capture == "$s"/fit/vast-* && $job == "$capture/"* && $grade == "$s"/fit/vast-grade-v* &&
   $audit == "$s/audit/batch-v"* && $version =~ ^v[1-9][0-9]*$ ]]
raw="$capture/train/root/device"
out="$s/audit/vast-train-$version"
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
for dependency in "$capture/train" "$raw" "$capture/receipts/train" "$grade" "$audit"; do
    verify "$dependency"
done
(cd "$capture" && sha256sum --strict --quiet -c train/authority.sha256)
sha256sum --strict --quiet -c "$audit/inputs.sha256"
grep -Fxq "$(sha256sum "$raw/SHA256SUMS")" "$audit/inputs.sha256"
grep -Eq '^# uor-pod job by=codex/codex-4292-vast-.+ pod=vast-[0-9]+ gpu=0 queued' "$job"
grep -Fq '(holding /root/gpu0.lock)' "$job"
grep -Eq '^# exit=0 end=' "$job"
cmp "$capture/train/initial/reader.jsonl" "$original/step-0.raw.jsonl"
cmp "$capture/train/initial/edges.jsonl" "$original/initial.jsonl"
mkdir "$out"
jq -e --slurpfile initial "$original/step-0.pairs.json" '
    .pass and .candidate_count==246 and (.metrics[0] as $b |
    $b.near_full==([$initial[0][]|select(.distance=="near" and .full)|.pair]|sort) and
    $b.far_full==([$initial[0][]|select(.block=="common" and .full)|.pair]|sort))
' "$audit/readback.json" > "$out/premise.json"
jq -e --slurpfile origin "$audit/readback.json" --slurpfile grade "$grade/candidates.jsonl" \
    --slurpfile device "$grade/election.json" -f "$here/election.jq" "$audit/readback.json" \
    > "$out/verified.json"
sha256sum "$capture/train/SHA256SUMS" "$raw/SHA256SUMS" "$capture/receipts/train/SHA256SUMS" \
    "$grade/SHA256SUMS" "$job" "$audit/SHA256SUMS" "$here/election.jq" "$here/verify-vast-train-wick.sh" \
    > "$out/inputs.sha256"
(cd "$out" && sha256sum -- *.json inputs.sha256 > SHA256SUMS)
