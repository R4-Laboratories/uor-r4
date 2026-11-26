#!/usr/bin/env bash
set -euo pipefail
umask 077
export LC_ALL=C
runs=/mnt/plexpool/uor-r4-corpus/runs
s="$runs/20261126-codex-protected-transport"
here=$(cd "$(dirname "$0")" && pwd)
copied=$(realpath "${1:?restored provider root}")
stage=$(realpath "${2:?Wick AOT stage directory}")
job=$(realpath "${3:?restored provider job log}")
digest=${4:?exact provider finite manifest SHA}
pod=${5:?provider pod}
session=${6:?provider lease session}
build=${7:?independent replay build version}
version=${8:?unused Wick audit version}
[[ $(hostname -s) == wick && $(findmnt -no FSTYPE --target "$s") == zfs ]]
[[ $copied == "$s"/fit/vast-* && $stage == "$s"/fit/vast-stage-v* && $job == "$copied/"* ]]
[[ $digest =~ ^[0-9a-f]{64}$ && $pod =~ ^vast-[0-9]+$ && $session =~ ^codex-4292-vast-[a-z0-9-]+$ ]]
for id in "$build" "$version"; do [[ $id =~ ^v[1-9][0-9]*$ ]]; done
f="$copied/finite"
r="$copied/receipts/finite"
payload="$stage/payload"
o="$s/audit/vast-finite-$version"
verify() (cd "$1" && sha256sum --strict --quiet -c SHA256SUMS)
sha() { sha256sum "$1" | cut -d' ' -f1; }
pin() { printf '%s  %s\n' "$1" "$2" | sha256sum --strict --quiet -c -; }
verify "$stage"; verify "$payload"; verify "$f"; verify "$r"
pin "$digest" "$f/SHA256SUMS"
(cd "$copied" && sha256sum --strict --quiet -c finite/lease.sha256)
grep -Fq "# uor-pod job by=codex/$session pod=$pod gpu=0 queued" "$job"
grep -Fq '(holding /root/gpu0.lock)' "$job"
grep -Eq '^# exit=0 end=' "$job"
while read -r hash remote; do
    [[ $remote == /workspace/codex-4292-vast/payload/* ]]
    local_input="$payload/${remote#/workspace/codex-4292-vast/payload/}"
    pin "$hash" "$local_input"
done < "$f/inputs.sha256"
[[ ! -s $r/compute-processes.txt ]]
grep -Fq 'run under the inherited uor-pod GPU 0 lease on descriptor 20' "$r/unleased.stderr"
grep -Fq 'uor-pod GPU 0 descriptor does not hold the exclusive lease' "$r/wrong-descriptor.stderr"
if grep -Fq 'not found' "$r/loader.txt"; then exit 1; fi
mkdir "$o"
jq -e --slurpfile lease "$r/lease.json" --arg pod "$pod" --arg session "$session" \
    --arg l "$(sha "$r/lease.json")" --arg p "$(sha "$payload/SHA256SUMS")" '
    . as $a | $lease[0] as $j |
    $a.schema=="uor-r4.codex-4292.vast-lease/1" and $a.phase=="finite" and $a.lab=="codex" and
    $a.pod==$pod and $a.session==$session and $a.gpu==0 and $a.fd==20 and $a.held and
    $a.lock=="/root/gpu0.lock" and $a.payload_sha256==$p and $a.lease_sha256==$l and
    $j.lab=="codex" and $j.pod==$pod and $j.session==$session and $j.gpus==[0] and
    ($j.expires|fromdateiso8601)>($a.at|fromdateiso8601)
' "$r/admission.json" > "$o/lease-check.json"
jq -eRs '
    split("\n") | map(select(length>0)|split(", ")) |
    length==1 and .[0][0]=="0" and .[0][1]=="NVIDIA GeForce RTX 4090" and
    (.[0][2]|startswith("GPU-")) and .[0][3]=="595.99.02" and .[0][4]=="8.9"
' "$r/device.csv" > "$o/device-check.json"
"$here/finite-wick.sh" "$f" "$build" "$version"
finite="$s/audit/finite-$version"
verify "$finite"
jq -e '.pass and .same_fresh_retry and .rejected_tampered_edge and .candidate_count_per_batch==6' \
    "$finite/checks.json" > "$o/finite-check.json"
sha256sum "$stage/SHA256SUMS" "$payload/SHA256SUMS" "$f/SHA256SUMS" "$r/SHA256SUMS" \
    "$finite/SHA256SUMS" "$job" "$here/verify-vast-finite-wick.sh" > "$o/inputs.sha256"
(cd "$o" && sha256sum -- *.json inputs.sha256 > SHA256SUMS)
