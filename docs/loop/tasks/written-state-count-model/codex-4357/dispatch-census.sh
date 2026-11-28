#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=${1:?absolute stage}; out=${2:?fresh results}; uuid=${3:?leased GPU UUID}
[[ $stage == /* && $out == /* && $uuid == GPU-* ]]
cd "$stage"
mkdir -m 700 "$out"
exec > "$out/dispatch.log" 2>&1
sample=
finish() {
    local rc=$?
    if [[ -n $sample ]]; then kill "$sample" 2>/dev/null || true; wait "$sample" 2>/dev/null || true; fi
    date -u +%FT%TZ > "$out/end.utc"
    echo "$rc" > "$out/status.tmp"
    mv "$out/status.tmp" "$out/status"
}
trap finish EXIT
date -u +%FT%TZ > "$out/start.utc"
sha256sum --strict -c approved.sha256
(cd input && sha256sum --strict -c MEMBERS.sha256)
nvidia-smi --id="$uuid" --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$out/device.csv"
[[ $(nvidia-smi --id="$uuid" --query-gpu=uuid --format=csv,noheader) == "$uuid" ]]
[[ -z $(nvidia-smi --id="$uuid" --query-compute-apps=pid --format=csv,noheader) ]]
nvidia-smi --id="$uuid" --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader --loop=1 > "$out/gpu-samples.csv" &
sample=$!
CUDA_VISIBLE_DEVICES="$uuid" /usr/bin/time -v -o "$out/usage.txt" bin/uor-midpoint-address-census p4.d11 input protected frozen-sparse.jsonl eligibility.json "$out/trial"
