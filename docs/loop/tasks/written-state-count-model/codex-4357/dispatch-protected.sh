#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=${1:?stage}; out=${2:?new attempt}; gpu=${3:?leased GPU UUID}
[[ $stage == /* && $out == /* && $gpu == GPU-* ]]
cd "$stage"
mkdir -m700 "$out"
exec > "$out/dispatch.log" 2>&1
finish() {
    local rc=$?
    date -u +%FT%TZ > "$out/end.utc"
    echo "$rc" > "$out/status.tmp"
    mv "$out/status.tmp" "$out/status"
}
trap finish EXIT
date -u +%FT%TZ > "$out/start.utc"
sha256sum --strict -c approved.sha256
(cd input && sha256sum --strict -c MEMBERS.sha256)
nvidia-smi --id="$gpu" --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$out/device.csv"
[[ $(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader) == "$gpu" ]]
[[ -z $(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader) ]]
CUDA_VISIBLE_DEVICES="$gpu" bin/uor-midpoint-address protected frozen/trunk.bin b75fb885ec23ef43c7f248b1fda89b9d75841d8ff5e269129f5d565df302ecf3 input protected "$out/trial"
