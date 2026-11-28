#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=${1:?stage root}
out=${2:?fresh results parent}
uuid=${3:?leased device uuid}
cd "$stage"
mkdir -m 700 "$out"
exec > >(tee "$out/dispatch.log") 2>&1
trap 'rc=$?; date -u +%FT%TZ > "$out/end.utc"; echo "$rc" > "$out/status"' EXIT
date -u +%FT%TZ > "$out/start.utc"
sha256sum -c approved.sha256
(cd custody && sha256sum -c ALL.sha256)
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "$uuid"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader > "$out/device.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader --loop=1 > "$out/gpu-samples.csv" &
sample=$!
trap 'rc=$?; kill "$sample" 2>/dev/null || true; date -u +%FT%TZ > "$out/end.utc"; echo "$rc" > "$out/status"' EXIT
/usr/bin/time -v -o "$out/usage.txt" ./bin/uor-lexical-fit run p4.d11 custody "$out/trial"
