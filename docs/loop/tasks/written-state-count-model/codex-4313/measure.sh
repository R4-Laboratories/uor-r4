#!/usr/bin/env bash
# Run inside the holder's GPU lease, using the frozen panel and unchanged artifacts.
set -euo pipefail
umask 077
go_hash=${1:?expected pre-run decision SHA-256}
out=${2:?new output path}
hash() { sha256sum "$1" | cut -d' ' -f1; }
test "$(hash go.json)" = "$go_hash"
jq -e '.authorized and .fit=="TEST FITNESS: FIT"' go.json > /dev/null
test "$(hash approved.sha256)" = "$(jq -r .approved_sha256 go.json)"
sha256sum --strict --quiet -c approved.sha256
test "$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .device_uuid go.json)"
jq -e '.status=="pass" and .rows==6 and (.controls|length)>=11' finite-v1.json > /dev/null
mkdir "$out"
finish() {
    rc=$?
    printf '%s\n' "$rc" > "$out/exit.tmp"
    mv "$out/exit.tmp" "$out/exit"
}
trap finish EXIT
export CUDA_DISABLE_PTX_JIT=1
export CUDA_FORCE_PTX_JIT=0
date -u +%FT%T.%NZ > "$out/started"
for arm in reasoning-first answer-first; do
    ./trace-v1 gpu "$arm/artifact.d11" rows.jsonl "$out/$arm.jsonl"
done
date -u +%FT%T.%NZ > "$out/completed"
(cd "$out" && sha256sum reasoning-first.jsonl answer-first.jsonl started completed > SHA256SUMS)
