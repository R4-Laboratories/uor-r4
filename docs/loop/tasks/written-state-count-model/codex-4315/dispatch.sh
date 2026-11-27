#!/usr/bin/env bash
# Stage only original inputs and run while the named GPU lease is held.
set -euo pipefail
umask 077
cd "${1:?private experiment directory}"
expected=${2:?GO SHA-256}
finish() { rc=$?; echo "$rc" > dispatch.exit.tmp; mv dispatch.exit.tmp dispatch.exit; }
trap finish EXIT
test "$(sha256sum go.json | cut -d' ' -f1)" = "$expected"
test "$(sha256sum approved.sha256 | cut -d' ' -f1)" = "$(jq -r .approved_sha256 go.json)"
sha256sum --quiet --strict -c approved.sha256
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .device_uuid go.json)"
jq -e '.authorized and .fit=="TEST FITNESS: FIT"' go.json >/dev/null
jq -e '.status=="pass" and (.controls|length)==13' finite/finite.json >/dev/null
export CUDA_DISABLE_PTX_JIT=1
stamp() { date -u +%FT%T.%NZ; }
stamp > fit.started
./driver fit image/occurrence.cubin base-full-v1 product-inputs-v1/token-classes.u8 common/donor.jsonl panel-v1/train/rows.jsonl panel-v1/train-absent/rows.jsonl fit
(cd fit && sha256sum weights.i16le fit.json > SHA256SUMS)
stamp > fit.completed
./common/baseline score common/baseline.cubin base-v1 qualified-panel-v1/rows.jsonl common/baseline-state marker-out
stamp > marker.completed
./driver score image/occurrence.cubin base-full-v1 product-inputs-v1/token-classes.u8 qualified-panel-v1/rows.jsonl marker-out/marker.jsonl fit held
(cd held && sha256sum observations.jsonl score.json > SHA256SUMS)
(cd marker-out && sha256sum colon.jsonl marker.jsonl > SHA256SUMS)
stamp > held.completed
