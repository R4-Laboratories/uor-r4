#!/usr/bin/env bash
# Run in a fresh directory on the measured GPU. All paths are relative to it.
set -euo pipefail
umask 077
cd "${1:?frozen run directory}"
go_hash=${2:?GO digest}
trap 'echo $? > qualified.exit' EXIT
test "$(sha256sum go.json | cut -d' ' -f1)" = "$go_hash"
test "$(sha256sum approved.sha256 | cut -d' ' -f1)" = "$(jq -r .approved_sha256 go.json)"
sha256sum --strict --quiet -c approved.sha256
jq -e '.authorized and .test_fitness=="FIT"' go.json >/dev/null
test "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)" = "$(jq -r .device_uuid go.json)"
export CUDA_DISABLE_PTX_JIT=1
date -u +%FT%T.%NZ > fit.started
./ending-device train image/ending.cubin base-full-v1 train.jsonl fit
date -u +%FT%T.%NZ > fit.completed
(cd fit && sha256sum weights.i16le train.json) > fit.sha256
(cd fit && sha256sum -c ../fit.sha256 --quiet)
date -u +%FT%T.%NZ > held.started
./ending-device score image/ending.cubin copy.cubin frozen/occurrence.cubin frozen/baseline.cubin base-full-v1 frozen/token-classes.u8 frozen/weights.i16le frozen/baseline-state/marker.tsv fit/weights.i16le held.jsonl measured
date -u +%FT%T.%NZ > held.completed
(cd measured && sha256sum observations.jsonl reverse.jsonl replay.json) > measured.sha256
